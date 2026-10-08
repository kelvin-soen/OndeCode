//! Terminal UI. It is a regular ACP client: it launches this same binary with `--acp` as the
//! agent subprocess and renders the session with ratatui, so it exercises exactly the protocol
//! path an editor would.

use std::path::{Path, PathBuf};
use std::time::Duration;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, ClientCapabilities, ContentBlock, FileSystemCapabilities,
    InitializeRequest, NewSessionRequest, PermissionOptionKind, PromptRequest, ReadTextFileRequest,
    ReadTextFileResponse, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionId, SessionNotification,
    SessionUpdate, StopReason, TextContent, ToolCallContent, ToolCallId, ToolCallStatus,
    WriteTextFileRequest, WriteTextFileResponse,
};
use agent_client_protocol::{AcpAgent, AcpAgentConfig, Agent, ConnectionTo};
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures::StreamExt;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::{mpsc, oneshot};

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const MAX_TOOL_OUTPUT_LINES: usize = 8;

enum AppEvent {
    Update(SessionUpdate),
    Permission(
        RequestPermissionRequest,
        oneshot::Sender<RequestPermissionOutcome>,
    ),
    TurnDone(Result<StopReason, String>),
}

pub async fn run(yolo: bool, extra_roots: Vec<PathBuf>) -> anyhow::Result<()> {
    let mut config = AcpAgentConfig::new(std::env::current_exe()?).arg("--acp");
    config = config.env("ONDE_CODE_SURFACE", "tui");
    if yolo {
        config = config.arg("--yolo");
    }
    let cwd = std::env::current_dir()?;
    let llm = crate::llm::LlmConfig::from_env();
    let model = format!("{} · {}", llm.provider.name(), llm.model);
    let (tx, rx) = mpsc::unbounded_channel();

    agent_client_protocol::Client
        .builder()
        .on_receive_notification(
            {
                let tx = tx.clone();
                async move |n: SessionNotification, _cx| {
                    let _ = tx.send(AppEvent::Update(n.update));
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            {
                let tx = tx.clone();
                async move |req: RequestPermissionRequest, responder, cx| {
                    // Wait for the user's answer off the dispatch loop so updates keep flowing.
                    let (answer_tx, answer_rx) = oneshot::channel();
                    let _ = tx.send(AppEvent::Permission(req, answer_tx));
                    cx.spawn(async move {
                        let outcome = answer_rx
                            .await
                            .unwrap_or(RequestPermissionOutcome::Cancelled);
                        responder.respond(RequestPermissionResponse::new(outcome))
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        // ACP v1 fs/read_text_file: serve file contents from the session cwd,
        // honoring the optional 1-based `line` and `limit` params.
        .on_receive_request(
            async move |req: ReadTextFileRequest, responder, _cx| match tokio::task::spawn_blocking(
                {
                    let path = req.path.clone();
                    move || read_text_file_blocking(&path, req.line, req.limit)
                },
            )
            .await
            {
                Ok(Ok(r)) => responder.respond(r),
                Ok(Err(e)) => responder.respond_with_error(e),
                Err(e) => responder.respond_with_internal_error(e.to_string()),
            },
            agent_client_protocol::on_receive_request!(),
        )
        // ACP v1 fs/write_text_file: the client MUST create the file (and parent
        // directories) if it doesn't exist.
        .on_receive_request(
            async move |req: WriteTextFileRequest, responder, _cx| {
                match tokio::task::spawn_blocking({
                    let path = req.path.clone();
                    let content = req.content.clone();
                    move || write_text_file_blocking(&path, &content)
                })
                .await
                {
                    Ok(Ok(r)) => responder.respond(r),
                    Ok(Err(e)) => responder.respond_with_error(e),
                    Err(e) => responder.respond_with_internal_error(e.to_string()),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(
            AcpAgent::new(config),
            |conn: ConnectionTo<Agent>| async move {
                conn.send_request(
                    InitializeRequest::new(ProtocolVersion::V1).client_capabilities(
                        ClientCapabilities::new().fs(FileSystemCapabilities::new()
                            .read_text_file(true)
                            .write_text_file(true)),
                    ),
                )
                .block_task()
                .await?;
                let session = conn
                    .send_request(
                        NewSessionRequest::new(cwd.clone())
                            .additional_directories(extra_roots.clone()),
                    )
                    .block_task()
                    .await?;
                let mut app = App::new(conn, session.session_id, tx, model, cwd, extra_roots);
                let mut terminal = ratatui::init();
                let result = app.run(&mut terminal, rx).await;
                ratatui::restore();
                result.map_err(|e| {
                    agent_client_protocol::Error::internal_error().data(format!("{e:#}"))
                })
            },
        )
        .await?;
    Ok(())
}

/// `fs/read_text_file` handler body: reads `path`, applying the spec's optional
/// 1-based `line` and `limit` params.
fn read_text_file_blocking(
    path: &Path,
    line: Option<u32>,
    limit: Option<u32>,
) -> agent_client_protocol::Result<ReadTextFileResponse> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            agent_client_protocol::Error::resource_not_found(Some(path.display().to_string()))
        } else {
            agent_client_protocol::Error::internal_error().data(e.to_string())
        }
    })?;
    let skip = line.map_or(0, |l| l.saturating_sub(1) as usize);
    let take = limit.map_or(usize::MAX, |l| l as usize);
    let content = content
        .lines()
        .skip(skip)
        .take(take)
        .collect::<Vec<_>>()
        .join("\n");
    Ok(ReadTextFileResponse::new(content))
}

/// `fs/write_text_file` handler body: the client MUST create the file if it
/// doesn't exist (spec: https://agentclientprotocol.com/protocol/v1/file-system).
fn write_text_file_blocking(
    path: &Path,
    content: &str,
) -> agent_client_protocol::Result<WriteTextFileResponse> {
    let write = || {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    };
    write().map_err(|e| agent_client_protocol::Error::internal_error().data(e.to_string()))?;
    Ok(WriteTextFileResponse::new())
}

enum Entry {
    User(String),
    Agent(String),
    Thought(String),
    Tool {
        id: ToolCallId,
        title: String,
        status: ToolCallStatus,
        output: String,
    },
    Info(String),
    Error(String),
}

struct App {
    conn: ConnectionTo<Agent>,
    session_id: SessionId,
    tx: mpsc::UnboundedSender<AppEvent>,
    model: String,
    /// Workspace roots, primary working directory first, then any additional roots.
    roots: Vec<PathBuf>,
    entries: Vec<Entry>,
    input: Vec<char>,
    cursor: usize,
    busy: bool,
    permission: Option<(
        RequestPermissionRequest,
        oneshot::Sender<RequestPermissionOutcome>,
    )>,
    /// Lines scrolled up from the bottom of the transcript; 0 follows new output.
    scroll_up: usize,
    tick: usize,
    quit: bool,
}

impl App {
    fn new(
        conn: ConnectionTo<Agent>,
        session_id: SessionId,
        tx: mpsc::UnboundedSender<AppEvent>,
        model: String,
        cwd: PathBuf,
        extra_roots: Vec<PathBuf>,
    ) -> Self {
        let roots = std::iter::once(cwd.clone())
            .chain(extra_roots.into_iter().filter(|r| *r != cwd))
            .collect();
        Self {
            conn,
            session_id,
            tx,
            model,
            roots,
            entries: vec![Entry::Info(
                "Ask for a change, or type /quit to exit.".into(),
            )],
            input: Vec::new(),
            cursor: 0,
            busy: false,
            permission: None,
            scroll_up: 0,
            tick: 0,
            quit: false,
        }
    }

    async fn run(
        &mut self,
        terminal: &mut DefaultTerminal,
        mut rx: mpsc::UnboundedReceiver<AppEvent>,
    ) -> anyhow::Result<()> {
        let mut keys = EventStream::new();
        let mut ticker = tokio::time::interval(Duration::from_millis(80));
        while !self.quit {
            terminal.draw(|f| self.draw(f))?;
            tokio::select! {
                ev = keys.next() => match ev {
                    Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => self.on_key(key),
                    Some(Err(e)) => return Err(e.into()),
                    None => break,
                    _ => {}
                },
                Some(ev) = rx.recv() => self.on_event(ev),
                _ = ticker.tick(), if self.busy => self.tick += 1,
            }
        }
        Ok(())
    }

    fn on_event(&mut self, ev: AppEvent) {
        match ev {
            AppEvent::Update(update) => self.on_update(update),
            AppEvent::Permission(req, answer) => self.permission = Some((req, answer)),
            AppEvent::TurnDone(result) => {
                self.busy = false;
                match result {
                    Ok(StopReason::EndTurn) => {}
                    Ok(StopReason::Cancelled) => {
                        self.entries.push(Entry::Info("Cancelled.".into()))
                    }
                    Ok(other) => self
                        .entries
                        .push(Entry::Info(format!("Stopped: {other:?}"))),
                    Err(e) => self.entries.push(Entry::Error(e)),
                }
            }
        }
    }

    fn on_update(&mut self, update: SessionUpdate) {
        match update {
            SessionUpdate::AgentMessageChunk(chunk) => match self.entries.last_mut() {
                Some(Entry::Agent(text)) => text.push_str(&block_text(&chunk.content)),
                _ => self.entries.push(Entry::Agent(block_text(&chunk.content))),
            },
            SessionUpdate::AgentThoughtChunk(chunk) => match self.entries.last_mut() {
                Some(Entry::Thought(text)) => text.push_str(&block_text(&chunk.content)),
                _ => self
                    .entries
                    .push(Entry::Thought(block_text(&chunk.content))),
            },
            SessionUpdate::ToolCall(call) => self.entries.push(Entry::Tool {
                id: call.tool_call_id,
                title: call.title,
                status: call.status,
                output: summarize(&call.content),
            }),
            SessionUpdate::ToolCallUpdate(update) => {
                let entry =
                    self.entries.iter_mut().rev().find(
                        |e| matches!(e, Entry::Tool { id, .. } if *id == update.tool_call_id),
                    );
                if let Some(Entry::Tool {
                    title,
                    status,
                    output,
                    ..
                }) = entry
                {
                    let fields = update.fields;
                    if let Some(s) = fields.status {
                        *status = s;
                    }
                    if let Some(t) = fields.title {
                        *title = t;
                    }
                    if let Some(c) = fields.content {
                        *output = summarize(&c);
                    }
                }
            }
            _ => {}
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if self.permission.is_some() {
            match key.code {
                KeyCode::Char('y') => self.answer_permission(Some(PermissionOptionKind::AllowOnce)),
                KeyCode::Char('a') => {
                    self.answer_permission(Some(PermissionOptionKind::AllowAlways))
                }
                KeyCode::Char('n') | KeyCode::Esc => {
                    self.answer_permission(Some(PermissionOptionKind::RejectOnce))
                }
                KeyCode::Char('c') if ctrl => self.cancel_turn(),
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy {
                    self.cancel_turn();
                } else if !self.input.is_empty() {
                    self.input.clear();
                    self.cursor = 0;
                } else {
                    self.quit = true;
                }
            }
            KeyCode::Char('d') if ctrl && self.input.is_empty() => self.quit = true,
            KeyCode::Char('a') if ctrl => self.cursor = 0,
            KeyCode::Char('e') if ctrl => self.cursor = self.input.len(),
            KeyCode::Char('u') if ctrl => {
                self.input.drain(..self.cursor);
                self.cursor = 0;
            }
            KeyCode::Esc if self.busy => self.cancel_turn(),
            KeyCode::Enter if !self.busy => self.submit(),
            KeyCode::Char(c) if !ctrl => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
            }
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                self.input.remove(self.cursor);
            }
            KeyCode::Delete if self.cursor < self.input.len() => {
                self.input.remove(self.cursor);
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.input.len()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::PageUp => self.scroll_up += 10,
            KeyCode::PageDown => self.scroll_up = self.scroll_up.saturating_sub(10),
            KeyCode::Up => self.scroll_up += 1,
            KeyCode::Down => self.scroll_up = self.scroll_up.saturating_sub(1),
            _ => {}
        }
    }

    fn submit(&mut self) {
        let text: String = self.input.iter().collect::<String>().trim().to_string();
        if text.is_empty() {
            return;
        }
        self.input.clear();
        self.cursor = 0;
        if text == "/quit" || text == "/exit" {
            self.quit = true;
            return;
        }
        self.entries.push(Entry::User(text.clone()));
        self.busy = true;
        self.scroll_up = 0;

        let conn = self.conn.clone();
        let request = PromptRequest::new(
            self.session_id.clone(),
            vec![ContentBlock::Text(TextContent::new(text))],
        );
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let result = conn.send_request(request).block_task().await;
            let _ = tx.send(AppEvent::TurnDone(
                result.map(|r| r.stop_reason).map_err(|e| format!("{e}")),
            ));
        });
    }

    fn cancel_turn(&mut self) {
        if let Some((_, answer)) = self.permission.take() {
            let _ = answer.send(RequestPermissionOutcome::Cancelled);
        }
        if let Err(e) = self
            .conn
            .send_notification(CancelNotification::new(self.session_id.clone()))
        {
            self.entries
                .push(Entry::Error(format!("cancel failed: {e}")));
        }
    }

    fn answer_permission(&mut self, kind: Option<PermissionOptionKind>) {
        let Some((req, answer)) = self.permission.take() else {
            return;
        };
        let outcome = req
            .options
            .iter()
            .find(|o| Some(o.kind) == kind)
            .map(|o| {
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    o.option_id.clone(),
                ))
            })
            .unwrap_or(RequestPermissionOutcome::Cancelled);
        let _ = answer.send(outcome);
    }

    /// Show paths under a workspace root relative to it; secondary roots get a `name/` prefix.
    fn short(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (i, root) in self.roots.iter().enumerate() {
            let mut prefix = root.display().to_string();
            prefix.push('/');
            let replacement = if i == 0 {
                String::new()
            } else {
                root_name(root)
            };
            out = out.replace(&prefix, &replacement);
        }
        out
    }

    fn draw(&self, f: &mut Frame) {
        let [body, input, status] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(f.area());
        self.draw_transcript(f, body);
        self.draw_input(f, input);
        self.draw_status(f, status);
        if let Some((req, _)) = &self.permission {
            let title = req.tool_call.fields.title.clone().or_else(|| {
                self.entries.iter().rev().find_map(|e| match e {
                    Entry::Tool { id, title, .. } if *id == req.tool_call.tool_call_id => {
                        Some(title.clone())
                    }
                    _ => None,
                })
            });
            draw_permission(f, &self.short(&title.unwrap_or_else(|| "Run tool".into())));
        }
    }

    fn draw_transcript(&self, f: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        for entry in &self.entries {
            match entry {
                Entry::User(text) => {
                    for (i, l) in text.lines().enumerate() {
                        let prefix = if i == 0 { "› " } else { "  " };
                        lines.push(Line::from(vec![prefix.cyan().bold(), l.to_string().bold()]));
                    }
                }
                Entry::Agent(text) => {
                    lines.extend(text.trim().lines().map(|l| Line::from(l.to_string())));
                }
                Entry::Thought(text) => {
                    let style = Style::new()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC);
                    lines.extend(
                        text.trim()
                            .lines()
                            .map(|l| Line::styled(format!("  {l}"), style)),
                    );
                }
                Entry::Tool {
                    title,
                    status,
                    output,
                    ..
                } => {
                    let icon = match status {
                        ToolCallStatus::Completed => "✓".green(),
                        ToolCallStatus::Failed => "✗".red(),
                        _ => SPINNER[self.tick % SPINNER.len()].yellow(),
                    };
                    lines.push(Line::from(vec![icon, " ".into(), self.short(title).bold()]));
                    let output = self.short(output);
                    let out: Vec<&str> = output
                        .lines()
                        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("```"))
                        .collect();
                    for l in out.iter().take(MAX_TOOL_OUTPUT_LINES) {
                        lines.push(Line::from(vec![
                            "  │ ".dark_gray(),
                            l.to_string().dark_gray(),
                        ]));
                    }
                    if out.len() > MAX_TOOL_OUTPUT_LINES {
                        let more = out.len() - MAX_TOOL_OUTPUT_LINES;
                        lines.push(Line::from(format!("  │ … {more} more lines").dark_gray()));
                    }
                }
                Entry::Info(text) => lines.push(Line::from(text.clone().dark_gray())),
                Entry::Error(text) => lines.push(Line::from(format!("error: {text}").red())),
            }
            lines.push(Line::default());
        }

        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        let total = paragraph.line_count(area.width);
        let max_offset = total.saturating_sub(area.height as usize);
        let offset = max_offset.saturating_sub(self.scroll_up);
        f.render_widget(
            paragraph.scroll((offset.min(u16::MAX as usize) as u16, 0)),
            area,
        );
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let border = if self.busy {
            Color::DarkGray
        } else {
            Color::Cyan
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(border))
            .title(" onde-code ");
        let inner = block.inner(area);
        // Scroll horizontally so the cursor stays visible.
        let width = inner.width.saturating_sub(1) as usize;
        let start = self.cursor.saturating_sub(width);
        let visible: String = self.input[start..].iter().take(width + 1).collect();
        let text = if self.input.is_empty() && !self.busy {
            Line::from("Ask onde-code to do something…".dark_gray())
        } else {
            Line::from(visible)
        };
        f.render_widget(Paragraph::new(text).block(block), area);
        if self.permission.is_none() {
            f.set_cursor_position((inner.x + (self.cursor - start) as u16, inner.y));
        }
    }

    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let state = if self.permission.is_some() {
            "waiting for approval".yellow()
        } else if self.busy {
            format!("{} working", SPINNER[self.tick % SPINNER.len()]).yellow()
        } else {
            "ready".green()
        };
        let hints = if self.busy {
            "Esc cancel"
        } else {
            "Enter send · Ctrl-C quit"
        };
        let roots = self
            .roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let line = Line::from(vec![
            " ".into(),
            state,
            " · ".dark_gray(),
            self.model.clone().cyan(),
            " · ".dark_gray(),
            roots.dark_gray(),
            " · ".dark_gray(),
            hints.dark_gray(),
            " · ↑↓/PgUp/PgDn scroll".dark_gray(),
        ]);
        f.render_widget(Paragraph::new(line), area);
    }
}

fn draw_permission(f: &mut Frame, title: &str) {
    let area = f.area();
    let width = (title.chars().count() as u16 + 6).clamp(44, area.width.saturating_sub(4).max(1));
    let popup = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(7) / 2,
        width: width.min(area.width),
        height: 7.min(area.height),
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Yellow))
        .title(" Allow this? ");
    let text = vec![
        Line::from(title.to_string().bold()),
        Line::default(),
        Line::from(vec![
            Span::from("[y]").green().bold(),
            " allow   ".into(),
            Span::from("[a]").green().bold(),
            " always   ".into(),
            Span::from("[n]").red().bold(),
            " reject".into(),
        ]),
    ];
    f.render_widget(Clear, popup);
    f.render_widget(
        Paragraph::new(text).block(block).wrap(Wrap { trim: false }),
        popup,
    );
}

/// Display name for a workspace root: its directory name with a trailing slash.
fn root_name(root: &Path) -> String {
    match root.file_name() {
        Some(name) => format!("{}/", name.to_string_lossy()),
        None => root.display().to_string(),
    }
}

fn block_text(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text(t) => t.text.clone(),
        _ => String::new(),
    }
}

/// Short, display-friendly summary of tool output.
fn summarize(content: &[ToolCallContent]) -> String {
    content
        .iter()
        .map(|c| match c {
            ToolCallContent::Content(c) => block_text(&c.content),
            ToolCallContent::Diff(d) => {
                let old = d.old_text.as_deref().map_or(0, |t| t.lines().count());
                format!(
                    "{} (-{old} +{} lines)",
                    d.path.display(),
                    d.new_text.lines().count()
                )
            }
            _ => String::new(),
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod fs_handler_tests {
    use super::*;

    #[test]
    fn read_text_file_slices_by_1_based_line_and_limit() {
        let dir = std::env::temp_dir().join("onde-code-tui-fs-test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("lines.txt");
        std::fs::write(&file, "one\ntwo\nthree\nfour\n").unwrap();

        let all = read_text_file_blocking(&file, None, None).unwrap();
        assert_eq!(all.content, "one\ntwo\nthree\nfour");

        let from_line = read_text_file_blocking(&file, Some(2), None).unwrap();
        assert_eq!(from_line.content, "two\nthree\nfour");

        let limited = read_text_file_blocking(&file, Some(2), Some(1)).unwrap();
        assert_eq!(limited.content, "two");

        let past_end = read_text_file_blocking(&file, Some(99), None).unwrap();
        assert_eq!(past_end.content, "");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_text_file_missing_is_resource_not_found() {
        let err = read_text_file_blocking(Path::new("/definitely/does/not/exist.txt"), None, None)
            .unwrap_err();
        assert_eq!(err.code, agent_client_protocol::ErrorCode::ResourceNotFound);
    }

    #[test]
    fn write_text_file_creates_missing_file_and_parents() {
        let dir = std::env::temp_dir()
            .join("onde-code-tui-fs-test-2")
            .join("nested");
        let file = dir.join("created.txt");
        write_text_file_blocking(&file, "hello").unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello");

        // Overwrite existing.
        write_text_file_blocking(&file, "again").unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "again");

        std::fs::remove_dir_all(dir.parent().unwrap()).ok();
    }
}
