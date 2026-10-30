#!/usr/bin/env node

import { spawnSync } from "child_process";

/**
 * Returns the executable path which is located inside node_modules
 * The naming convention is onde-code-OS-ARCH
 * If the platform is win32 or cygwin, executable will include a .exe extension.
 * @see https://nodejs.org/api/os.html#osarch
 * @see https://nodejs.org/api/os.html#osplatform
 * @example "x/xx/node_modules/@ondeinference/onde-code-darwin-arm64"
 */
function getExePath() {
  const arch = process.arch;
  let os = process.platform as string;
  let extension = "";
  if (["win32", "cygwin"].includes(process.platform)) {
    os = "windows";
    extension = ".exe";
  }

  try {
    // Since the binary will be located inside node_modules, we can simply call require.resolve
    return require.resolve(
      `@ondeinference/onde-code-${os}-${arch}/bin/onde-code${extension}`,
    );
  } catch (e) {
    throw new Error(
      `Couldn't find application binary inside node_modules for ${os}-${arch}`,
    );
  }
}

/**
 * Runs the application with args using nodejs spawn. In ACP mode stdout is the
 * JSON-RPC pipe to the editor, so nothing is written to it here; errors go to
 * stderr.
 */
function run() {
  const args = process.argv.slice(2);
  const result = spawnSync(getExePath(), args, { stdio: "inherit" });
  if (result.error) {
    throw result.error;
  }
  if (result.signal) {
    // Killed by a signal: re-raise it so the caller sees the same cause.
    process.kill(process.pid, result.signal);
    return;
  }
  process.exit(result.status ?? 1);
}

run();
