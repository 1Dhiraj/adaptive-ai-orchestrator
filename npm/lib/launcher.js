"use strict";

const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");

function dataDirectory(env = process.env, platform = process.platform) {
  if (env.ORCHESTRATOR_DATA_DIR) return path.resolve(env.ORCHESTRATOR_DATA_DIR);
  if (platform === "win32") return path.join(env.LOCALAPPDATA || os.homedir(), "AdaptiveAIOrchestrator");
  if (platform === "darwin") return path.join(os.homedir(), "Library", "Application Support", "AdaptiveAIOrchestrator");
  return path.join(env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"), "adaptive-ai-orchestrator");
}

async function availablePort(preferred = 8000, host = "127.0.0.1") {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once("error", error => error.code === "EADDRINUSE" ? resolve(availablePort(0, host)) : reject(error));
    server.listen({ port: preferred, host }, () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
  });
}

function runtimeCommand(root, env = process.env, platform = process.platform) {
  if (env.ORCHESTRATOR_RUNTIME) return { command: env.ORCHESTRATOR_RUNTIME, args: [] };
  const executable = platform === "win32" ? "orchestrator.exe" : "orchestrator";
  const bundled = path.join(root, "vendor", `${platform}-${process.arch}`, executable);
  if (fs.existsSync(bundled)) return { command: bundled, args: [] };
  // Source checkouts remain convenient for contributors. Published releases
  // include the bundled executable and therefore require no Python install.
  const python = env.ORCHESTRATOR_PYTHON || (platform === "win32" ? "python" : "python3");
  return { command: python, args: ["-m", "orchestrator.cli"] };
}

function openBrowser(url, platform = process.platform) {
  const command = platform === "win32" ? "rundll32" : platform === "darwin" ? "open" : "xdg-open";
  const args = platform === "win32" ? ["url.dll,FileProtocolHandler", url] : [url];
  const child = spawn(command, args, { detached: true, stdio: "ignore", windowsHide: true });
  child.unref();
}

async function waitForHealth(url, child, attempts = 600) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (child.exitCode !== null) throw new Error(`orchestrator exited with code ${child.exitCode}`);
    try {
      const response = await fetch(`${url}/api/health`);
      if (response.ok) return;
    } catch (_) { /* server is still starting */ }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error("orchestrator did not become ready within 60 seconds");
}

async function launch(options = {}) {
  const root = options.root || path.resolve(__dirname, "..", "..");
  const host = "127.0.0.1";
  const port = await availablePort(Number(options.port || process.env.ORCHESTRATOR_PORT || 8000), host);
  const data = dataDirectory();
  const workspace = path.join(data, "workspace");
  fs.mkdirSync(workspace, { recursive: true });
  const runtime = runtimeCommand(root);
  const env = {
    ...process.env,
    ORCHESTRATOR_DATA_DIR: data,
    ORCHESTRATOR_DB: path.join(data, "orchestrator_state.db"),
    ORCHESTRATOR_WORKSPACE: workspace,
    ORCHESTRATOR_SECRET_BACKEND: process.env.ORCHESTRATOR_SECRET_BACKEND || "keyring"
  };
  const args = [...runtime.args, "serve", "--host", host, "--port", String(port)];
  const child = spawn(runtime.command, args, { cwd: root, env, stdio: "inherit", windowsHide: true });
  const url = `http://${host}:${port}`;
  try {
    await waitForHealth(url, child);
  } catch (error) {
    if (!child.killed) child.kill("SIGTERM");
    throw error;
  }
  console.log(`\nAdaptive AI Orchestrator is ready at ${url}`);
  console.log(`Your data is stored in ${data}\n`);
  if (!options.noOpen && process.env.ORCHESTRATOR_NO_OPEN !== "1") openBrowser(url);
  return { child, url, data };
}

module.exports = { availablePort, dataDirectory, launch, openBrowser, runtimeCommand, waitForHealth };
