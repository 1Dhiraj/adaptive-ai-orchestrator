"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { availablePort, dataDirectory, runtimeCommand } = require("../lib/launcher");

test("uses an explicit user data directory", () => {
  assert.equal(dataDirectory({ ORCHESTRATOR_DATA_DIR: "./custom-data" }), path.resolve("custom-data"));
});

test("uses a platform-specific local data directory", () => {
  assert.equal(dataDirectory({ LOCALAPPDATA: "C:\\Users\\Me\\AppData\\Local" }, "win32"),
    path.join("C:\\Users\\Me\\AppData\\Local", "AdaptiveAIOrchestrator"));
});

test("selects a packaged runtime when present", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "orchestrator-launcher-"));
  const folder = path.join(root, "vendor", `${process.platform}-${process.arch}`);
  fs.mkdirSync(folder, { recursive: true });
  const binary = path.join(folder, process.platform === "win32" ? "orchestrator.exe" : "orchestrator");
  fs.writeFileSync(binary, "runtime");
  assert.equal(runtimeCommand(root).command, binary);
  fs.rmSync(root, { recursive: true, force: true });
});

test("finds a usable localhost port", async () => {
  const port = await availablePort(0);
  assert.ok(Number.isInteger(port) && port > 0);
});
