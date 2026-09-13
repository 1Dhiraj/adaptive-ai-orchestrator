#!/usr/bin/env node
"use strict";

const { launch } = require("../lib/launcher");

launch({ noOpen: process.argv.includes("--no-open") }).then(({ child }) => {
  const stop = signal => {
    if (!child.killed) child.kill(signal);
  };
  process.on("SIGINT", () => stop("SIGINT"));
  process.on("SIGTERM", () => stop("SIGTERM"));
  child.on("exit", code => { process.exitCode = code === null ? 1 : code; });
}).catch(error => {
  console.error(`Could not start Adaptive AI Orchestrator: ${error.message}`);
  console.error("Set ORCHESTRATOR_RUNTIME to a packaged runtime path if this is a development checkout.");
  process.exitCode = 1;
});
