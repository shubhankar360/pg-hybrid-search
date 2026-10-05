import { defineConfig } from "vitest/config";

// Starting PGlite compiles Postgres from WASM: a few seconds on an idle
// laptop, longer on a busy CI runner. Generous hook timeouts keep a slow
// machine from reading as a failing test.
export default defineConfig({ test: { hookTimeout: 60_000, testTimeout: 30_000 } });
