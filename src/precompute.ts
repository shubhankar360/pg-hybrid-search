/**
 * Embed the SciFact corpus and queries once, in parallel, resumably.
 *
 *   npx tsx src/precompute.ts            # one worker per core (max 4), then merge
 *
 * WASM inference is single-threaded, so the corpus takes ~40 min on one core
 * of a 2019 laptop. Each worker owns a contiguous shard and appends every
 * batch to its own file as it goes: an interrupted run resumes where it
 * stopped instead of starting again.
 */
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { availableParallelism } from "node:os";
import { join } from "node:path";
import { fetchScifact, loadScifact } from "./data.js";

export const CACHE = "data/cache";
const DIMS = 384;
const NAME = "all-MiniLM-L6-v2";

export const docText = (d: { title: string; body: string }) => (d.title ? `${d.title}. ${d.body}` : d.body);
export const cachePath = (set: string, n: number) => join(CACHE, `${NAME}-${set}-${n}.f32`);

async function texts() {
  const { corpus, queries } = await loadScifact(await fetchScifact());
  return { corpus: corpus.map(docText), queries: queries.map((q) => q.text) };
}

async function worker(set: "corpus" | "queries", shard: number, of: number) {
  const all = (await texts())[set];
  const per = Math.ceil(all.length / of);
  const lo = shard * per, hi = Math.min(all.length, lo + per);
  const file = join(CACHE, `${NAME}-${set}-${all.length}.shard${shard}of${of}`);
  const done = existsSync(file) ? statSync(file).size / 4 / DIMS : 0;
  if (lo + done >= hi) return;
  const { MiniLM } = await import("./embed.js");
  const m = await MiniLM.load();
  for (let i = lo + done; i < hi; i += 16) {
    const vs = await m.embed(all.slice(i, Math.min(i + 16, hi)));
    const flat = new Float32Array(vs.length * DIMS);
    vs.forEach((v, j) => flat.set(v, j * DIMS));
    appendFileSync(file, Buffer.from(flat.buffer));
    if (shard === 0) process.stdout.write(`\r  ${set}: shard 0 at ${i + vs.length - lo}/${hi - lo}`);
  }
  if (shard === 0) process.stdout.write("\n");
}

async function run(set: "corpus" | "queries", n: number, of: number) {
  const out = cachePath(set, n);
  if (existsSync(out)) return;
  const t0 = Date.now();
  await Promise.all(Array.from({ length: of }, (_, s) => new Promise<void>((res, rej) => {
    const p = spawn(process.execPath, [...process.execArgv, process.argv[1], "--worker", set, String(s), String(of)], { stdio: ["ignore", "inherit", "inherit"] });
    p.on("exit", (c) => (c === 0 ? res() : rej(new Error(`worker ${s} exited ${c}`))));
  })));
  const parts = Array.from({ length: of }, (_, s) => readFileSync(join(CACHE, `${NAME}-${set}-${n}.shard${s}of${of}`)));
  const merged = Buffer.concat(parts);
  if (merged.length !== n * DIMS * 4) throw new Error(`expected ${n} vectors, got ${merged.length / 4 / DIMS}`);
  writeFileSync(out, merged);
  console.log(`  ${set}: ${n} vectors in ${((Date.now() - t0) / 60000).toFixed(1)} min -> ${out}`);
}

export function readCache(set: string, n: number): Float32Array[] {
  const buf = readFileSync(cachePath(set, n));
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  return Array.from({ length: n }, (_, i) => all.slice(i * DIMS, (i + 1) * DIMS));
}

const argv = process.argv;
if (argv.includes("--worker")) {
  const i = argv.indexOf("--worker");
  await worker(argv[i + 1] as "corpus" | "queries", Number(argv[i + 2]), Number(argv[i + 3]));
} else if (argv[1]?.endsWith("precompute.ts")) {
  mkdirSync(CACHE, { recursive: true });
  const t = await texts();
  const of = Math.min(4, availableParallelism());
  await run("queries", t.queries.length, of);
  await run("corpus", t.corpus.length, of);
}
