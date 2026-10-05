/**
 * BEIR SciFact: 5,183 scientific abstracts and 300 test claims with expert
 * relevance judgements (Wadden et al., 2020; Thakur et al., 2021). Downloaded
 * on demand rather than committed: the dataset is CC BY-NC, so this repo
 * ships the code that measures it, not the data.
 */
import { createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { createInterface } from "node:readline";
import type { Qrels } from "./metrics.js";

const URL_ = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip";

export interface Doc { id: string; title: string; body: string }
export interface Query { id: string; text: string }

export async function fetchScifact(dir = "data"): Promise<string> {
  const root = join(dir, "scifact");
  if (existsSync(join(root, "corpus.jsonl"))) return root;
  mkdirSync(dir, { recursive: true });
  const r = await fetch(URL_);
  if (!r.ok) throw new Error(`download failed: ${r.status}`);
  const zip = join(dir, "scifact.zip");
  writeFileSync(zip, Buffer.from(await r.arrayBuffer()));
  execFileSync("unzip", ["-q", "-o", zip, "-d", dir]);
  return root;
}

async function jsonl<T>(path: string): Promise<T[]> {
  const out: T[] = [];
  for await (const line of createInterface({ input: createReadStream(path) })) if (line.trim()) out.push(JSON.parse(line));
  return out;
}

export async function loadScifact(root: string) {
  const corpus = (await jsonl<{ _id: string; title: string; text: string }>(join(root, "corpus.jsonl")))
    .map((d): Doc => ({ id: d._id, title: d.title, body: d.text }));
  const qrels: Qrels = new Map();
  for (const line of readFileSync(join(root, "qrels", "test.tsv"), "utf8").trim().split("\n").slice(1)) {
    const [qid, did, score] = line.split("\t");
    if (!qrels.has(qid)) qrels.set(qid, new Map());
    qrels.get(qid)!.set(did, Number(score));
  }
  const queries = (await jsonl<{ _id: string; text: string }>(join(root, "queries.jsonl")))
    .filter((q) => qrels.has(q._id))
    .map((q): Query => ({ id: q._id, text: q.text }));
  return { corpus, queries, qrels };
}

if (process.argv.includes("--fetch")) {
  fetchScifact().then((p) => console.log(`SciFact ready in ${p}`));
}
