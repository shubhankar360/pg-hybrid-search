/**
 * Benchmark on BEIR SciFact (300 test claims, 5,183 abstracts).
 *
 *   npx tsx src/precompute.ts   # once: embeddings, parallel and resumable
 *   npm run bench
 *
 * Writes results/benchmark.md and results/benchmark.json.
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { openDb, migrate } from "./db.js";
import { fetchScifact, loadScifact } from "./data.js";
import { cachePath, readCache } from "./precompute.js";
import { upsertDocs } from "./ingest.js";
import { fts, hybrid, vector, type Hit } from "./search.js";
import { mean, mrrAt, ndcgAt, percentile, recallAt } from "./metrics.js";


async function main() {
  const root = await fetchScifact();
  const { corpus, queries, qrels } = await loadScifact(root);
  console.log(`SciFact: ${corpus.length} docs, ${queries.length} test queries`);

  if (!existsSync(cachePath("corpus", corpus.length)) || !existsSync(cachePath("queries", queries.length)))
    throw new Error("embeddings not computed yet: run `npx tsx src/precompute.ts` first");
  const dv = readCache("corpus", corpus.length);
  const qv = readCache("queries", queries.length);
  const embedder = { name: "all-MiniLM-L6-v2", dims: 384 };

  const db = await openDb();
  await db.exec("DROP TABLE IF EXISTS docs");
  await migrate(db, embedder.dims);
  let t = Date.now();
  await upsertDocs(db, corpus, dv);
  const ingestMs = Date.now() - t;
  console.log(`ingested + indexed in ${(ingestMs / 1000).toFixed(1)}s (${db.kind})`);

  type Method = { name: string; note: string; setup?: string; run: (i: number, k: number) => Promise<Hit[]> };
  const K = 100;
  const methods: Method[] = [
    { name: "fts-and", note: "plainto_tsquery: every term required", run: (i, k) => fts(db, queries[i].text, k, "and") },
    { name: "fts-or", note: "OR of terms, ranked by ts_rank_cd", run: (i, k) => fts(db, queries[i].text, k, "or") },
    { name: "vector-exact", note: "sequential scan, true nearest neighbours", run: (i, k) => vector(db, qv[i], k, true) },
    { name: "vector-hnsw ef_search=40", note: "pgvector default", setup: "SET hnsw.ef_search = 40", run: (i, k) => vector(db, qv[i], k) },
    { name: "vector-hnsw ef_search=200", note: "ef_search raised above k", setup: "SET hnsw.ef_search = 200", run: (i, k) => vector(db, qv[i], k) },
    { name: "hybrid-rrf", note: "fts-or + hnsw(200), RRF k=60, depth 100", setup: "SET hnsw.ef_search = 200", run: (i, k) => hybrid(db, queries[i].text, qv[i], k, 100) },
  ];

  const exact = new Map<number, string[]>();
  const rows = [];
  for (const m of methods) {
    if (m.setup) await db.exec(m.setup);
    const nd: number[] = [], rc: number[] = [], mr: number[] = [], lat: number[] = [], returned: number[] = [], overlap: number[] = [];
    for (let i = 0; i < queries.length; i++) {
      t = performance.now();
      const hits = await m.run(i, K);
      lat.push(performance.now() - t);
      const ids = hits.map((h) => h.id);
      const rel = qrels.get(queries[i].id)!;
      nd.push(ndcgAt(ids, rel, 10)); rc.push(recallAt(ids, rel, 100)); mr.push(mrrAt(ids, rel, 10));
      returned.push(ids.length);
      if (m.name === "vector-exact") exact.set(i, ids);
      else if (m.name.startsWith("vector-hnsw")) {
        const truth = new Set(exact.get(i)!.slice(0, 10));
        overlap.push(ids.slice(0, 10).filter((x) => truth.has(x)).length / 10);
      }
    }
    const row = {
      method: m.name, note: m.note, ndcg10: mean(nd), recall100: mean(rc), mrr10: mean(mr),
      p50ms: percentile(lat, 50), p95ms: percentile(lat, 95), meanReturned: mean(returned),
      annRecall10: overlap.length ? mean(overlap) : null,
    };
    rows.push(row);
    console.log(`${m.name.padEnd(28)} nDCG@10 ${row.ndcg10.toFixed(3)}  R@100 ${row.recall100.toFixed(3)}  MRR@10 ${row.mrr10.toFixed(3)}  p50 ${row.p50ms.toFixed(1)}ms  returned ${row.meanReturned.toFixed(1)}`);
  }
  await db.close();

  mkdirSync("results", { recursive: true });
  const meta = { dataset: "BEIR SciFact (test)", docs: corpus.length, queries: queries.length, embedder: embedder.name,
    backend: db.kind, ingestSeconds: ingestMs / 1000, node: process.version, date: new Date().toISOString().slice(0, 10) };
  writeFileSync("results/benchmark.json", JSON.stringify({ meta, rows }, null, 2) + "\n");
  const f = (x: number | null, d = 3) => (x === null ? "—" : x.toFixed(d));
  const md = [
    `# Benchmark — ${meta.dataset}`, "",
    `${meta.docs} abstracts, ${meta.queries} test claims, embeddings from \`${meta.embedder}\` (384-d), ${meta.backend === "pglite" ? "PGlite (Postgres 18, WASM, in-process)" : "Postgres"}. ` +
      `Ingest plus GIN and HNSW index build: ${meta.ingestSeconds.toFixed(1)} s. Latencies are single-threaded and include the round trip through the driver.`, "",
    "| method | nDCG@10 | Recall@100 | MRR@10 | rows returned (asked 100) | HNSW top-10 overlap with exact | p50 ms | p95 ms |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ...rows.map((r) => `| \`${r.method}\` — ${r.note} | ${f(r.ndcg10)} | ${f(r.recall100)} | ${f(r.mrr10)} | ${r.meanReturned.toFixed(1)} | ${f(r.annRecall10, 2)} | ${f(r.p50ms, 1)} | ${f(r.p95ms, 1)} |`),
    "", `Generated by \`npm run bench\` on ${meta.date}, Node ${meta.node}.`, "",
  ].join("\n");
  writeFileSync("results/benchmark.md", md);
  console.log("\nwrote results/benchmark.md");
}

main().catch((e) => { console.error(e); process.exit(1); });
