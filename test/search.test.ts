/**
 * Runs on PGlite by default and, in CI, a second time against a real
 * pgvector/pgvector server (DATABASE_URL), so every query here is proven
 * on both backends.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { enableBm25, migrate, openDb, toVector, type Db } from "../src/db.js";
import { HashEmbedder, cosine } from "../src/embed.js";
import { upsertDocs } from "../src/ingest.js";
import { bm25, fts, hybrid, orQuery, rrf, vector } from "../src/search.js";
import { mrrAt, ndcgAt, percentile, recallAt } from "../src/metrics.js";

const DOCS = [
  { id: "d1", title: "Vitamin D and bone density", body: "Vitamin D supplementation increased bone mineral density in older adults." },
  { id: "d2", title: "Sleep and memory", body: "Deep sleep consolidates declarative memory in healthy volunteers." },
  { id: "d3", title: "Exercise and mood", body: "Aerobic exercise reduced symptoms of depression over twelve weeks." },
  { id: "d4", title: "Calcium intake", body: "Dietary calcium was not associated with fracture risk in this cohort." },
  { id: "d5", title: "Coffee and alertness", body: "Caffeine improved alertness but disrupted sleep when taken late." },
  { id: "d6", title: "Bone health review", body: "Bone density depends on calcium, vitamin D and weight-bearing exercise." },
];

let db: Db;
let hasBm25 = false;
const emb = new HashEmbedder(64);

beforeAll(async () => {
  db = await openDb(undefined, { schema: "test_search" });
  await db.exec("DROP TABLE IF EXISTS docs");
  await migrate(db, emb.dims);
  await upsertDocs(db, DOCS, await emb.embed(DOCS.map((d) => `${d.title}. ${d.body}`)));
  hasBm25 = await enableBm25(db); // PGlite bundles pg_textsearch; the stock pgvector image does not
});
afterAll(async () => db?.close());

describe("schema", () => {
  it("generates the weighted tsvector and indexes both columns", async () => {
    const idx = await db.query<{ indexname: string }>("SELECT indexname FROM pg_indexes WHERE tablename = 'docs' ORDER BY 1");
    expect(idx.map((r) => r.indexname).filter((n) => n !== "docs_bm25_idx")).toEqual(["docs_embedding_idx", "docs_pkey", "docs_tsv_idx"]);
    const [r] = await db.query<{ tsv: string }>("SELECT tsv::text AS tsv FROM docs WHERE id = 'd1'");
    expect(r.tsv).toMatch(/'vitamin':1A/); // title terms carry weight A
  });

  it("upserts rather than duplicating", async () => {
    await upsertDocs(db, [{ ...DOCS[0], title: "Vitamin D and bone density (revised)" }], await emb.embed(["x"]));
    const rows = await db.query<{ n: number; title: string }>("SELECT count(*)::int AS n, max(title) AS title FROM docs WHERE id = 'd1'");
    expect(rows[0]).toEqual({ n: 1, title: "Vitamin D and bone density (revised)" });
    await upsertDocs(db, [DOCS[0]], await emb.embed([`${DOCS[0].title}. ${DOCS[0].body}`]));
  });
});

describe("full-text", () => {
  it("AND needs every term; OR ranks partial matches", async () => {
    const q = "does vitamin D improve memory";
    expect(await fts(db, q, 10, "and")).toEqual([]);
    const or = (await fts(db, q, 10, "or")).map((h) => h.id);
    expect(or).toContain("d1");
    expect(or).toContain("d2");
  });

  it("builds a safe OR query from arbitrary text", () => {
    expect(orQuery("Vitamin-D's effect: (bone) & 'density'!")).toBe("vitamin | d | s | effect | bone | density");
    expect(orQuery("!!!")).toBe("");
  });

  it("title matches outrank body matches", async () => {
    const ids = (await fts(db, "calcium", 10)).map((h) => h.id);
    expect(ids[0]).toBe("d4");
  });
});

describe("vector", () => {
  it("HNSW and exact search agree on a small corpus, and match cosine computed in JS", async () => {
    const [qv] = await emb.embed(["bone density vitamin"]);
    const exact = await vector(db, qv, 3, true);
    const ann = await vector(db, qv, 3);
    // Compare scores, not ids: equal distances may come back in either order.
    ann.forEach((h, i) => expect(h.score).toBeCloseTo(exact[i].score, 6));
    const all = await emb.embed(DOCS.map((d) => `${d.title}. ${d.body}`));
    const js = DOCS.map((d, i) => ({ id: d.id, s: cosine(qv, all[i]) })).sort((a, b) => b.s - a.s);
    expect(exact[0].id).toBe(js[0].id);
    expect(exact[0].score).toBeCloseTo(js[0].s, 4);
  });

  it("pgvector text format round-trips", async () => {
    const [r] = await db.query<{ v: string }>("SELECT $1::vector::text AS v", [toVector([0.5, -1, 0.25])]);
    expect(r.v).toBe("[0.5,-1,0.25]");
  });
});

describe("hybrid", () => {
  it("SQL fusion equals reference RRF over the two legs", async () => {
    const q = "bone density and calcium";
    const [qv] = await emb.embed([q]);
    const lex = (await fts(db, q, 100)).map((h) => h.id);
    const sem = (await vector(db, qv, 100, true)).map((h) => h.id);
    const sql = await hybrid(db, q, qv, 5);
    const ref = rrf([lex, sem], 5);
    expect(sql.map((h) => h.id)).toEqual(ref.map((h) => h.id));
    sql.forEach((h, i) => expect(h.score).toBeCloseTo(ref[i].score, 9));
  });

  it("still answers when the text has no indexable terms", async () => {
    const [qv] = await emb.embed(["the of and"]);
    const hits = await hybrid(db, "the of and", qv, 3);
    expect(hits).toHaveLength(3); // vector leg alone
  });
});

describe("bm25 (pg_textsearch)", () => {
  it("ranks by BM25 and skips non-matching documents", async (ctx) => {
    if (!hasBm25) return ctx.skip();
    const hits = await bm25(db, "vitamin bone density", 10);
    expect(hits[0].id).toBe("d1");
    expect(hits.map((h) => h.id)).not.toContain("d2"); // no shared terms
    expect(hits.every((h) => h.score > 0)).toBe(true);
  });

  it("BM25 hybrid equals reference RRF", async (ctx) => {
    if (!hasBm25) return ctx.skip();
    const q = "sleep and memory";
    const [qv] = await emb.embed([q]);
    const lex = (await bm25(db, q, 100)).map((h) => h.id);
    const sem = (await vector(db, qv, 100, true)).map((h) => h.id);
    const sql = await hybrid(db, q, qv, 5, 100, 60, "bm25");
    expect(sql.map((h) => h.id)).toEqual(rrf([lex, sem], 5).map((h) => h.id));
  });
});

describe("metrics", () => {
  const rel = new Map([["a", 1], ["b", 1]]);
  it("nDCG, recall and MRR", () => {
    expect(ndcgAt(["a", "b", "c"], rel)).toBeCloseTo(1);
    expect(ndcgAt(["c", "a"], rel)).toBeCloseTo((1 / Math.log2(3)) / (1 + 1 / Math.log2(3)));
    expect(recallAt(["a", "x"], rel, 100)).toBe(0.5);
    expect(mrrAt(["x", "y", "b"], rel)).toBeCloseTo(1 / 3);
    expect(mrrAt(["x"], rel)).toBe(0);
    expect(percentile([5, 1, 3, 2, 4], 50)).toBe(3);
  });
});
