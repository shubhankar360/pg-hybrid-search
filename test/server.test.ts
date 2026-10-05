import type { AddressInfo } from "node:net";
import type { Server } from "node:http";
import { afterAll, beforeAll, expect, it } from "vitest";
import { migrate, openDb, type Db } from "../src/db.js";
import { HashEmbedder } from "../src/embed.js";
import { makeApp } from "../src/server.js";

let db: Db;
let server: Server;
let base: string;

beforeAll(async () => {
  db = await openDb(undefined, { schema: "test_server" });
  await db.exec("DROP TABLE IF EXISTS docs");
  const emb = new HashEmbedder(32);
  await migrate(db, emb.dims);
  server = await makeApp(db, emb);
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", () => r()));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
afterAll(async () => {
  if (server) await new Promise((r) => server.close(r));
  await db?.close();
});

it("ingests and searches in every mode", async () => {
  const docs = [
    { id: "a", title: "Postgres full-text search", body: "tsvector and tsquery rank documents with ts_rank_cd." },
    { id: "b", title: "Vector search", body: "pgvector stores embeddings and builds HNSW indexes." },
  ];
  const r = await fetch(`${base}/docs`, { method: "POST", body: JSON.stringify(docs) });
  expect(await r.json()).toEqual({ upserted: 2 });
  for (const mode of ["hybrid", "fts", "vector"]) {
    const s = await (await fetch(`${base}/search?q=${encodeURIComponent("hnsw embeddings")}&mode=${mode}&k=2`)).json();
    expect(s.mode).toBe(mode);
    expect(s.results.length).toBeGreaterThan(0);
    expect(s.results[0]).toHaveProperty("title");
  }
  const fts = await (await fetch(`${base}/search?q=tsquery&mode=fts`)).json();
  expect(fts.results[0].id).toBe("a");
});

it("rejects bad input with a reason", async () => {
  expect((await fetch(`${base}/docs`, { method: "POST", body: JSON.stringify([{ id: 1 }]) })).status).toBe(400);
  expect((await fetch(`${base}/docs`, { method: "POST", body: "{not json" })).status).toBe(400);
  expect((await fetch(`${base}/search?q=`)).status).toBe(400);
  expect((await fetch(`${base}/search?q=x&mode=nope`)).status).toBe(400);
  expect((await fetch(`${base}/nope`)).status).toBe(404);
  const h = await (await fetch(`${base}/health`)).json();
  expect(h.ok).toBe(true);
});
