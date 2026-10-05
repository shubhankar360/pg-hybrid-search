/**
 * A small HTTP API over the same SQL. No framework: node:http is enough for
 * three routes, and it keeps the container to Node plus two drivers.
 *
 *   GET  /health
 *   POST /docs                 [{ "id", "title"?, "body" }]  -> embeds and upserts
 *   GET  /search?q=&mode=hybrid|fts|vector&k=10
 *
 * EMBEDDER=hash swaps in the deterministic test embedder (no model download).
 */
import { createServer, type IncomingMessage, type Server } from "node:http";
import { pathToFileURL } from "node:url";
import { migrate, openDb, type Db } from "./db.js";
import type { Doc } from "./data.js";
import { HashEmbedder, MiniLM, type Embedder } from "./embed.js";
import { upsertDocs } from "./ingest.js";
import { fts, hybrid, vector } from "./search.js";

const MAX_BODY = 5 * 1024 * 1024;

async function readJson(req: IncomingMessage): Promise<unknown> {
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const c of req) {
    size += c.length;
    if (size > MAX_BODY) throw Object.assign(new Error("body too large"), { status: 413 });
    chunks.push(c);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "null");
}

export async function makeApp(db: Db, embedder: Embedder): Promise<Server> {
  await db.exec("SET hnsw.ef_search = 200").catch(() => undefined);
  return createServer(async (req, res) => {
    const send = (status: number, body: unknown) => {
      res.writeHead(status, { "content-type": "application/json" });
      res.end(JSON.stringify(body));
    };
    try {
      const url = new URL(req.url ?? "/", "http://x");
      if (req.method === "GET" && url.pathname === "/health") return send(200, { ok: true, backend: db.kind, embedder: embedder.name });

      if (req.method === "POST" && url.pathname === "/docs") {
        const body = await readJson(req);
        const docs = (Array.isArray(body) ? body : [body]) as Partial<Doc>[];
        if (!docs.length || docs.some((d) => !d || typeof d.id !== "string" || typeof d.body !== "string"))
          return send(400, { error: "expected [{ id: string, title?: string, body: string }]" });
        const clean = docs.map((d) => ({ id: d.id!, title: d.title ?? "", body: d.body! }));
        await upsertDocs(db, clean, await embedder.embed(clean.map((d) => (d.title ? `${d.title}. ${d.body}` : d.body))));
        return send(200, { upserted: clean.length });
      }

      if (req.method === "GET" && url.pathname === "/search") {
        const q = (url.searchParams.get("q") ?? "").trim();
        const mode = url.searchParams.get("mode") ?? "hybrid";
        const k = Math.min(100, Math.max(1, Number(url.searchParams.get("k") ?? 10) || 10));
        if (!q) return send(400, { error: "q is required" });
        const t0 = performance.now();
        const hits = mode === "fts" ? await fts(db, q, k)
          : mode === "vector" ? await vector(db, (await embedder.embed([q]))[0], k)
          : mode === "hybrid" ? await hybrid(db, q, (await embedder.embed([q]))[0], k)
          : null;
        if (!hits) return send(400, { error: "mode must be hybrid, fts or vector" });
        const titles = hits.length
          ? await db.query<{ id: string; title: string }>("SELECT id, title FROM docs WHERE id = ANY($1)", [hits.map((h) => h.id)])
          : [];
        const t = new Map(titles.map((r) => [r.id, r.title]));
        return send(200, { mode, ms: Math.round(performance.now() - t0), results: hits.map((h) => ({ ...h, title: t.get(h.id) ?? "" })) });
      }
      send(404, { error: "not found" });
    } catch (e) {
      const status = (e as { status?: number }).status ?? (e instanceof SyntaxError ? 400 : 500);
      send(status, { error: status === 500 ? "internal error" : (e as Error).message });
      if (status === 500) console.error(e);
    }
  });
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const embedder: Embedder = process.env.EMBEDDER === "hash" ? new HashEmbedder(384) : await MiniLM.load();
  const db = await openDb();
  await migrate(db, embedder.dims);
  const port = Number(process.env.PORT ?? 8080);
  (await makeApp(db, embedder)).listen(port, () => console.log(`listening on :${port} (${db.kind}, ${embedder.name})`));
}
