# pg-hybrid-search

**Hybrid search inside Postgres: full-text plus pgvector, fused with reciprocal rank
fusion in one SQL statement. Measured on BEIR SciFact and tested on two Postgres engines.**

Most RAG stacks put keyword search in one service, vectors in another, and fuse in
application code. Postgres can do all three in a single query against a single table, and
the two indexes can never disagree about which documents exist. This repo is that design,
plus the measurements needed to run it without guessing:

- a benchmark on a public retrieval set with expert relevance judgements, where each
  method is a few lines of SQL;
- a test suite that runs on **PGlite** (Postgres 18 compiled to WASM, in-process) and
  again in CI on a **real `pgvector/pgvector` server**, so every query is proven portable;
- a small HTTP API with Docker Compose for the production shape.

<!-- RESULTS -->

## The query

One table carries both indexes: a stored, weighted `tsvector` (title outranks body) with
a GIN index, and a 384-d embedding with an HNSW index ([`src/schema.sql`](src/schema.sql)).
Hybrid search ranks each leg inside its own `LIMIT`ed subquery, so the vector leg can use
its index, then numbers the ranks and fuses them:

```sql
WITH tq  AS (SELECT to_tsquery('english', $1) AS query),
     lex AS (SELECT id, row_number() OVER (ORDER BY s DESC, id) AS r FROM (
               SELECT id, ts_rank_cd(tsv, tq.query) AS s FROM docs, tq
                WHERE tsv @@ tq.query ORDER BY s DESC, id LIMIT $3) l),
     sem AS (SELECT id, row_number() OVER (ORDER BY d, id) AS r FROM (
               SELECT id, embedding <=> $2::vector AS d FROM docs ORDER BY d LIMIT $3) v)
SELECT id, sum(1.0 / ($4 + r)) AS score
  FROM (SELECT id, r FROM lex UNION ALL SELECT id, r FROM sem) u
 GROUP BY id ORDER BY score DESC, id LIMIT $5;
```

A test checks this SQL against a reference RRF written in TypeScript, score for score.

## Two engines, one set of SQL

[`src/db.ts`](src/db.ts) is one small interface over PGlite and `node-postgres`;
`DATABASE_URL` picks the real server. Development, tests and the benchmark need no
Docker and no running database. CI runs three jobs:

| job | proves |
| --- | --- |
| tests on PGlite (Node 20 and 22) | the logic |
| the same tests against `pgvector/pgvector:pg17` | the SQL is portable to a real server |
| `docker compose up`, ingest, search | the image builds and serves |

## Embeddings without a native runtime

`all-MiniLM-L6-v2` runs through **onnxruntime-web's WebAssembly backend**, not
onnxruntime-node, which ships no binary for Intel macOS. The tokenizer comes from
transformers.js, and mean pooling and normalisation are written out in
[`src/embed.ts`](src/embed.ts). The output matches the model card's reference pair
(cosine 0.755 between "A man is eating food." and "A man is eating a piece of bread.").
WASM inference is single-threaded, so [`src/precompute.ts`](src/precompute.ts) embeds the
corpus with one worker per core, and each worker appends to its own shard file. An
interrupted run resumes instead of starting over.

Tests use a deterministic hashing embedder, so the suite needs no model and no network.

## Run it

```bash
npm ci
npm test                         # 12 tests on in-process Postgres, ~8 s
npx tsx src/precompute.ts        # once: downloads SciFact + model, embeds (parallel, resumable)
npm run bench                    # writes results/benchmark.md

docker compose up --build        # Postgres + pgvector, API on :8080
curl -X POST localhost:8080/docs -d '[{"id":"1","title":"Hello","body":"pgvector and full-text in one query"}]'
curl 'localhost:8080/search?q=vector+search&mode=hybrid'   # mode = hybrid | fts | vector
```

## Data

[SciFact](https://github.com/allenai/scifact) (Wadden et al., 2020) in its
[BEIR](https://github.com/beir-cellar/beir) form (Thakur et al., 2021): 5,183 abstracts and
300 test claims. It is CC BY-NC, so it is downloaded on demand and not committed.

## Layout

```
src/
  schema.sql    one table, weighted tsvector + GIN, vector + HNSW
  db.ts         PGlite or node-postgres behind one interface
  search.ts     fts (AND / OR), vector (HNSW / exact), hybrid RRF; reference rrf()
  embed.ts      MiniLM on onnxruntime-web; deterministic HashEmbedder for tests
  precompute.ts parallel, resumable embedding of the benchmark set
  bench.ts      nDCG@10, Recall@100, MRR@10, latency, HNSW overlap with exact
  server.ts     GET /health · POST /docs · GET /search
test/           search.test.ts (engine-agnostic), server.test.ts
```

## Licence

MIT
