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

## Findings: BEIR SciFact, 300 test claims, 5,183 abstracts

| method | nDCG@10 | Recall@100 | rows returned (asked 100) | p50 |
| --- | --- | --- | --- | --- |
| `plainto_tsquery` (the usual full-text call) | 0.070 | 0.069 | **0.1** | 29 ms |
| full-text OR query, `ts_rank_cd` | 0.362 | 0.773 | 99.8 | 98 ms |
| **BM25** (`pg_textsearch`) | **0.689** | 0.918 | 99.8 | 129 ms |
| vector, exact | 0.643 | 0.925 | 100 | 77 ms |
| vector, HNSW, pgvector default `ef_search = 40` | 0.637 | **0.875** | **40** | 4 ms |
| vector, HNSW, `ef_search = 200` | 0.643 | 0.925 | 100 | 7 ms |
| hybrid RRF: `ts_rank_cd` + vector | **0.597** | 0.937 | 100 | 134 ms |
| **hybrid RRF: BM25 + vector** | **0.712** | **0.953** | 100 | 163 ms |

Full table with MRR, p95 and HNSW overlap: [`results/benchmark.md`](results/benchmark.md).
Embeddings are `all-MiniLM-L6-v2`. Its 0.643 here matches the model's published SciFact
score, which is the check that the pipeline is sound.

**1. The full-text call most tutorials show is nearly useless for questions.**
`plainto_tsquery` ANDs every word, and a natural-language claim almost never has all its
words in the one relevant abstract. It returned **0.1 rows per query** on average.
OR-ing the terms fixes recall.

**2. Postgres's built-in ranking is not BM25.** `ts_rank_cd` has no inverse document
frequency, so "the patients" counts as much as "ketamine". It scored 0.362 against 0.689
for real BM25 on the same tokens.

**3. Hybrid search only helps if the lexical leg is real BM25.** Fusing `ts_rank_cd` with
vectors made the top 10 *worse* than vectors alone (0.597 vs 0.643). Fusing BM25 with
vectors beat both of its legs (0.712). This matches what
[hybrid-rag-eval](https://github.com/shubhankar360/hybrid-rag-eval) found from the other
direction: fusion is not free, and a weak leg drags the result down.

**4. pgvector's default `ef_search` silently truncates results.** HNSW returns at most
`ef_search` candidates, so `LIMIT 100` at the default of 40 returned **40 rows**, and
Recall@100 fell from 0.925 to 0.875 with no error. Raising it to 200 matched exact search
at a tenth of the latency.

**5. Raising `ef_search` can make the planner drop the index.** With table statistics
present, `ef_search = 200` made Postgres cost the HNSW scan above a sequential scan plus
sort on 5,183 rows, and it quietly ran the exact query instead: same results, 10× slower.
The benchmark forces the index for that row (`enable_seqscan = off`) so it measures HNSW.
In production, check `EXPLAIN` after changing `ef_search`.

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
| tests on PGlite (Node 20 and 22) | the logic, including BM25 (PGlite bundles `pg_textsearch`) |
| the same tests against `pgvector/pgvector:pg17` | the SQL is portable to a real server (BM25 tests skip: that image has no `pg_textsearch`) |
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

The real server also caught two bugs the in-process database could not show: test files
running in parallel raced on `CREATE EXTENSION`, and without `SCHEMA public` the extension
installed into whichever schema came first on the search path. Each test file now gets its
own schema, and the extension always goes into `public`.

## Run it

```bash
npm ci
npm test                         # 14 tests on in-process Postgres, ~10 s
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
  search.ts     fts (AND / OR), bm25, vector (HNSW / exact), hybrid RRF; reference rrf()
  embed.ts      MiniLM on onnxruntime-web; deterministic HashEmbedder for tests
  precompute.ts parallel, resumable embedding of the benchmark set
  bench.ts      nDCG@10, Recall@100, MRR@10, latency, HNSW overlap with exact
  server.ts     GET /health · POST /docs · GET /search
test/           search.test.ts (engine-agnostic), server.test.ts
```

## Licence

MIT
