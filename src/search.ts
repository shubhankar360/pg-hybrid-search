/**
 * Three retrieval methods, all plain SQL, so the database does the work and
 * the same queries run on PGlite and on any Postgres with pgvector.
 */
import type { Db } from "./db.js";
import { toVector } from "./db.js";

export interface Hit {
  id: string;
  score: number;
}

/**
 * Full-text query text. `plainto_tsquery` ANDs every term, which is right
 * for a search box and wrong for retrieval: a natural-language question
 * rarely has *all* its words in the one relevant document. "or" builds the
 * disjunction and lets ts_rank_cd order by how much matched.
 */
export function orQuery(text: string): string {
  const words = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  return [...new Set(words)].join(" | ");
}

function tsq(mode: "and" | "or"): string {
  return mode === "and" ? "plainto_tsquery('english', $1)" : "to_tsquery('english', $1)";
}

export async function fts(db: Db, q: string, k: number, mode: "and" | "or" = "or"): Promise<Hit[]> {
  const arg = mode === "and" ? q : orQuery(q);
  if (!arg) return [];
  return db.query<Hit>(
    `SELECT id, ts_rank_cd(tsv, query)::float8 AS score
       FROM docs, ${tsq(mode)} AS query
      WHERE tsv @@ query
      ORDER BY score DESC, id
      LIMIT $2`,
    [arg, k],
  );
}

/**
 * Okapi BM25 through pg_textsearch (k1 = 1.2, b = 0.75). Its operator returns
 * the negated score, so lower is better and 0 means no term matched.
 */
export async function bm25(db: Db, q: string, k: number): Promise<Hit[]> {
  if (!orQuery(q)) return [];
  return db.query<Hit>(
    `SELECT id, -(content <@> to_bm25query($1, 'docs_bm25_idx'))::float8 AS score
       FROM docs
      WHERE content <@> to_bm25query($1, 'docs_bm25_idx') < 0
      ORDER BY content <@> to_bm25query($1, 'docs_bm25_idx'), id
      LIMIT $2`,
    [q, k],
  );
}

/**
 * Nearest neighbours by cosine distance. ``exact`` defeats the HNSW index on
 * purpose ("+ 0" means the ORDER BY no longer matches the indexed operator),
 * which gives the ground truth the approximate index is measured against.
 */
export async function vector(db: Db, qv: ArrayLike<number>, k: number, exact = false): Promise<Hit[]> {
  return db.query<Hit>(
    `SELECT id, (1 - (embedding <=> $1::vector))::float8 AS score
       FROM docs
      ORDER BY ${exact ? "(embedding <=> $1::vector) + 0" : "embedding <=> $1::vector"}
      LIMIT $2`,
    [toVector(qv), k],
  );
}

/**
 * Reciprocal rank fusion of full-text and vector results in one statement.
 * Each leg is ranked inside a LIMITed subquery first, so the vector leg can
 * use its HNSW index; only then is the rank numbered and fused.
 */
export async function hybrid(db: Db, q: string, qv: ArrayLike<number>, k: number, depth = 100, rrfK = 60,
                             lexical: "fts" | "bm25" = "fts"): Promise<Hit[]> {
  const lexLeg = lexical === "bm25"
    ? `lex AS (
       SELECT id, row_number() OVER (ORDER BY s, id) AS r FROM (
         SELECT id, content <@> to_bm25query($6, 'docs_bm25_idx') AS s FROM docs
          WHERE content <@> to_bm25query($6, 'docs_bm25_idx') < 0 ORDER BY s, id LIMIT $3) l),`
    : `lex AS (
       SELECT id, row_number() OVER (ORDER BY s DESC, id) AS r FROM (
         SELECT id, ts_rank_cd(tsv, tq.query) AS s FROM docs, tq
          WHERE $1 <> '' AND tsv @@ tq.query ORDER BY s DESC, id LIMIT $3) l),`;
  return db.query<Hit>(
    `WITH tq AS (SELECT to_tsquery('english', $1) AS query),
     ${lexLeg}
     sem AS (
       SELECT id, row_number() OVER (ORDER BY d, id) AS r FROM (
         SELECT id, embedding <=> $2::vector AS d FROM docs ORDER BY d LIMIT $3) v)
     SELECT id, sum(1.0 / ($4 + r))::float8 AS score
       FROM (SELECT id, r FROM lex UNION ALL SELECT id, r FROM sem) u
      GROUP BY id
      ORDER BY score DESC, id
      LIMIT $5`,
    lexical === "bm25" ? [orQuery(q), toVector(qv), depth, rrfK, k, q] : [orQuery(q), toVector(qv), depth, rrfK, k],
  );
}

/** Reference RRF in TypeScript, used by the tests to check the SQL. */
export function rrf(lists: string[][], k: number, rrfK = 60): Hit[] {
  const s = new Map<string, number>();
  for (const list of lists) list.forEach((id, i) => s.set(id, (s.get(id) ?? 0) + 1 / (rrfK + i + 1)));
  return [...s.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1))
    .slice(0, k);
}
