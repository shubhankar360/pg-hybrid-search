/**
 * One small interface over two Postgres backends:
 *
 * - **PGlite** — real Postgres 18 compiled to WASM, in-process, with pgvector.
 *   Used for development, tests and the benchmark: no Docker, no server.
 * - **node-postgres** — any real server, selected by DATABASE_URL. CI runs the
 *   same test suite against `pgvector/pgvector` to prove the SQL is portable,
 *   and docker-compose runs the API this way.
 *
 * Everything above this file speaks plain SQL with $n parameters, so the two
 * are interchangeable by construction.
 */
import { readFileSync } from "node:fs";

export interface Db {
  query<T = Record<string, unknown>>(sql: string, params?: unknown[]): Promise<T[]>;
  exec(sql: string): Promise<void>;
  close(): Promise<void>;
  readonly kind: "pglite" | "postgres";
}

export async function openDb(url = process.env.DATABASE_URL): Promise<Db> {
  if (url) {
    const { default: pg } = await import("pg");
    const pool = new pg.Pool({ connectionString: url, max: 4 });
    return {
      kind: "postgres",
      async query(sql, params) {
        return (await pool.query(sql, params as unknown[])).rows;
      },
      async exec(sql) {
        await pool.query(sql);
      },
      close: () => pool.end(),
    };
  }
  const { PGlite } = await import("@electric-sql/pglite");
  const { vector } = await import("@electric-sql/pglite-pgvector");
  const db = new PGlite({ extensions: { vector } });
  return {
    kind: "pglite",
    async query(sql, params) {
      return (await db.query(sql, params as unknown[])).rows as never;
    },
    async exec(sql) {
      await db.exec(sql);
    },
    close: () => db.close(),
  };
}

export async function migrate(db: Db, dims: number): Promise<void> {
  const sql = readFileSync(new URL("./schema.sql", import.meta.url), "utf8").replaceAll("{{DIMS}}", String(dims));
  await db.exec(sql);
}

/** pgvector's text input format. */
export const toVector = (v: ArrayLike<number>): string => "[" + Array.from(v, (x) => x.toFixed(6)).join(",") + "]";
