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

/**
 * ``schema`` isolates callers that share one server: every pooled connection
 * gets it first on its search_path. Test files run in parallel, and on a
 * shared server two of them creating the same table at once is a race that
 * an in-process database never shows.
 */
export async function openDb(url = process.env.DATABASE_URL, opts: { schema?: string } = {}): Promise<Db> {
  if (url) {
    const { default: pg } = await import("pg");
    const schema = opts.schema && /^[a-z_][a-z0-9_]*$/.test(opts.schema) ? opts.schema : undefined;
    if (schema) {
      const admin = new pg.Client({ connectionString: url });
      await admin.connect();
      await admin.query(`CREATE SCHEMA IF NOT EXISTS ${schema}`);
      await admin.end();
    }
    const pool = new pg.Pool({ connectionString: url, max: 4, ...(schema ? { options: `-c search_path=${schema},public` } : {}) });
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
  // Two clients racing CREATE EXTENSION IF NOT EXISTS can both pass the
  // "not exists" check; the loser gets a unique violation. It is harmless:
  // the extension exists either way. SCHEMA public matters too: without it
  // the extension lands in the first schema on the search_path, and every
  // other schema then fails with 'type "vector" does not exist'.
  try {
    await db.exec("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public");
  } catch (e) {
    if (!["23505", "42710"].includes((e as { code?: string }).code ?? "")) throw e;
  }
  const sql = readFileSync(new URL("./schema.sql", import.meta.url), "utf8").replaceAll("{{DIMS}}", String(dims));
  await db.exec(sql);
}

/** pgvector's text input format. */
export const toVector = (v: ArrayLike<number>): string => "[" + Array.from(v, (x) => x.toFixed(6)).join(",") + "]";
