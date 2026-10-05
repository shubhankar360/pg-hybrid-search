import type { Db } from "./db.js";
import { toVector } from "./db.js";
import type { Doc } from "./data.js";

/** Upsert in multi-row batches: one round trip per 200 documents. */
export async function upsertDocs(db: Db, docs: Doc[], vectors: ArrayLike<number>[], batch = 200): Promise<void> {
  if (docs.length !== vectors.length) throw new Error("one vector per document");
  for (let i = 0; i < docs.length; i += batch) {
    const rows = docs.slice(i, i + batch);
    const params: unknown[] = [];
    const values = rows.map((d, j) => {
      params.push(d.id, d.title, d.body, toVector(vectors[i + j]));
      const n = j * 4;
      return `($${n + 1}, $${n + 2}, $${n + 3}, $${n + 4}::vector)`;
    });
    await db.query(
      `INSERT INTO docs (id, title, body, embedding) VALUES ${values.join(", ")}
       ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, body = EXCLUDED.body, embedding = EXCLUDED.embedding`,
      params,
    );
  }
}
