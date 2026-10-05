-- One table holds both indexes: a stored, weighted tsvector for full-text
-- search (title outranks body) and a pgvector embedding with an HNSW index.
-- Keeping them in one row means hybrid search is a single query and a
-- document can never exist in one index but not the other.
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

CREATE TABLE IF NOT EXISTS docs (
  id        text PRIMARY KEY,
  title     text NOT NULL DEFAULT '',
  body      text NOT NULL,
  tsv       tsvector GENERATED ALWAYS AS (
              setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
              setweight(to_tsvector('english', body), 'B')
            ) STORED,
  embedding vector({{DIMS}}) NOT NULL
);

CREATE INDEX IF NOT EXISTS docs_tsv_idx ON docs USING gin (tsv);
CREATE INDEX IF NOT EXISTS docs_embedding_idx ON docs USING hnsw (embedding vector_cosine_ops);
