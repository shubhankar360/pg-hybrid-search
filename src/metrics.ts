/** Standard IR metrics over ranked id lists and graded relevance judgements. */

export type Qrels = Map<string, Map<string, number>>; // query id -> doc id -> grade

export function ndcgAt(ranked: string[], rel: Map<string, number>, k = 10): number {
  const dcg = ranked.slice(0, k).reduce((s, id, i) => s + ((2 ** (rel.get(id) ?? 0) - 1) / Math.log2(i + 2)), 0);
  const ideal = [...rel.values()].sort((a, b) => b - a).slice(0, k)
    .reduce((s, g, i) => s + (2 ** g - 1) / Math.log2(i + 2), 0);
  return ideal ? dcg / ideal : 0;
}

export function recallAt(ranked: string[], rel: Map<string, number>, k = 100): number {
  const relevant = [...rel.entries()].filter(([, g]) => g > 0).map(([id]) => id);
  if (!relevant.length) return 0;
  const top = new Set(ranked.slice(0, k));
  return relevant.filter((id) => top.has(id)).length / relevant.length;
}

export function mrrAt(ranked: string[], rel: Map<string, number>, k = 10): number {
  const i = ranked.slice(0, k).findIndex((id) => (rel.get(id) ?? 0) > 0);
  return i < 0 ? 0 : 1 / (i + 1);
}

export function percentile(xs: number[], p: number): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.ceil((p / 100) * s.length) - 1)];
}

export const mean = (xs: number[]): number => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
