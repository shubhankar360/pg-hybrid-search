/**
 * Embedders.
 *
 * MiniLM runs locally through onnxruntime-web's WebAssembly backend rather
 * than onnxruntime-node: the native package ships no binary for Intel macOS,
 * and WASM runs the same everywhere, CI included. The tokenizer comes from
 * transformers.js; the pooling (mean over real tokens, then L2-normalise) is
 * written out here so it can be checked against the model card.
 *
 * HashEmbedder is a deterministic stand-in for tests: no model, no download,
 * same vector for the same text on every machine.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";

export interface Embedder {
  readonly name: string;
  readonly dims: number;
  embed(texts: string[]): Promise<Float32Array[]>;
}

export class HashEmbedder implements Embedder {
  readonly name = "hash";
  constructor(readonly dims = 64) {}

  async embed(texts: string[]): Promise<Float32Array[]> {
    return texts.map((t) => {
      const v = new Float32Array(this.dims);
      for (const tok of t.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
        const h = createHash("md5").update(tok).digest();
        v[h.readUInt32LE(0) % this.dims] += h[4] & 1 ? 1 : -1;
      }
      return normalise(v);
    });
  }
}

const MODEL = "Xenova/all-MiniLM-L6-v2";
const FILES = ["onnx/model.onnx", "tokenizer.json", "tokenizer_config.json"];

export class MiniLM implements Embedder {
  readonly name = "all-MiniLM-L6-v2";
  readonly dims = 384;
  private constructor(
    private tok: (texts: string[], o: object) => { input_ids: number[][]; attention_mask: number[][]; token_type_ids?: number[][] },
    private ort: typeof import("onnxruntime-web"),
    private session: import("onnxruntime-web").InferenceSession,
    private maxTokens: number,
  ) {}

  static async load(dir = join(process.cwd(), ".models", "minilm"), maxTokens = 256): Promise<MiniLM> {
    await ensureModel(dir);
    const ort = await import("onnxruntime-web");
    // The web build is the one that works without the native addon; its
    // tokenizer classes are plain JS.
    // The package's exports map only exposes its Node build; the web build
    // sits beside it in dist/, so resolve the package and swap the filename.
    const nodeBuild = import.meta.resolve("@huggingface/transformers");
    const T = await import(new URL("transformers.web.js", nodeBuild).href);
    const tokenizer = new T.BertTokenizer(
      JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8")),
      JSON.parse(readFileSync(join(dir, "tokenizer_config.json"), "utf8")),
    );
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(join(dir, "model.onnx"), { executionProviders: ["wasm"] });
    return new MiniLM((t, o) => tokenizer(t, o), ort, session, maxTokens);
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    const out: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += 16) out.push(...(await this.batch(texts.slice(i, i + 16))));
    return out;
  }

  private async batch(texts: string[]): Promise<Float32Array[]> {
    const enc = this.tok(texts, { padding: true, truncation: true, max_length: this.maxTokens, return_tensor: false });
    const B = texts.length;
    const T = enc.input_ids[0].length;
    const t = (rows: number[][]) => new this.ort.Tensor("int64", BigInt64Array.from(rows.flat(), (x) => BigInt(x)), [B, T]);
    const res = await this.session.run({
      input_ids: t(enc.input_ids),
      attention_mask: t(enc.attention_mask),
      token_type_ids: t(enc.token_type_ids ?? enc.input_ids.map((r) => r.map(() => 0))),
    });
    const hidden = res[this.session.outputNames[0]];
    const [, , D] = hidden.dims as number[];
    const data = hidden.data as Float32Array;
    return enc.attention_mask.map((mask, b) => {
      const v = new Float32Array(D);
      let n = 0;
      mask.forEach((m, tIdx) => {
        if (!m) return;
        n++;
        const off = (b * T + tIdx) * D;
        for (let d = 0; d < D; d++) v[d] += data[off + d];
      });
      for (let d = 0; d < D; d++) v[d] /= n;
      return normalise(v);
    });
  }
}

async function ensureModel(dir: string): Promise<void> {
  if (FILES.every((f) => existsSync(join(dir, f.split("/").pop()!)))) return;
  mkdirSync(dir, { recursive: true });
  for (const f of FILES) {
    const r = await fetch(`https://huggingface.co/${MODEL}/resolve/main/${f}`);
    if (!r.ok) throw new Error(`model download failed: ${f} ${r.status}`);
    writeFileSync(join(dir, f.split("/").pop()!), Buffer.from(await r.arrayBuffer()));
  }
}

export function normalise(v: Float32Array): Float32Array {
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

export const cosine = (a: Float32Array, b: Float32Array): number => a.reduce((s, x, i) => s + x * b[i], 0);
