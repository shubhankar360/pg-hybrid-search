// Download and smoke-test the embedding model (used at Docker build time).
import { MiniLM } from "./embed.js";

const m = await MiniLM.load();
const [v] = await m.embed(["warm-up"]);
console.log(`${m.name} ready, ${v.length} dims`);
