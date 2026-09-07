export class EpithetEngine {
  constructor() {
    this.embedder = null;
  }

  // Initialize the local ML model in the browser
  async init() {
    const { pipeline, env } = await import(
      "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1"
    );

    env.allowLocalModels = false;

    this.embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }

  async getEmbedding(text) {
    const output = await this.embedder(text, { pooling: "mean", normalize: true });
    const list = output.tolist();

    return Array.isArray(list[0]) ? list[0] : list;
  }

  async processPoem(poemText) {
    if (!this.embedder) throw new Error("Engine not initialized. Call init() first.");

    const poemSplit = poemText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    const poemEmbeddings = [];
    for (const line of poemSplit) {
      poemEmbeddings.push(await this.getEmbedding(line));
    }

    return poemEmbeddings;
  }
}
