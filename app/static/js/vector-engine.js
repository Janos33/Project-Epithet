export class EpithetEngine {
  constructor() {
    this.embedder = null;
  }

  // Initialize the local ML model in the browser
  async init() {
    // Dynamically import Transformers.js from CDN
    const { pipeline, env } = await import(
      "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1"
    );

    // Use HuggingFace hub, don't look for local files
    env.allowLocalModels = false;

    // Load a lightweight embedding model (must match your Python model exactly)
    // This downloads once (~22MB) and caches in the browser.
    this.embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }

  // Helper: Generate a normalized vector embedding for a string
  async getEmbedding(text) {
    const output = await this.embedder(text, { pooling: "mean", normalize: true });
    const list = output.tolist();

    // Transformers.js often adds a batch dimension [1, 384], so we grab the inner array
    return Array.isArray(list[0]) ? list[0] : list;
  }

  /**
   * Splits the poem into lines and returns a 2D array of their embeddings.
   */
  async processPoem(poemText) {
    if (!this.embedder) throw new Error("Engine not initialized. Call init() first.");

    // 1. Split into lines & remove empty ones
    const poemSplit = poemText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    // 2. Embed the poem lines
    const poemEmbeddings = [];
    for (const line of poemSplit) {
      poemEmbeddings.push(await this.getEmbedding(line));
    }

    // 3. Return the raw vectors (e.g., [[0.12, -0.05, ...], [0.88, 0.22, ...]])
    return poemEmbeddings;
  }
}
