import { EpithetEngine } from "./vector-engine.js";

// --- TEXT PREPROCESSING FUNCTION ---
function formatPoemText(rawText) {
  const LINE_LENGTH_THRESHOLD = 15;
  const MAX_LINE_LENGTH = 50;

  const rawLines = rawText.split(/\r?\n/);
  const processedLines = [];
  let currentCombined = "";

  for (let i = 0; i < rawLines.length; i++) {
    const rawLine = rawLines[i].trim();
    if (!rawLine) continue;

    let subLines = [];

    if (rawLine.length > MAX_LINE_LENGTH) {
      const parts = rawLine.split(/(?<=[,;:!?.])\s+/);

      for (let p of parts) {
        p = p.trim();
        if (!p) continue;

        while (p.length > MAX_LINE_LENGTH) {
          let splitIndex = p.lastIndexOf(" ", MAX_LINE_LENGTH);
          if (splitIndex === -1) splitIndex = MAX_LINE_LENGTH;

          subLines.push(p.slice(0, splitIndex).trim());
          p = p.slice(splitIndex).trim();
        }
        if (p.length > 0) subLines.push(p);
      }
    } else {
      subLines = [rawLine];
    }

    for (const line of subLines) {
      if (line.length < LINE_LENGTH_THRESHOLD) {
        currentCombined += currentCombined ? " " + line : line;
      } else {
        if (currentCombined) {
          processedLines.push(currentCombined);
          currentCombined = "";
        }
        processedLines.push(line);
      }
    }
  }

  if (currentCombined) {
    if (processedLines.length > 0 && currentCombined.length < LINE_LENGTH_THRESHOLD) {
      // Append short trailing text to the previous line so no words are lost
      processedLines[processedLines.length - 1] += " " + currentCombined;
    } else {
      // Push as a standalone line (handles single words or short inputs)
      processedLines.push(currentCombined);
    }
  }

  // Safety fallback: if processing yields nothing, return the trimmed raw text
  if (processedLines.length === 0 && rawText.trim().length > 0) {
    return rawText.trim();
  }

  return processedLines.join("\n");
}

document.addEventListener("DOMContentLoaded", () => {
  // --- 1. RESULTS PAGE LOGIC ---
  const authorSubtitle = document.getElementById("author-subtitle");
  if (authorSubtitle) {
    const storedAuthor = sessionStorage.getItem("epithet_author");
    if (storedAuthor && storedAuthor.trim() !== "") {
      authorSubtitle.textContent = `The Epithet of ${storedAuthor}`;
    }
  }

  // --- 2. FORM PAGE LOGIC ---
  const form = document.querySelector(".app-form");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const oldInput = form.querySelector('input[name="client_poem_embeddings"]');
      if (oldInput) oldInput.remove();

      const submitBtn = form.querySelector(".primary-submit-btn");
      const poemText = document.getElementById("poem_text").value;
      const authorName = document.getElementById("author_name").value;

      sessionStorage.setItem("epithet_author", authorName);

      submitBtn.textContent = "Loading Engine...";
      submitBtn.style.cursor = "wait";
      submitBtn.style.opacity = "0.7";
      submitBtn.disabled = true;

      try {
        const engine = new EpithetEngine();
        await engine.init();

        const processedPoemText = formatPoemText(poemText);
        const poemEmbeddings = await engine.processPoem(processedPoemText);

        const embeddingsInput = document.createElement("input");
        embeddingsInput.type = "hidden";
        embeddingsInput.name = "client_poem_embeddings";

        embeddingsInput.value = JSON.stringify(poemEmbeddings, (key, val) => {
          return typeof val === "number" ? Number(val.toFixed(5)) : val;
        });

        form.appendChild(embeddingsInput);

        submitBtn.textContent = "Processing Data...";
        form.submit();
      } catch (error) {
        console.error("Error processing poem:", error);

        const failedInput = form.querySelector('input[name="client_poem_embeddings"]');
        if (failedInput) failedInput.remove();

        submitBtn.textContent = "Error processing";
        submitBtn.style.cursor = "pointer";
        submitBtn.style.opacity = "1";
        submitBtn.disabled = false;
      }
    });
  }
});

// --- 3. BROWSER BACK-BUTTON FIX (bfcache) ---
window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    const form = document.querySelector(".app-form");
    if (form) {
      const submitBtn = form.querySelector(".primary-submit-btn");
      if (submitBtn) {
        submitBtn.textContent = "Find your words";
        submitBtn.style.cursor = "pointer";
        submitBtn.style.opacity = "1";
        submitBtn.disabled = false;
      }

      const leftoverInput = form.querySelector('input[name="client_poem_embeddings"]');
      if (leftoverInput) leftoverInput.remove();
    }
  }
});
