import { EpithetEngine } from "./vector-engine.js";

let isWaitingInQueue = false;

// Warn user before closing/refreshing the tab while holding embeddings in queue
window.addEventListener("beforeunload", (e) => {
  if (isWaitingInQueue) {
    e.preventDefault();
  }
});

// --- TEXT PREPROCESSING FUNCTION ---
function formatPoemText(rawText) {
  const LINE_LENGTH_THRESHOLD = 15;
  const MAX_LINE_LENGTH = 250;

  const rawLines = rawText.split(/\r\n|\r|\n/);
  const processedLines = [];
  let currentCombined = "";

  for (let i = 0; i < rawLines.length; i++) {
    const rawLine = rawLines[i].trim();
    if (!rawLine) continue;

    let subLines = [];

    if (rawLine.length > MAX_LINE_LENGTH) {
      const parts = rawLine
        .split(/(?<=[,;:!?.])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

      for (let p of parts) {
        while (p.length > MAX_LINE_LENGTH) {
          let splitIndex = p.slice(0, MAX_LINE_LENGTH).lastIndexOf(" ");
          if (splitIndex === -1) {
            splitIndex = MAX_LINE_LENGTH;
          }

          subLines.push(p.slice(0, splitIndex).trim());
          p = p.slice(splitIndex).trim();
        }
        if (p.length > 0) {
          subLines.push(p);
        }
      }
    } else {
      subLines = [rawLine];
    }

    // Recombine tiny lines together to give the model more context
    for (const line of subLines) {
      currentCombined = currentCombined ? currentCombined + " " + line : line;

      if (currentCombined.length >= LINE_LENGTH_THRESHOLD) {
        processedLines.push(currentCombined);
        currentCombined = "";
      }
    }
  }

  // Handle lingering combined fragments at the end of the text
  if (currentCombined) {
    if (processedLines.length > 0 && currentCombined.length < LINE_LENGTH_THRESHOLD) {
      processedLines[processedLines.length - 1] = (
        processedLines[processedLines.length - 1] +
        " " +
        currentCombined.trim()
      ).trim();
    } else {
      processedLines.push(currentCombined);
    }
    currentCombined = "";
  }

  if (processedLines.length === 0 && rawText.trim().length > 0) {
    return rawText.trim();
  }

  return processedLines.join("\n");
}

// --- QUEUE POLLING LOGIC ---
async function startQueuePolling(ticketId, payload, submitBtn) {
  isWaitingInQueue = true;

  const pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/queue/status/${ticketId}`);
      if (!res.ok) throw new Error("Failed to check queue status");

      const statusData = await res.json();

      if (statusData.status === "waiting") {
        submitBtn.textContent = `In Queue: Spot #${statusData.position}`;
      } else if (statusData.status === "ready") {
        clearInterval(pollInterval);
        submitBtn.textContent = "Processing Data...";
        await sendFinalPayload(ticketId, payload, submitBtn);
      } else if (statusData.status === "expired") {
        clearInterval(pollInterval);
        isWaitingInQueue = false;
        resetSubmitButton(submitBtn, "Session Expired. Try Again.");
      }
    } catch (error) {
      clearInterval(pollInterval);
      isWaitingInQueue = false;
      console.error("Queue Polling Error:", error);
      resetSubmitButton(submitBtn, "Queue Error. Try Again.");
    }
  }, 2500);
}

// --- FINAL PAYLOAD SUBMISSION ---
async function sendFinalPayload(ticketId, payload, submitBtn) {
  try {
    const response = await fetch("/api/queue/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticket_id: ticketId,
        ...payload
      })
    });

    if (!response.ok) throw new Error("Processing failed on server.");

    const resultData = await response.json();

    isWaitingInQueue = false;
    submitBtn.textContent = "Complete! Loading...";

    // 1. Save data to sessionStorage so ui-controller.js can render it
    sessionStorage.setItem("epithet_results", JSON.stringify(resultData));

    // 2. Redirect to the results page
    window.location.href = "/results";
  } catch (error) {
    isWaitingInQueue = false;
    console.error("Submission Error:", error);
    resetSubmitButton(submitBtn, "Error Processing Poem");
  }
}

function resetSubmitButton(btn, text) {
  btn.textContent = text;
  btn.style.cursor = "pointer";
  btn.style.opacity = "1";
  btn.disabled = false;
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector(".app-form");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const submitBtn = form.querySelector(".primary-submit-btn");
      const poemText = document.getElementById("poem_text").value;
      const authorName = document.getElementById("author_name").value;

      // Save author early in case of failure or reload
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

        // FIXED: Matched these IDs to your hidden input HTML IDs
        const payload = {
          embeddings: poemEmbeddings,
          emotional_weight: document.getElementById("form-closeness-slider")?.value || 60,
          line_weight: document.getElementById("form-granularity-slider")?.value || 85,
          context_line_amount: document.getElementById("form-context-slider")?.value || 10,
          neighborhood_amount: document.getElementById("form-connections-slider")?.value || 40,
          max_results: document.getElementById("form-results-slider")?.value || 20
        };

        submitBtn.textContent = "Joining Queue...";
        const joinResponse = await fetch("/api/queue/join", {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        });

        if (!joinResponse.ok) throw new Error("Could not join server queue.");

        const { ticket_id, position } = await joinResponse.json();

        if (position === 0) {
          submitBtn.textContent = "Processing Data...";
          await sendFinalPayload(ticket_id, payload, submitBtn);
        } else {
          submitBtn.textContent = `In Queue: Spot #${position}`;
          startQueuePolling(ticket_id, payload, submitBtn);
        }
      } catch (error) {
        console.error("Error in submission pipeline:", error);
        resetSubmitButton(submitBtn, "Error Processing");
      }
    });
  }
});
