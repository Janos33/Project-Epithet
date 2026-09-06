document.addEventListener("DOMContentLoaded", () => {
  // ==========================================
  // 1. TEXT INPUTS: PERSISTENCE
  // ==========================================
  const textInputs = [
    { id: "author_name", storageKey: "epithet_author" },
    { id: "poem_text", storageKey: "epithet_poem" }
  ];

  textInputs.forEach((config) => {
    const inputEl = document.getElementById(config.id);
    if (!inputEl) return;

    // Restore saved value on page load
    const savedValue = localStorage.getItem(config.storageKey);
    if (savedValue !== null) {
      inputEl.value = savedValue;
    }

    // Save to localStorage whenever user types
    inputEl.addEventListener("input", () => {
      localStorage.setItem(config.storageKey, inputEl.value);
    });
  });

  // ==========================================
  // 2. LIVE CHARACTER COUNTER
  // ==========================================
  const poemTextarea = document.getElementById("poem_text");
  const charCounter = document.getElementById("char-counter");

  if (poemTextarea && charCounter) {
    const updateCounter = () => {
      const currentLength = poemTextarea.value.length;
      const maxLength = poemTextarea.maxLength || 50000;

      charCounter.textContent = `${currentLength.toLocaleString()} / ${maxLength.toLocaleString()}`;

      // Turn red if maxed out, revert to normal if not
      if (currentLength >= maxLength) {
        charCounter.style.color = "#e74c3c";
      } else {
        charCounter.style.color = "";
      }
    };

    // Run immediately to count any text restored from localStorage
    updateCounter();

    // Update dynamically as the user types
    poemTextarea.addEventListener("input", updateCounter);
  }

  // ==========================================
  // 3. SLIDERS: SYNC & PERSISTENCE
  // ==========================================
  const sliderConfigs = [
    {
      modalId: "emotional-slider",
      formId: "form-closeness-slider",
      displayId: "emotional-value-display",
      storageKey: "epithet_emotional_slider",
      formatDisplay: (val) => `${100 - val}% literal meaning, ${val}% emotional resonance`
    },
    {
      modalId: "line-slider",
      formId: "form-granularity-slider",
      displayId: "line-value-display",
      storageKey: "epithet_find_slider",
      formatDisplay: (val) => ` ${100 - val}% specific words, ${val}% broader themes`
    },
    {
      modalId: "context-slider",
      formId: "form-context-slider",
      displayId: "context-value-display",
      storageKey: "epithet_context_slider",
      formatDisplay: (val) => `${val} Lines`
    },
    {
      modalId: "connections-slider",
      formId: "form-connections-slider",
      displayId: "connections-value-display",
      storageKey: "epithet_connections_slider",
      formatDisplay: (val) => `${val} Nodes`
    },
    {
      modalId: "results-slider",
      formId: "form-results-slider",
      displayId: "results-value-display",
      storageKey: "epithet_results_slider",
      formatDisplay: (val) => `${val} Keywords`
    }
  ];

  sliderConfigs.forEach((config) => {
    const modalSlider = document.getElementById(config.modalId);
    const formSlider = document.getElementById(config.formId);
    const displayEl = document.getElementById(config.displayId);

    if (!modalSlider) return;

    // Restore saved value from localStorage
    const savedVal = localStorage.getItem(config.storageKey);
    if (savedVal !== null) {
      modalSlider.value = savedVal;
    }

    // Sync function: updates UI, hidden forms, and local storage
    const syncValues = () => {
      const currentVal = modalSlider.value;
      if (formSlider) formSlider.value = currentVal;
      if (displayEl) displayEl.innerText = config.formatDisplay(currentVal);
      localStorage.setItem(config.storageKey, currentVal);
    };

    // Run sync on page load to apply saved values
    syncValues();

    // Run sync every time the slider is moved
    modalSlider.addEventListener("input", syncValues);
  });

  // ==========================================
  // 4. PRESETS
  // ==========================================
  const PRESETS = {
    default: {
      emotional_weight: 60,
      theme_weight: 85,
      context_depth: 10,
      max_connections: 40,
      max_results: 20
    },
    literal: {
      emotional_weight: 40,
      theme_weight: 70,
      context_depth: 5,
      max_connections: 10,
      max_results: 20
    },
    subtle: {
      emotional_weight: 60,
      theme_weight: 100,
      context_depth: 35,
      max_connections: 60,
      max_results: 40
    },
    feel: {
      emotional_weight: 80,
      theme_weight: 90,
      context_depth: 20,
      max_connections: 100,
      max_results: 40
    }
  };

  const presetButtons = document.querySelectorAll(".preset-btn");

  presetButtons.forEach((button) => {
    button.addEventListener("click", (e) => {
      e.preventDefault();

      const presetKey = e.target.dataset.preset;
      const config = PRESETS[presetKey];

      if (!config) return;

      // Apply the preset values to the visible modal sliders
      document.getElementById("emotional-slider").value = config.emotional_weight;
      document.getElementById("line-slider").value = config.theme_weight;
      document.getElementById("context-slider").value = config.context_depth;
      document.getElementById("connections-slider").value = config.max_connections;
      document.getElementById("results-slider").value = config.max_results;

      // Safely dispatch the 'input' event to trigger UI updates and localStorage saves
      const sliderIds = [
        "emotional-slider",
        "line-slider",
        "context-slider",
        "connections-slider",
        "results-slider"
      ];
      sliderIds.forEach((id) => {
        const slider = document.getElementById(id);
        if (slider) {
          slider.dispatchEvent(new Event("input"));
        }
      });
    });
  });

  // ==========================================
  // 5. MODAL SETUP
  // ==========================================
  function setupModal(openBtnSelector, modalId, closeBtnId) {
    const openBtn = document.querySelector(openBtnSelector);
    const modal = document.getElementById(modalId);
    const closeBtn = document.getElementById(closeBtnId);

    if (openBtn && modal) {
      // Open modal
      openBtn.addEventListener("click", (e) => {
        e.preventDefault();
        modal.showModal();
      });

      // Close modal via button
      if (closeBtn) {
        closeBtn.addEventListener("click", () => {
          modal.close();
        });
      }

      // Close modal by clicking outside the bounds
      modal.addEventListener("click", (e) => {
        const bounds = modal.getBoundingClientRect();
        if (
          e.clientX < bounds.left ||
          e.clientX > bounds.right ||
          e.clientY < bounds.top ||
          e.clientY > bounds.bottom
        ) {
          modal.close();
        }
      });
    }
  }

  setupModal(".info-btn", "info-modal", "close-info-modal");
  setupModal(".settings-btn", "settings-modal", "close-settings-modal");
});
