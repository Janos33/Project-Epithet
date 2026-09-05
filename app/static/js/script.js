document.addEventListener("DOMContentLoaded", () => {
  // --- MODAL SETUP ---
  function setupModal(openBtnSelector, modalId, closeBtnId) {
    const openBtn = document.querySelector(openBtnSelector);
    const modal = document.getElementById(modalId);
    const closeBtn = document.getElementById(closeBtnId);

    if (openBtn && modal) {
      openBtn.addEventListener("click", (e) => {
        e.preventDefault();
        modal.showModal();
      });

      if (closeBtn) {
        closeBtn.addEventListener("click", () => {
          modal.close();
        });
      }

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

  // --- TEXT INPUT PERSISTENCE ---
  const textInputs = [
    { id: "author_name", storageKey: "epithet_author" },
    { id: "poem_text", storageKey: "epithet_poem" }
  ];

  textInputs.forEach((config) => {
    const inputEl = document.getElementById(config.id);
    if (!inputEl) return;

    // Restore value
    const savedValue = localStorage.getItem(config.storageKey);
    if (savedValue !== null) {
      inputEl.value = savedValue;
    }

    // Save on input
    inputEl.addEventListener("input", () => {
      localStorage.setItem(config.storageKey, inputEl.value);
    });
  });

  // --- SLIDER SYNC & PERSISTENCE ---
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

    // 1. Restore saved value from localStorage
    const savedVal = localStorage.getItem(config.storageKey);
    if (savedVal !== null) {
      modalSlider.value = savedVal;
    }

    // 2. Initial sync function
    const syncValues = () => {
      const currentVal = modalSlider.value;

      // Sync to hidden form input
      if (formSlider) formSlider.value = currentVal;

      // Update center text display
      if (displayEl) displayEl.innerText = config.formatDisplay(currentVal);

      // Save to localStorage
      localStorage.setItem(config.storageKey, currentVal);
    };

    // 3. Run sync on page load (to apply saved values)
    syncValues();

    // 4. Run sync every time the slider is moved
    modalSlider.addEventListener("input", syncValues);
  });

  // --- PRESETS ---
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
});
