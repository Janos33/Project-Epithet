// --- BROWSER BACK-BUTTON FIX (bfcache) ---
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
    }
  }
});

document.addEventListener("DOMContentLoaded", () => {
  // Grab window header buttons
  const profileBtn = document.getElementById("profile-btn");
  const settingsBtn = document.getElementById("settings-btn");

  // ==========================================
  // 1. RESULTS PAGE RENDERING & CLEANUP
  // ==========================================
  const resultsGrid = document.getElementById("keyword-grid");

  if (resultsGrid) {
    // ON RESULTS PAGE: Hide profile & settings buttons
    if (profileBtn) profileBtn.style.display = "none";
    if (settingsBtn) settingsBtn.style.display = "none";

    const authorSubtitle = document.getElementById("author-subtitle");
    const storedAuthor = sessionStorage.getItem("epithet_author");
    const storedResultsRaw = sessionStorage.getItem("epithet_results");

    if (authorSubtitle && storedAuthor && storedAuthor.trim() !== "") {
      authorSubtitle.textContent = `Epithet of ${storedAuthor}`;
    }

    if (storedResultsRaw) {
      try {
        const results = JSON.parse(storedResultsRaw);

        if (Array.isArray(results) && results.length > 0) {
          resultsGrid.innerHTML = "";

          results.forEach((item) => {
            const card = document.createElement("div");
            card.className = "keyword-card";

            const word = item.word !== undefined ? item.word : Array.isArray(item) ? item[0] : item;

            const score = item.score !== undefined ? item.score : Array.isArray(item) ? item[1] : 0;

            const emotions = [item.mainEmotion, item.secondaryEmotion].filter(Boolean).join(", ");

            if (item.color) {
              card.style.setProperty("--card-accent", item.color);
              card.style.backgroundColor = item.color;
            }

            card.innerHTML = `
          <h3 class="keyword-text">${word}</h3>
          <div class="hover-details">
            <div>
              <span>${(score * 100).toFixed(2)}% match</span>
            </div>
            <div>
              <span>${emotions}</span>
            </div>
          </div>
        `;

            resultsGrid.appendChild(card);
          });
        } else {
          resultsGrid.innerHTML = `<p class="empty-state">No matching neighborhoods or keywords found.</p>`;
        }
      } catch (err) {
        console.error("Error parsing stored epithet results:", err);
        resultsGrid.innerHTML = `<p class="empty-state">Error loading results.</p>`;
      }

      sessionStorage.removeItem("epithet_results");
      sessionStorage.removeItem("epithet_author");
    } else {
      resultsGrid.innerHTML = `<p class="empty-state">No data found. Please return home and submit a poem.</p>`;
    }
  } else {
    // ON MAIN PAGE: Ensure buttons are visible
    if (profileBtn) profileBtn.style.display = "";
    if (settingsBtn) settingsBtn.style.display = "";
  }

  // ==========================================
  // 2. TEXT INPUTS: PERSISTENCE
  // ==========================================
  const textInputs = [
    { id: "author_name", storageKey: "epithet_author_input" },
    { id: "poem_text", storageKey: "epithet_poem" }
  ];

  textInputs.forEach((config) => {
    const inputEl = document.getElementById(config.id);
    if (!inputEl) return;

    const savedValue = localStorage.getItem(config.storageKey);
    if (savedValue !== null) {
      inputEl.value = savedValue;
    }

    inputEl.addEventListener("input", () => {
      localStorage.setItem(config.storageKey, inputEl.value);
    });
  });

  // ==========================================
  // 3. LIVE CHARACTER COUNTER
  // ==========================================
  const poemTextarea = document.getElementById("poem_text");
  const charCounter = document.getElementById("char-counter");

  if (poemTextarea && charCounter) {
    const updateCounter = () => {
      const currentLength = poemTextarea.value.length;
      const maxLength = poemTextarea.maxLength || 25000;

      charCounter.textContent = `${currentLength.toLocaleString()} / ${maxLength.toLocaleString()}`;

      if (currentLength >= maxLength) {
        charCounter.style.color = "#e74c3c";
      } else {
        charCounter.style.color = "";
      }
    };

    updateCounter();
    poemTextarea.addEventListener("input", updateCounter);
  }

  // ==========================================
  // 4. SLIDERS: SYNC & PERSISTENCE
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
      formatDisplay: (val) => ` ${100 - val}% individual words, ${val}% similar contexts`
    },
    {
      modalId: "context-slider",
      formId: "form-context-slider",
      displayId: "context-value-display",
      storageKey: "epithet_context_slider",
      formatDisplay: (val) => `${val} Matches`
    },
    {
      modalId: "connections-slider",
      formId: "form-connections-slider",
      displayId: "connections-value-display",
      storageKey: "epithet_connections_slider",
      formatDisplay: (val) => `${val} Topics`
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

    const savedVal = localStorage.getItem(config.storageKey);
    if (savedVal !== null) {
      modalSlider.value = savedVal;
    }

    const syncValues = () => {
      const currentVal = modalSlider.value;
      if (formSlider) formSlider.value = currentVal;
      if (displayEl) displayEl.innerText = config.formatDisplay(currentVal);
      localStorage.setItem(config.storageKey, currentVal);
    };

    syncValues();
    modalSlider.addEventListener("input", syncValues);
  });

  // ==========================================
  // 5. PRESETS
  // ==========================================
  const PRESETS = {
    default: {
      emotional_weight: 60,
      theme_weight: 85,
      context_depth: 500,
      max_connections: 1000
    },
    literal: {
      emotional_weight: 40,
      theme_weight: 100,
      context_depth: 100,
      max_connections: 200
    },
    subtle: {
      emotional_weight: 60,
      theme_weight: 70,
      context_depth: 750,
      max_connections: 1500
    },
    feel: {
      emotional_weight: 80,
      theme_weight: 80,
      context_depth: 1000,
      max_connections: 2000
    }
  };

  const presetButtons = document.querySelectorAll(".preset-btn");

  presetButtons.forEach((button) => {
    button.addEventListener("click", (e) => {
      e.preventDefault();

      const presetKey = e.target.dataset.preset;
      const config = PRESETS[presetKey];

      if (!config) return;

      document.getElementById("emotional-slider").value = config.emotional_weight;
      document.getElementById("line-slider").value = config.theme_weight;
      document.getElementById("context-slider").value = config.context_depth;
      document.getElementById("connections-slider").value = config.max_connections;

      const sliderIds = [
        "emotional-slider",
        "line-slider",
        "context-slider",
        "connections-slider",
        "results-slider"
      ];
      sliderIds.forEach((id) => {
        const slider = document.getElementById(id);
        if (slider) slider.dispatchEvent(new Event("input"));
      });
    });
  });

  // ==========================================
  // 6. MODAL SETUP
  // ==========================================
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
        closeBtn.addEventListener("click", () => modal.close());
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
  setupModal(".profile-btn", "profile-modal", "close-profile-modal");

  // ==========================================
  // 7. PROFILE SELECTION & SYNC
  // ==========================================
  const profileOptionsContainer = document.getElementById("profile-options-list");
  const hiddenProfileInput = document.getElementById("selected_profile");

  function setSelectedProfile(profileId) {
    localStorage.setItem("epithet_selected_profile", profileId);
    if (hiddenProfileInput) {
      hiddenProfileInput.value = profileId;
    }
  }

  async function loadProfiles() {
    if (!profileOptionsContainer) return;

    try {
      const response = await fetch("/api/profiles");
      if (!response.ok) throw new Error("Failed to load profiles");

      const data = await response.json();
      const profiles = data.profiles || [];
      const defaultProfile = data.default;

      if (profiles.length === 0) {
        profileOptionsContainer.innerHTML = `<p class="empty-state">No profiles available.</p>`;
        return;
      }

      let activeProfile =
        localStorage.getItem("epithet_selected_profile") || defaultProfile || profiles[0].id;

      setSelectedProfile(activeProfile);

      profileOptionsContainer.innerHTML = "";

      profiles.forEach((p) => {
        const isChecked = p.id === activeProfile;

        const card = document.createElement("label");
        card.className = `profile-option-card ${isChecked ? "active" : ""}`;
        card.innerHTML = `
          <input 
            type="radio" 
            name="profile_choice" 
            value="${p.id}" 
            ${isChecked ? "checked" : ""} 
          />
          <div class="profile-option-details">
            <span class="profile-option-name">${p.name || p.id}</span>
            ${p.description ? `<p class="profile-option-desc">${p.description}</p>` : ""}
          </div>
        `;

        card.querySelector("input").addEventListener("change", (e) => {
          document
            .querySelectorAll(".profile-option-card")
            .forEach((c) => c.classList.remove("active"));
          card.classList.add("active");
          setSelectedProfile(e.target.value);
        });

        profileOptionsContainer.appendChild(card);
      });
    } catch (err) {
      console.error("Error loading profiles:", err);
      profileOptionsContainer.innerHTML = `<p class="empty-state">Unable to load profiles.</p>`;
    }
  }

  loadProfiles();
});
