document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector(".app-form");

  if (form) {
    form.addEventListener("submit", () => {
      const submitBtn = form.querySelector(".primary-submit-btn");

      submitBtn.textContent = "Finding your words...";
      submitBtn.style.cursor = "wait";
      submitBtn.style.opacity = "0.7";
    });
  }
});
