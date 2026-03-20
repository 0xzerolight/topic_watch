/**
 * Theme switcher for Topic Watch.
 *
 * THEMES array is the single source of truth for available themes.
 * To add a new theme:
 *   1. Add its CSS block in themes.css under [data-theme="your-theme"]
 *   2. Add { id: "your-theme", name: "Your Theme" } to the THEMES array below
 */
(function () {
  "use strict";

  var THEMES = [
    { id: "light", name: "Light", dark: false },
    { id: "dark", name: "Dark", dark: false },
    { id: "nord", name: "Nord", dark: true },
    { id: "dracula", name: "Dracula", dark: true },
    { id: "solarized", name: "Solarized Dark", dark: true },
    { id: "high-contrast", name: "High Contrast", dark: true },
    { id: "tokyo-night", name: "Tokyo Night", dark: true },
  ];

  var STORAGE_KEY = "topic-watch-theme";
  var DEFAULT_THEME = "light";

  function getSavedTheme() {
    try {
      var saved = localStorage.getItem(STORAGE_KEY);
      if (saved && THEMES.some(function (t) { return t.id === saved; })) {
        return saved;
      }
    } catch (e) {
      // localStorage may be unavailable
    }
    return DEFAULT_THEME;
  }

  function applyTheme(themeId) {
    var theme = THEMES.find(function (t) { return t.id === themeId; });
    if (theme && theme.dark) {
      document.documentElement.setAttribute("data-theme", "dark");
      document.documentElement.setAttribute("data-custom-theme", themeId);
    } else {
      document.documentElement.setAttribute("data-theme", themeId);
      document.documentElement.removeAttribute("data-custom-theme");
    }
    try {
      localStorage.setItem(STORAGE_KEY, themeId);
    } catch (e) {
      // Ignore storage errors
    }
  }

  function populateSelect(select, currentTheme) {
    THEMES.forEach(function (theme) {
      var option = document.createElement("option");
      option.value = theme.id;
      option.textContent = theme.name;
      if (theme.id === currentTheme) {
        option.selected = true;
      }
      select.appendChild(option);
    });
  }

  // Apply saved theme immediately (before body renders) to prevent FOUC
  var currentTheme = getSavedTheme();
  applyTheme(currentTheme);

  // Once DOM is ready, wire up the theme selector
  document.addEventListener("DOMContentLoaded", function () {
    var select = document.getElementById("theme-select");
    if (select) {
      populateSelect(select, currentTheme);
      select.addEventListener("change", function () {
        applyTheme(this.value);
      });
    }
  });
})();
