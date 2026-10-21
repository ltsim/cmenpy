/* cmenpy header behaviour: auto-hide on scroll, ⌘K search, platform hint.
 * Loaded after Material's bundle; the header element survives instant
 * navigation, so the listeners are bound once. */
(() => {
  "use strict";

  const header = document.querySelector(".md-header");
  if (!header) return;

  const HIDDEN = "md-header--hidden";
  const reveal = () => header.classList.remove(HIDDEN);

  /* Auto-hide: only while scrolling down, never with focus in the bar or the
   * search overlay open. */
  let lastY = window.scrollY;
  let frame = 0;

  const maybeHide = () => {
    frame = 0;
    const y = window.scrollY;

    if (Math.abs(y - lastY) < 8) return;

    const hide =
      y > lastY &&
      y > 160 &&
      !document.querySelector("#__search:checked") &&
      !header.contains(document.activeElement);

    header.classList.toggle(HIDDEN, hide);
    lastY = y;
  };

  window.addEventListener(
    "scroll",
    () => {
      if (!frame) frame = window.requestAnimationFrame(maybeHide);
    },
    { passive: true }
  );

  header.addEventListener("focusin", reveal);

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") reveal();
  });

  /* ⌘K / Ctrl-K opens the search overlay. */
  window.addEventListener("keydown", (event) => {
    if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
    if (event.key.toLowerCase() !== "k") return;

    const trigger = document.querySelector('label[for="__search"]');
    if (!trigger) return;

    event.preventDefault();
    reveal();
    trigger.click();
  });

  /* Show the shortcut that applies to this platform. */
  const form = document.querySelector(".md-search__form");
  if (form) {
    const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
    form.dataset.hint = isMac ? "⌘K" : "Ctrl K";
  }

  /* Instant navigation resets the scroll position; never leave the bar hidden. */
  if (typeof document$ !== "undefined") {
    document$.subscribe(reveal);
  }
})();
