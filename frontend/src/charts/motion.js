// Shared motion tokens for chart entrance/loading animation.
// Reuses the easing already used on the app's progress-bar transitions
// (see fightpath.css) so chart motion feels consistent with the rest of the UI.
export const EASE_REVEAL = [0.2, 0.7, 0.2, 1];

// Container fade + rise, used by ChartReveal on every chart mount.
export function revealVariants(reduced) {
  if (reduced) {
    return {
      initial: { opacity: 1, y: 0 },
      animate: { opacity: 1, y: 0 },
      transition: { duration: 0 },
    };
  }
  return {
    initial: { opacity: 0, y: 10 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, ease: EASE_REVEAL },
  };
}

// Curve draw-in (SVG pathLength trace / clip-rect wipe).
export function traceTransition(reduced) {
  return reduced ? { duration: 0 } : { duration: 0.9, ease: "easeInOut" };
}

// Short opacity fade for fills that can't be clip-wiped cleanly (e.g. the
// duration PDF's under/over/full fill regions, which are mutually exclusive
// and computed conditionally on lineX).
export function fadeTransition(reduced, duration = 0.5) {
  return reduced ? { duration: 0 } : { duration, ease: EASE_REVEAL };
}

// Matchup-header fighter names sliding in from their own corner on fight
// change: side "a" enters from the left, "b" from the right.
export function cornerSlideVariants(side, reduced, delay = 0) {
  if (reduced) {
    return {
      initial: { opacity: 1, x: 0 },
      animate: { opacity: 1, x: 0 },
      transition: { duration: 0 },
    };
  }
  return {
    initial: { opacity: 0, x: side === "a" ? -14 : 14 },
    animate: { opacity: 1, x: 0 },
    transition: { duration: 0.5, ease: EASE_REVEAL, delay },
  };
}
