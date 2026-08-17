import { motion, useReducedMotion } from "motion/react";
import { revealVariants } from "./motion.js";

// Reusable entrance wrapper for chart mounts: quiet fade + rise on mount, and
// replays whenever `replayKey` changes (e.g. a new fight or a new tab) — but
// NOT on every prop tweak, so dragging a line slider doesn't retrigger it.
export function ChartReveal({ children, replayKey, delay = 0, className }) {
  const reduced = useReducedMotion();
  const v = revealVariants(reduced);
  return (
    <motion.div
      key={replayKey}
      className={className}
      initial={v.initial}
      animate={v.animate}
      transition={{ ...v.transition, delay }}
      style={{ willChange: "transform, opacity" }}
    >
      {children}
    </motion.div>
  );
}
