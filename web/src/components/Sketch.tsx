import * as Switch from "@radix-ui/react-switch";
import { type ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { hashString, sketchCircle, sketchRect, sketchUnderline } from "../lib/sketch";

function useSize<T extends HTMLElement>(): [React.RefObject<T>, { w: number; h: number }] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => {
      const w = Math.round(el.offsetWidth);
      const h = Math.round(el.offsetHeight);
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }));
    };
    update();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

/** Wobbly pencil rectangle drawn behind/around its parent (parent must be position: relative). */
export function SketchBorder({ seed, strokeWidth = 1.6 }: { seed: string; strokeWidth?: number }) {
  const [ref, { w, h }] = useSize<HTMLSpanElement>();
  const d = useMemo(() => (w && h ? sketchRect(w, h, hashString(seed)) : ""), [w, h, seed]);
  return (
    <span ref={ref} aria-hidden="true" style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
      {d && (
        <svg className="sketch-border" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
          <path d={d} fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </span>
  );
}

export function SketchButton({
  children,
  seed,
  className = "icon-btn",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { seed: string }) {
  return (
    <button type="button" className={className} {...props}>
      <SketchBorder seed={seed} />
      {children}
    </button>
  );
}

export function SketchUnderline({ children, seed }: { children: ReactNode; seed: string }) {
  const [ref, { w }] = useSize<HTMLSpanElement>();
  const d = useMemo(() => (w ? sketchUnderline(w, hashString(seed)) : ""), [w, seed]);
  return (
    <span ref={ref} style={{ position: "relative", display: "inline-block" }}>
      {children}
      {d && (
        <svg aria-hidden="true" width={w} height={8} style={{ position: "absolute", left: 0, bottom: -4, overflow: "visible" }}>
          <path d={d} fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" />
        </svg>
      )}
    </span>
  );
}

export function SketchCircle({ children, seed }: { children: ReactNode; seed: string }) {
  const [ref, { w, h }] = useSize<HTMLSpanElement>();
  const d = useMemo(() => (w && h ? sketchCircle(w + 16, h + 8, hashString(seed)) : ""), [w, h, seed]);
  return (
    // Side margins leave room for the circle's 8px overshoot so it doesn't run into neighbouring words.
    <span ref={ref} style={{ position: "relative", display: "inline-block", margin: "0 9px" }}>
      {children}
      {d && (
        <svg aria-hidden="true" width={w + 16} height={h + 8} style={{ position: "absolute", left: -8, top: -4, overflow: "visible", pointerEvents: "none" }}>
          <path d={d} fill="none" stroke="currentColor" strokeWidth={1.4} strokeLinecap="round" />
        </svg>
      )}
    </span>
  );
}

/** Radix Switch rendered as a hand-drawn checkbox. */
export function SketchCheck({
  id,
  checked,
  onCheckedChange,
  disabled,
  label,
}: {
  id: string;
  checked: boolean;
  onCheckedChange: (v: boolean) => void;
  disabled?: boolean;
  label: string;
}) {
  const box = useMemo(() => sketchRect(28, 28, hashString(id), 1.8, 3), [id]);
  return (
    <div className="check-row">
      <Switch.Root id={id} className="sketch-check" checked={checked} onCheckedChange={onCheckedChange} disabled={disabled} aria-label={label}>
        <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
          <path d={box} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <path className="tick" d="M6 15 L12 21 L23 6" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <Switch.Thumb />
      </Switch.Root>
      <label htmlFor={id} style={{ cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.55 : 1 }}>
        {label}
      </label>
    </div>
  );
}

export function PadlockIcon({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <path d="M10.5 15 v-4.5 a5.6 5.6 0 0 1 11.2 0.3 V15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M7.4 15.2 L24.8 14.8 L25.1 27.3 L7.1 27.6 Z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      <path d="M16 19.6 v3.4" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="16" cy="19.4" r="1.5" fill="currentColor" />
    </svg>
  );
}

export function useReducedMotion(): boolean {
  const query = "(prefers-reduced-motion: reduce)";
  const get = () => {
    try {
      return window.matchMedia(query).matches;
    } catch {
      return false;
    }
  };
  const [reduced, setReduced] = useState(get);
  useEffect(() => {
    let mql: MediaQueryList;
    try {
      mql = window.matchMedia(query);
    } catch {
      return;
    }
    const onChange = () => setReduced(mql.matches);
    mql.addEventListener?.("change", onChange);
    return () => mql.removeEventListener?.("change", onChange);
  }, []);
  return reduced;
}
