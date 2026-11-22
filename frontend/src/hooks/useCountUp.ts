import { useEffect, useRef, useState } from 'react'

const reducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Tweens from the previous value to `target` so changing lifts reads as a change. */
export function useCountUp(target: number | undefined, durationMs = 650) {
  const [value, setValue] = useState(target)
  const fromRef = useRef(target ?? 0)

  useEffect(() => {
    if (target === undefined) return
    const from = fromRef.current
    if (reducedMotion() || from === target) {
      setValue(target)
      fromRef.current = target
      return
    }
    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs)
      const eased = 1 - Math.pow(1 - t, 4)
      const next = from + (target - from) * eased
      setValue(next)
      fromRef.current = next
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, durationMs])

  return value
}
