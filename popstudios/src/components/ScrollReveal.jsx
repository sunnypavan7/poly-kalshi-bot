import { useRef } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'

export default function ScrollReveal({
  children,
  delay = 0,
  direction = 'up',
  distance = 40,
  duration = 0.7,
  className = '',
  once = true,
}) {
  const ref = useRef(null)
  const inView = useInView(ref, { once, margin: '-80px 0px' })
  const reduced = useReducedMotion()

  const offsets = {
    up:    { y: distance, x: 0 },
    down:  { y: -distance, x: 0 },
    left:  { x: distance, y: 0 },
    right: { x: -distance, y: 0 },
  }

  const initial = reduced ? { opacity: 0 } : { opacity: 0, ...offsets[direction] }
  const animate = inView
    ? { opacity: 1, x: 0, y: 0, transition: { duration: reduced ? 0.08 : duration, delay, ease: [0.16, 1, 0.3, 1] } }
    : initial

  return (
    <motion.div ref={ref} initial={initial} animate={animate} className={className}>
      {children}
    </motion.div>
  )
}
