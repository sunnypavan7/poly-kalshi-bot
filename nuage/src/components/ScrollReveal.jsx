import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

export default function ScrollReveal({
  children,
  delay = 0,
  direction = 'up',
  className = '',
  style = {},
}) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-60px 0px' })

  const variants = {
    hidden: {
      opacity: 0,
      y: direction === 'up' ? 36 : direction === 'down' ? -36 : 0,
      x: direction === 'left' ? 36 : direction === 'right' ? -36 : 0,
      scale: direction === 'scale' ? 0.96 : 1,
    },
    show: {
      opacity: 1,
      y: 0, x: 0, scale: 1,
      transition: { duration: 0.8, delay, ease: [0.16, 1, 0.3, 1] },
    },
  }

  return (
    <motion.div
      ref={ref}
      variants={variants}
      initial="hidden"
      animate={inView ? 'show' : 'hidden'}
      className={className}
      style={style}
    >
      {children}
    </motion.div>
  )
}
