import { useEffect, useRef } from 'react'
import { useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

const wipeVariants = {
  initial: { scaleY: 0, transformOrigin: 'bottom' },
  animate: { scaleY: 1, transformOrigin: 'bottom', transition: { duration: 0.5, ease: [0.76, 0, 0.24, 1] } },
  exit:    { scaleY: 0, transformOrigin: 'top',    transition: { duration: 0.5, ease: [0.76, 0, 0.24, 1], delay: 0.05 } },
}

export function PageTransitionOverlay() {
  const location = useLocation()
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname + '-overlay'}
        variants={wipeVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        className="fixed inset-0 z-[9999] bg-[var(--color-black)] pointer-events-none"
        aria-hidden="true"
      />
    </AnimatePresence>
  )
}

const pageVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.4, delay: 0.3, ease: 'easeOut' } },
  exit:    { opacity: 0, transition: { duration: 0.2, ease: 'easeIn' } },
}

export function PageWrapper({ children }) {
  const location = useLocation()
  const prevKey = useRef(location.pathname)

  useEffect(() => {
    if (prevKey.current !== location.pathname) {
      window.scrollTo(0, 0)
      prevKey.current = location.pathname
    }
  }, [location.pathname])

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        variants={pageVariants}
        initial="initial"
        animate="animate"
        exit="exit"
      >
        {children}
      </motion.div>
    </AnimatePresence>
  )
}
