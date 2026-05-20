import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function Loader() {
  const [visible, setVisible] = useState(true)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const t1 = setTimeout(() => setProgress(100), 80)
    const t2 = setTimeout(() => setVisible(false), 1900)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          className="loader"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.7, ease: [0.25, 0.1, 0.0, 1.0] }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.15 }}
          >
            <div className="loader-word">Nuage</div>
            <div className="loader-sub">Private Estate · Zermatt, Switzerland</div>
            <div className="loader-bar-wrap" style={{ marginTop: 20 }}>
              <motion.div
                className="loader-bar-fill"
                animate={{ width: `${progress}%` }}
                transition={{ duration: 1.5, ease: [0.25, 0.1, 0.0, 1.0] }}
                style={{ width: 0 }}
              />
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
