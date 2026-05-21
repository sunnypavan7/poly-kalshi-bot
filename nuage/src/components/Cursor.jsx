import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function Cursor() {
  const [pos, setPos] = useState({ x: -100, y: -100 })
  const [big, setBig] = useState(false)

  useEffect(() => {
    const move = e => setPos({ x: e.clientX, y: e.clientY })
    const over = e => {
      if (e.target.closest('a, button, img, .cursor-grow')) setBig(true)
    }
    const out = e => {
      if (e.target.closest('a, button, img, .cursor-grow')) setBig(false)
    }
    document.addEventListener('mousemove', move)
    document.addEventListener('mouseover', over)
    document.addEventListener('mouseout', out)
    return () => {
      document.removeEventListener('mousemove', move)
      document.removeEventListener('mouseover', over)
      document.removeEventListener('mouseout', out)
    }
  }, [])

  return (
    <>
      <motion.div
        className="cursor-dot"
        animate={{ x: pos.x - 4, y: pos.y - 4, scale: big ? 0 : 1 }}
        transition={{ type: 'tween', duration: 0.025, ease: 'linear' }}
      />
      <motion.div
        className="cursor-ring"
        animate={{
          x: pos.x - (big ? 28 : 15),
          y: pos.y - (big ? 28 : 15),
          width: big ? 56 : 30,
          height: big ? 56 : 30,
        }}
        transition={{ type: 'tween', duration: 0.14, ease: 'linear' }}
      />
    </>
  )
}
