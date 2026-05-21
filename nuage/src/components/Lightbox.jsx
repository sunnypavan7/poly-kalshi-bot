import { useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function Lightbox({ images, index, onClose, onPrev, onNext }) {
  const img = images[index]

  const handleKey = useCallback(e => {
    if (e.key === 'Escape') onClose()
    if (e.key === 'ArrowLeft') onPrev()
    if (e.key === 'ArrowRight') onNext()
  }, [onClose, onPrev, onNext])

  useEffect(() => {
    document.addEventListener('keydown', handleKey)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', handleKey)
      document.body.style.overflow = ''
    }
  }, [handleKey])

  return (
    <motion.div
      className="lightbox"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      onClick={onClose}
    >
      <AnimatePresence mode="wait">
        <motion.img
          key={index}
          className="lightbox-img"
          src={img.src}
          alt={img.alt}
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 1.02 }}
          transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
          onClick={e => e.stopPropagation()}
        />
      </AnimatePresence>

      <button className="lightbox-close" onClick={onClose} aria-label="Close">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M18 6 6 18M6 6l12 12"/>
        </svg>
      </button>

      {images.length > 1 && (
        <>
          <button className="lightbox-prev" onClick={e => { e.stopPropagation(); onPrev() }} aria-label="Previous">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 5l-7 7 7 7"/></svg>
          </button>
          <button className="lightbox-next" onClick={e => { e.stopPropagation(); onNext() }} aria-label="Next">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          </button>
        </>
      )}

      <div className="lightbox-label">{img.label}</div>
      <div className="lightbox-counter">{index + 1} / {images.length}</div>
    </motion.div>
  )
}
