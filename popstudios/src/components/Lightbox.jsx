import { useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function Lightbox({ images, currentIndex, onClose, onPrev, onNext }) {
  const closeRef = useRef(null)

  const handleKey = useCallback((e) => {
    if (e.key === 'Escape') onClose()
    if (e.key === 'ArrowLeft') onPrev()
    if (e.key === 'ArrowRight') onNext()
  }, [onClose, onPrev, onNext])

  useEffect(() => {
    document.addEventListener('keydown', handleKey)
    document.body.style.overflow = 'hidden'
    closeRef.current?.focus()
    return () => {
      document.removeEventListener('keydown', handleKey)
      document.body.style.overflow = ''
    }
  }, [handleKey])

  const image = images[currentIndex]

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1, transition: { duration: 0.3 } }}
        exit={{ opacity: 0, transition: { duration: 0.25 } }}
        className="fixed inset-0 z-[1000] bg-black/95 flex items-center justify-center"
        role="dialog"
        aria-modal="true"
        aria-label="Image lightbox"
        onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
      >
        {/* Close */}
        <button
          ref={closeRef}
          onClick={onClose}
          className="absolute top-6 right-8 text-[var(--color-warm-white)] hover:text-[var(--color-accent)] transition-colors z-10"
          aria-label="Close lightbox"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        {/* Counter */}
        <div className="absolute top-6 left-1/2 -translate-x-1/2 text-xs tracking-[0.12em] text-[var(--color-muted)]">
          {currentIndex + 1} / {images.length}
        </div>

        {/* Prev */}
        <button
          onClick={onPrev}
          disabled={currentIndex === 0}
          className="absolute left-4 md:left-8 text-[var(--color-warm-white)] hover:text-[var(--color-accent)] transition-colors disabled:opacity-20 disabled:cursor-not-allowed z-10 p-2"
          aria-label="Previous image"
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M15 18l-6-6 6-6" />
          </svg>
        </button>

        {/* Image */}
        <motion.div
          key={currentIndex}
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1, transition: { duration: 0.35, ease: [0.16, 1, 0.3, 1] } }}
          exit={{ opacity: 0, scale: 0.97, transition: { duration: 0.2 } }}
          className="max-w-[90vw] max-h-[85vh] flex flex-col items-center gap-4"
        >
          <img
            src={image.src}
            alt={image.alt}
            className="max-w-full max-h-[80vh] object-contain"
            loading="eager"
          />
          {image.alt && (
            <p className="text-xs text-[var(--color-muted)] tracking-[0.06em] text-center max-w-md">
              {image.alt}
            </p>
          )}
        </motion.div>

        {/* Next */}
        <button
          onClick={onNext}
          disabled={currentIndex === images.length - 1}
          className="absolute right-4 md:right-8 text-[var(--color-warm-white)] hover:text-[var(--color-accent)] transition-colors disabled:opacity-20 disabled:cursor-not-allowed z-10 p-2"
          aria-label="Next image"
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M9 18l6-6-6-6" />
          </svg>
        </button>
      </motion.div>
    </AnimatePresence>
  )
}
