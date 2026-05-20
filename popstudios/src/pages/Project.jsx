import { useState } from 'react'
import { useParams, Link, Navigate } from 'react-router-dom'
import { motion, useScroll, useTransform, useReducedMotion } from 'framer-motion'
import { useRef } from 'react'
import { projects } from '../data/portfolio'
import Lightbox from '../components/Lightbox'
import ScrollReveal from '../components/ScrollReveal'

function ParallaxImage({ src, alt, index }) {
  const ref = useRef(null)
  const reduced = useReducedMotion()
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] })
  const y = useTransform(scrollYProgress, [0, 1], reduced ? ['0%', '0%'] : ['-8%', '8%'])

  return (
    <div ref={ref} className={`overflow-hidden ${index % 2 === 0 ? 'md:col-span-2' : 'md:col-span-1'}`}>
      <motion.img
        src={src}
        alt={alt}
        style={{ y }}
        className="w-full h-full object-cover"
        loading="lazy"
      />
    </div>
  )
}

export default function Project() {
  const { id } = useParams()
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const [lightboxIndex, setLightboxIndex] = useState(0)

  const project = projects.find(p => p.id === id)
  if (!project) return <Navigate to="/portfolio" replace />

  const openLightbox = (i) => { setLightboxIndex(i); setLightboxOpen(true) }

  return (
    <main>
      {/* Hero */}
      <section className="relative h-[70vh] md:h-screen overflow-hidden">
        <img
          src={project.cover}
          alt={project.coverAlt}
          className="absolute inset-0 w-full h-full object-cover"
          loading="eager"
        />
        <div
          className="absolute inset-0"
          style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0.65) 100%)' }}
          aria-hidden="true"
        />
        <div className="relative z-10 h-full flex flex-col justify-end pb-16 md:pb-24 px-8 md:px-16 max-w-[1440px] mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          >
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-3">
              {project.category}
            </p>
            <h1 className="font-display font-light leading-none tracking-[-0.03em] text-[var(--color-warm-white)] mb-4"
                style={{ fontSize: 'clamp(3rem, 8vw, 7rem)' }}>
              {project.title}
            </h1>
            <p className="font-sans font-light text-[var(--color-warm-white)]/70 text-lg italic">
              {project.excerpt}
            </p>
          </motion.div>
        </div>
      </section>

      {/* Meta */}
      <section className="border-b border-[var(--color-border)]">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16 py-10 md:py-12">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { label: 'Client', value: project.client },
              { label: 'Category', value: project.category },
              { label: 'Location', value: project.location },
              { label: 'Year', value: project.year },
            ].map(({ label, value }) => (
              <div key={label}>
                <p className="font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted-dark)] mb-2">{label}</p>
                <p className="font-sans text-sm text-[var(--color-warm-white)]">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Story */}
      <section className="max-w-[1440px] mx-auto px-8 md:px-16 py-20 md:py-28">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12">
          <ScrollReveal className="md:col-span-7 md:col-start-3">
            <p className="font-sans font-light text-lg leading-relaxed text-[var(--color-muted)]">
              {project.description}
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Image gallery */}
      <section className="max-w-[1440px] mx-auto px-8 md:px-16 pb-20 md:pb-32">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-1">
          {project.images.map((image, i) => (
            <ScrollReveal key={i} delay={i * 0.1}>
              <button
                onClick={() => openLightbox(i)}
                className={`group block relative overflow-hidden bg-[var(--color-surface)] w-full cursor-pointer ${
                  i === 0 ? 'md:col-span-2 aspect-[16/10]' : 'aspect-[4/5]'
                }`}
                aria-label={`View full image: ${image.alt}`}
              >
                <img
                  src={image.src}
                  alt={image.alt}
                  className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-103"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/15 transition-colors duration-300 flex items-center justify-center">
                  <svg
                    className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 text-white"
                    width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1"
                  >
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
                  </svg>
                </div>
              </button>
            </ScrollReveal>
          ))}
        </div>
      </section>

      {/* Navigation to next/prev projects */}
      <section className="border-t border-[var(--color-border)]">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16 py-12">
          <div className="flex items-center justify-between">
            <Link
              to="/portfolio"
              className="font-sans text-xs tracking-[0.14em] uppercase text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors duration-200 flex items-center gap-2"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              All Projects
            </Link>
            <Link
              to="/contact"
              className="font-sans text-xs tracking-[0.14em] uppercase text-[var(--color-muted)] hover:text-[var(--color-accent)] transition-colors duration-200"
            >
              Inquire about a project
            </Link>
          </div>
        </div>
      </section>

      {lightboxOpen && (
        <Lightbox
          images={project.images}
          currentIndex={lightboxIndex}
          onClose={() => setLightboxOpen(false)}
          onPrev={() => setLightboxIndex(i => Math.max(0, i - 1))}
          onNext={() => setLightboxIndex(i => Math.min(project.images.length - 1, i + 1))}
        />
      )}
    </main>
  )
}
