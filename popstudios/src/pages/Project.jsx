import { useState, useRef } from 'react'
import { useParams, Link, Navigate } from 'react-router-dom'
import { motion, useScroll, useTransform, useReducedMotion } from 'framer-motion'
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
      <motion.img src={src} alt={alt} style={{ y }} className="w-full h-full object-cover" loading="lazy" />
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
      <section className="relative overflow-hidden" style={{ height: '70vh', minHeight: 480 }}>
        <img
          src={project.cover}
          alt={project.coverAlt}
          className="absolute inset-0 w-full h-full object-cover"
          loading="eager"
        />
        <div className="absolute inset-0" style={{ background: 'linear-gradient(to bottom, rgba(8,8,8,0.15) 0%, rgba(8,8,8,0.72) 100%)' }} aria-hidden="true" />
        <div className="relative z-10 h-full flex flex-col justify-end max-w-[1440px] mx-auto px-6 md:px-12" style={{ paddingBottom: 'clamp(48px, 6vw, 80px)' }}>
          <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.22em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 12 }}>
              {project.category}
            </p>
            <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, lineHeight: 0.9, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 14, fontSize: 'clamp(3rem, 8vw, 7rem)' }}>
              {project.title}
            </h1>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'rgba(245,245,245,0.65)', fontSize: '1.05rem', fontStyle: 'italic' }}>
              {project.excerpt}
            </p>
          </motion.div>
        </div>
      </section>

      {/* Meta */}
      <section style={{ borderBottom: '1px solid var(--color-border)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12 py-8 md:py-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { label: 'Client', value: project.client },
              { label: 'Category', value: project.category },
              { label: 'Location', value: project.location },
              { label: 'Year', value: project.year },
            ].map(({ label, value }) => (
              <div key={label}>
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.62rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 6 }}>{label}</p>
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.88rem', color: 'var(--color-white)' }}>{value}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Story */}
      <section className="max-w-[1440px] mx-auto px-6 md:px-12" style={{ paddingTop: 64, paddingBottom: 64 }}>
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12">
          <ScrollReveal className="md:col-span-7 md:col-start-3">
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '1.05rem', color: 'var(--color-muted)', lineHeight: 1.8 }}>
              {project.description}
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Gallery */}
      <section className="max-w-[1440px] mx-auto px-6 md:px-12" style={{ paddingBottom: 80 }}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-1">
          {project.images.map((image, i) => (
            <ScrollReveal key={i} delay={i * 0.1}>
              <button
                onClick={() => openLightbox(i)}
                className={`group block relative overflow-hidden w-full cursor-pointer ${i === 0 ? 'md:col-span-2' : ''}`}
                style={{ aspectRatio: i === 0 ? '16/10' : '4/5', background: 'var(--color-surface)' }}
                aria-label={`View full image: ${image.alt}`}
              >
                <img
                  src={image.src}
                  alt={image.alt}
                  className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-105"
                  loading="lazy"
                />
                <div className="absolute inset-0 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0)', transition: 'background 0.3s' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(0,0,0,0.18)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'rgba(0,0,0,0)'}
                >
                  <svg className="opacity-0 group-hover:opacity-100 transition-opacity duration-300" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1">
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
                  </svg>
                </div>
              </button>
            </ScrollReveal>
          ))}
        </div>
      </section>

      {/* Nav */}
      <section style={{ borderTop: '1px solid var(--color-border)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12 py-10">
          <div className="flex items-center justify-between">
            <Link
              to="/portfolio"
              className="flex items-center gap-2 transition-colors"
              style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)' }}
              onMouseEnter={e => e.currentTarget.style.color = 'var(--color-white)'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--color-muted)'}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              All Projects
            </Link>
            <Link
              to="/contact"
              className="transition-colors"
              style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)' }}
              onMouseEnter={e => e.currentTarget.style.color = 'var(--color-accent)'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--color-muted)'}
            >
              Book a similar shoot
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
