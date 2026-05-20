import { useState, useEffect, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { projects } from '../data/portfolio'

const SLIDES = [
  {
    service: 'Weddings',
    tagline: 'Every gesture, every glance',
    image: 'https://images.unsplash.com/photo-1519741497674-611481863552?auto=format&fit=crop&w=1920&q=80',
    alt: 'Wedding couple in soft golden light',
  },
  {
    service: 'Maternity',
    tagline: 'New life, timeless light',
    image: 'https://images.unsplash.com/photo-1520342868574-5fa3804e551c?auto=format&fit=crop&w=1920&q=80',
    alt: 'Maternity portrait in natural light',
  },
  {
    service: 'Editorial',
    tagline: 'Concept made visible',
    image: 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?auto=format&fit=crop&w=1920&q=80',
    alt: 'Fashion editorial photography',
  },
  {
    service: 'Model Shoots',
    tagline: 'Form, light, character',
    image: 'https://images.unsplash.com/photo-1469334031218-e382a71b716b?auto=format&fit=crop&w=1920&q=80',
    alt: 'Model portrait studio shoot',
  },
  {
    service: 'Food Shoot',
    tagline: 'Taste made visible',
    image: 'https://images.unsplash.com/photo-1476224203421-9ac39bcb3327?auto=format&fit=crop&w=1920&q=80',
    alt: 'Artistic food photography',
  },
]

// Lens barrel ring — tick marks rotate with each slide
function ApertureRing({ angle, size = 100 }) {
  const cx = size / 2
  const cy = size / 2
  const outerR = cx - 2
  const numTicks = 48

  const ticks = Array.from({ length: numTicks }, (_, i) => {
    const deg = (i * 360) / numTicks
    const rad = (deg * Math.PI) / 180
    const isMajor = i % 8 === 0
    const isMid = i % 4 === 0 && !isMajor
    const tickLen = isMajor ? 14 : isMid ? 9 : 5
    const innerR = outerR - tickLen
    return {
      x1: cx + innerR * Math.sin(rad),
      y1: cy - innerR * Math.cos(rad),
      x2: cx + outerR * Math.sin(rad),
      y2: cy - outerR * Math.cos(rad),
      isMajor,
      isMid,
    }
  })

  return (
    <motion.svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      animate={{ rotate: angle }}
      transition={{ duration: 0.9, ease: [0.25, 0.1, 0.0, 1.0] }}
      aria-hidden="true"
    >
      {ticks.map((t, i) => (
        <line
          key={i}
          x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
          stroke={
            t.isMajor
              ? 'rgba(197,160,104,0.85)'
              : t.isMid
              ? 'rgba(197,160,104,0.45)'
              : 'rgba(197,160,104,0.2)'
          }
          strokeWidth={t.isMajor ? 1.5 : t.isMid ? 1 : 0.6}
        />
      ))}
      <circle cx={cx} cy={cy} r={outerR - 20} fill="none" stroke="rgba(197,160,104,0.2)" strokeWidth="1" />
      <circle cx={cx} cy={cy} r={outerR - 22} fill="none" stroke="rgba(197,160,104,0.08)" strokeWidth="0.5" />
    </motion.svg>
  )
}

// Background image variants — rotation + scale + blur = lens focus throw
const bgVariants = {
  enter: (dir) => ({
    rotate: dir * 9,
    scale: 1.12,
    filter: 'blur(12px)',
    opacity: 0,
  }),
  center: {
    rotate: 0,
    scale: 1,
    filter: 'blur(0px)',
    opacity: 1,
    transition: { duration: 0.95, ease: [0.25, 0.1, 0.0, 1.0] },
  },
  exit: (dir) => ({
    rotate: -dir * 6,
    scale: 0.95,
    filter: 'blur(9px)',
    opacity: 0,
    transition: { duration: 0.5, ease: [0.76, 0, 0.24, 1] },
  }),
}

// Service text slides up/down with the lens
const textVariants = {
  enter: (dir) => ({
    opacity: 0,
    y: dir * 32,
    filter: 'blur(6px)',
  }),
  center: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.55, delay: 0.18, ease: [0.16, 1, 0.3, 1] },
  },
  exit: (dir) => ({
    opacity: 0,
    y: -dir * 18,
    filter: 'blur(4px)',
    transition: { duration: 0.3, ease: [0.76, 0, 0.24, 1] },
  }),
}

const featured = projects.slice(0, 4)

function HeroSection() {
  const [current, setCurrent] = useState(0)
  const [direction, setDirection] = useState(1)
  const [lensAngle, setLensAngle] = useState(0)
  const reduced = useReducedMotion()
  const intervalRef = useRef(null)

  // Inline advance so startTimer can reference it without a stale closure
  const advance = useCallback(() => {
    setDirection(1)
    setCurrent(c => (c + 1) % SLIDES.length)
    setLensAngle(a => a + 72)
  }, [])

  const startTimer = useCallback(() => {
    clearInterval(intervalRef.current)
    if (!reduced) intervalRef.current = setInterval(advance, 5500)
  }, [advance, reduced])

  const next = useCallback(() => {
    setDirection(1)
    setCurrent(c => (c + 1) % SLIDES.length)
    setLensAngle(a => a + 72)
    startTimer()
  }, [startTimer])

  const prev = useCallback(() => {
    setDirection(-1)
    setCurrent(c => (c - 1 + SLIDES.length) % SLIDES.length)
    setLensAngle(a => a - 72)
    startTimer()
  }, [startTimer])

  const goTo = useCallback((i) => {
    setCurrent(c => {
      const dir = i > c ? 1 : -1
      setDirection(dir)
      setLensAngle(a => a + dir * 72)
      return i
    })
    startTimer()
  }, [startTimer])

  // start auto-advance on mount
  useEffect(() => {
    startTimer()
    return () => clearInterval(intervalRef.current)
  }, [startTimer])

  // keyboard navigation
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight') next()
      else if (e.key === 'ArrowLeft') prev()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [next, prev])

  // horizontal swipe detection
  const handlePanEnd = useCallback((_, info) => {
    const { offset, velocity } = info
    if (Math.abs(offset.x) > Math.abs(offset.y) && (Math.abs(offset.x) > 60 || Math.abs(velocity.x) > 300)) {
      if (offset.x < 0) next()
      else prev()
    }
  }, [next, prev])

  return (
    <motion.section
      className="relative h-screen min-h-[600px] overflow-hidden flex items-end pb-16 md:pb-24"
      onPanEnd={handlePanEnd}
      aria-roledescription="carousel"
      aria-label="Services showcase"
    >
      {/* Rotating background slides */}
      <AnimatePresence custom={direction}>
        <motion.div
          key={current}
          custom={direction}
          variants={reduced ? undefined : bgVariants}
          initial="enter"
          animate="center"
          exit="exit"
          className="absolute inset-0"
        >
          <img
            src={SLIDES[current].image}
            alt={SLIDES[current].alt}
            className="absolute inset-0 w-full h-full object-cover"
          />
          {/* Cinematic scrim — heavier at bottom-left for legibility */}
          <div
            className="absolute inset-0"
            style={{
              background:
                'linear-gradient(155deg, rgba(10,10,10,0.62) 0%, rgba(10,10,10,0.08) 45%, rgba(10,10,10,0.55) 100%)',
            }}
            aria-hidden="true"
          />
        </motion.div>
      </AnimatePresence>

      {/* ── Aperture ring + counter ── */}
      <div
        className="absolute bottom-14 right-8 md:right-16 z-20 pointer-events-none"
        aria-hidden="true"
      >
        <div className="relative" style={{ width: 100, height: 100 }}>
          <ApertureRing angle={lensAngle} size={100} />
          {/* Slide counter inside the ring */}
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-0.5">
            <AnimatePresence mode="wait">
              <motion.span
                key={current}
                initial={{ opacity: 0, scale: 0.6 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.4 }}
                transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
                className="font-sans font-medium leading-none"
                style={{ fontSize: '0.9rem', color: 'var(--color-accent)', letterSpacing: '0.08em' }}
              >
                {String(current + 1).padStart(2, '0')}
              </motion.span>
            </AnimatePresence>
            <span
              className="font-sans"
              style={{ fontSize: '0.55rem', color: 'rgba(197,160,104,0.5)', letterSpacing: '0.15em' }}
            >
              / {String(SLIDES.length).padStart(2, '0')}
            </span>
          </div>
        </div>
      </div>

      {/* ── Service text & CTA ── */}
      <div className="relative z-10 w-full max-w-[1440px] mx-auto px-8 md:px-16">
        <AnimatePresence mode="wait" custom={direction}>
          <motion.div
            key={current}
            custom={direction}
            variants={reduced ? undefined : textVariants}
            initial="enter"
            animate="center"
            exit="exit"
          >
            <p className="font-sans text-xs tracking-[0.22em] uppercase text-[var(--color-accent)] mb-5">
              {String(current + 1).padStart(2, '0')}&nbsp;&nbsp;—&nbsp;&nbsp;PopStudios
            </p>
            <h1
              className="font-display font-light leading-[0.9] tracking-[-0.03em] text-[var(--color-warm-white)] mb-5"
              style={{ fontSize: 'clamp(3.2rem, 9vw, 7.5rem)' }}
            >
              {SLIDES[current].service}
            </h1>
            <p
              className="font-sans font-light tracking-wide mb-10"
              style={{ color: 'rgba(247,245,242,0.62)', fontSize: 'clamp(0.95rem, 1.5vw, 1.2rem)' }}
            >
              {SLIDES[current].tagline}
            </p>
            <Link
              to="/contact"
              className="inline-flex items-center gap-3 font-sans font-medium text-sm tracking-[0.1em] uppercase text-[var(--color-warm-white)] border border-[var(--color-warm-white)]/40 px-7 py-3.5 hover:bg-[var(--color-warm-white)] hover:text-[var(--color-black)] transition-all duration-300"
            >
              Book This Shoot
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </motion.div>
        </AnimatePresence>

        {/* Dot / progress indicators */}
        <div
          className="absolute bottom-0 left-8 md:left-16 flex items-center gap-2.5"
          role="tablist"
          aria-label="Slide navigation"
        >
          {SLIDES.map((slide, i) => (
            <button
              key={i}
              role="tab"
              aria-selected={i === current}
              aria-label={`${slide.service}`}
              onClick={() => goTo(i)}
              className="group p-1.5 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--color-accent)] rounded-sm"
            >
              <motion.div
                className="rounded-full"
                style={{ height: 2 }}
                animate={{
                  width: i === current ? 28 : 8,
                  opacity: i === current ? 1 : 0.3,
                  backgroundColor:
                    i === current ? 'var(--color-accent)' : 'var(--color-warm-white)',
                }}
                transition={{ duration: 0.4, ease: [0.25, 0.1, 0.0, 1.0] }}
              />
            </button>
          ))}
        </div>

        {/* Swipe hint on mobile — fades after first interaction */}
        <motion.p
          className="absolute bottom-1 right-28 md:right-36 font-sans text-[10px] tracking-[0.18em] uppercase pointer-events-none"
          style={{ color: 'rgba(247,245,242,0.3)' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 3 }}
          aria-hidden="true"
        >
          Swipe
        </motion.p>
      </div>

      {/* Prev / Next arrow buttons */}
      <button
        onClick={prev}
        className="absolute left-4 md:left-8 top-1/2 -translate-y-1/2 z-20 w-10 h-10 flex items-center justify-center border border-[var(--color-warm-white)]/15 text-[var(--color-warm-white)]/40 hover:border-[var(--color-warm-white)]/40 hover:text-[var(--color-warm-white)] transition-all duration-200 focus-visible:outline-none focus-visible:border-[var(--color-accent)]"
        aria-label="Previous service"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M19 12H5M12 5l-7 7 7 7" />
        </svg>
      </button>
      <button
        onClick={next}
        className="absolute right-4 md:right-8 top-1/2 -translate-y-1/2 z-20 w-10 h-10 flex items-center justify-center border border-[var(--color-warm-white)]/15 text-[var(--color-warm-white)]/40 hover:border-[var(--color-warm-white)]/40 hover:text-[var(--color-warm-white)] transition-all duration-200 focus-visible:outline-none focus-visible:border-[var(--color-accent)]"
        aria-label="Next service"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M5 12h14M12 5l7 7-7 7" />
        </svg>
      </button>
    </motion.section>
  )
}

function FeaturedWork() {
  return (
    <section className="max-w-[1440px] mx-auto px-8 md:px-16 py-24 md:py-36">
      <ScrollReveal>
        <div className="flex items-end justify-between mb-14 md:mb-20">
          <div>
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-3">
              Selected Work
            </p>
            <h2 className="font-display font-light text-4xl md:text-5xl leading-tight tracking-[-0.02em] text-[var(--color-warm-white)]">
              Recent Projects
            </h2>
          </div>
          <Link
            to="/portfolio"
            className="hidden md:flex items-center gap-2 font-sans text-sm tracking-[0.08em] uppercase text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors duration-200"
          >
            All Work
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </ScrollReveal>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-1">
        {featured.map((project, i) => (
          <ScrollReveal key={project.id} delay={i * 0.08}>
            <Link
              to={`/portfolio/${project.id}`}
              className="group block relative overflow-hidden bg-[var(--color-surface)]"
              style={{ aspectRatio: project.aspect === 'portrait' ? '4/5' : project.aspect === 'square' ? '1/1' : '3/2' }}
            >
              <img
                src={project.cover}
                alt={project.coverAlt}
                className="w-full h-full object-cover transition-transform duration-[800ms] ease-out group-hover:scale-105"
                loading={i < 2 ? 'eager' : 'lazy'}
              />
              <div
                className="absolute inset-0 flex flex-col justify-end p-6 md:p-8 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.75) 0%, transparent 60%)' }}
              >
                <p className="font-sans text-xs tracking-[0.14em] uppercase text-[var(--color-accent)] mb-1">
                  {project.category}
                </p>
                <h3 className="font-display font-light text-2xl text-[var(--color-warm-white)] leading-tight">
                  {project.title}
                </h3>
                <p className="font-sans text-sm text-[var(--color-warm-white)]/70 mt-1">
                  {project.location}, {project.year}
                </p>
              </div>
              <div className="md:hidden absolute bottom-4 left-4">
                <p className="font-sans text-xs tracking-[0.12em] uppercase text-[var(--color-accent)]">
                  {project.category}
                </p>
                <p className="font-display text-lg text-[var(--color-warm-white)]">{project.title}</p>
              </div>
            </Link>
          </ScrollReveal>
        ))}
      </div>

      <div className="mt-8 md:hidden text-center">
        <Link
          to="/portfolio"
          className="inline-flex items-center gap-2 font-sans text-sm tracking-[0.08em] uppercase text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors"
        >
          View All Work
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </Link>
      </div>
    </section>
  )
}

function AboutTeaser() {
  return (
    <section className="border-t border-[var(--color-border)] py-24 md:py-36">
      <div className="max-w-[1440px] mx-auto px-8 md:px-16">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-20 items-center">
          <ScrollReveal>
            <div className="aspect-[4/5] overflow-hidden bg-[var(--color-surface)]">
              <img
                src="https://images.unsplash.com/photo-1554151228-14d9def656e4?auto=format&fit=crop&w=900&q=80"
                alt="Photographer working in studio — natural window light"
                className="w-full h-full object-cover"
                loading="lazy"
              />
            </div>
          </ScrollReveal>

          <ScrollReveal delay={0.15} direction="left">
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-6">
              The Studio
            </p>
            <h2 className="font-display font-light text-4xl md:text-5xl leading-tight tracking-[-0.02em] text-[var(--color-warm-white)] mb-8">
              Light is the only<br />
              <em>material that matters</em>
            </h2>
            <p className="font-sans font-light text-[var(--color-muted)] leading-relaxed mb-4 max-w-[480px]">
              PopStudios is a London-based photography studio working across editorial, portrait, wedding, and commercial commissions. Every project begins with a conversation about light.
            </p>
            <p className="font-sans font-light text-[var(--color-muted)] leading-relaxed mb-10 max-w-[480px]">
              We believe in unhurried process, long-form collaboration, and photography that exists at the intersection of documentation and art.
            </p>
            <Link
              to="/about"
              className="inline-flex items-center gap-2 font-sans text-sm font-medium tracking-[0.1em] uppercase text-[var(--color-warm-white)] hover:text-[var(--color-accent)] transition-colors duration-200 group"
            >
              Our Story
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="group-hover:translate-x-1 transition-transform duration-200">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </ScrollReveal>
        </div>
      </div>
    </section>
  )
}

function ServicesTeaser() {
  const items = [
    { label: 'Editorial', desc: 'Fashion, commercial, and fine-art commissions' },
    { label: 'Portrait', desc: 'Individuals, creatives, and documentary series' },
    { label: 'Wedding', desc: 'Quiet, documentary, film-influenced coverage' },
  ]

  return (
    <section className="border-t border-[var(--color-border)] py-24 md:py-36">
      <div className="max-w-[1440px] mx-auto px-8 md:px-16">
        <ScrollReveal>
          <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-3">
            Services
          </p>
          <h2 className="font-display font-light text-4xl md:text-5xl leading-tight tracking-[-0.02em] text-[var(--color-warm-white)] mb-16 md:mb-20">
            What we offer
          </h2>
        </ScrollReveal>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-[var(--color-border)]">
          {items.map((item, i) => (
            <ScrollReveal key={item.label} delay={i * 0.1}>
              <div className="bg-[var(--color-black)] p-8 md:p-10 group hover:bg-[var(--color-surface)] transition-colors duration-300">
                <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-muted)] mb-6">
                  0{i + 1}
                </p>
                <h3 className="font-display font-light text-2xl text-[var(--color-warm-white)] mb-4 group-hover:text-[var(--color-accent)] transition-colors duration-300">
                  {item.label}
                </h3>
                <p className="font-sans font-light text-sm text-[var(--color-muted)] leading-relaxed">
                  {item.desc}
                </p>
              </div>
            </ScrollReveal>
          ))}
        </div>

        <ScrollReveal delay={0.2}>
          <div className="mt-12 flex justify-end">
            <Link
              to="/services"
              className="inline-flex items-center gap-2 font-sans text-sm tracking-[0.08em] uppercase text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors duration-200 group"
            >
              Packages &amp; Pricing
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="group-hover:translate-x-1 transition-transform duration-200">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

function CtaBanner() {
  return (
    <section className="border-t border-[var(--color-border)]">
      <div className="max-w-[1440px] mx-auto px-8 md:px-16 py-28 md:py-40">
        <ScrollReveal>
          <div className="max-w-2xl">
            <h2 className="font-display font-light text-4xl md:text-6xl leading-tight tracking-[-0.02em] text-[var(--color-warm-white)] mb-8">
              Let's make something<br />
              <em>worth looking at.</em>
            </h2>
            <Link
              to="/contact"
              className="inline-flex items-center gap-3 font-sans font-medium text-sm tracking-[0.1em] uppercase bg-[var(--color-accent)] text-[var(--color-black)] px-8 py-4 hover:bg-[var(--color-warm-white)] transition-colors duration-300"
            >
              Start a Project
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

export default function Home() {
  return (
    <main>
      <HeroSection />
      <FeaturedWork />
      <AboutTeaser />
      <ServicesTeaser />
      <CtaBanner />
    </main>
  )
}
