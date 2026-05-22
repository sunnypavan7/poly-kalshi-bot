import { useState, useEffect, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { projects } from '../data/portfolio'
import { TESTIMONIALS } from '../data/services'

const SLIDES = [
  {
    service: 'Weddings',
    tagline: 'Every ritual. Every emotion. Every breath.',
    image: 'https://images.unsplash.com/photo-1519741497674-4b50e55c7a8b?auto=format&fit=crop&w=1920&q=80',
    alt: 'Indian wedding couple in golden light',
  },
  {
    service: 'Portraits',
    tagline: 'Light finds the real you.',
    image: 'https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?auto=format&fit=crop&w=1920&q=80',
    alt: 'Portrait in natural warm light',
  },
  {
    service: 'Commercial',
    tagline: 'Brands built on honest images.',
    image: 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?auto=format&fit=crop&w=1920&q=80',
    alt: 'Commercial fashion photography',
  },
  {
    service: 'Events',
    tagline: 'The moments between moments.',
    image: 'https://images.unsplash.com/photo-1540575467063-178a50c2df87?auto=format&fit=crop&w=1920&q=80',
    alt: 'Event photography',
  },
]

const bgVariants = {
  enter: (dir) => ({ scale: 1.08, opacity: 0, filter: 'blur(8px)' }),
  center: { scale: 1, opacity: 1, filter: 'blur(0px)', transition: { duration: 0.9, ease: [0.16, 1, 0.3, 1] } },
  exit: (dir) => ({ scale: 0.97, opacity: 0, filter: 'blur(6px)', transition: { duration: 0.55, ease: [0.76, 0, 0.24, 1] } }),
}

const textVariants = {
  enter: (dir) => ({ opacity: 0, y: dir * 28 }),
  center: { opacity: 1, y: 0, transition: { duration: 0.55, delay: 0.2, ease: [0.16, 1, 0.3, 1] } },
  exit: (dir) => ({ opacity: 0, y: -dir * 18, transition: { duration: 0.3, ease: [0.76, 0, 0.24, 1] } }),
}

function HeroSection() {
  const [current, setCurrent] = useState(0)
  const [direction, setDirection] = useState(1)
  const reduced = useReducedMotion()
  const timerRef = useRef(null)

  const advance = useCallback(() => {
    setDirection(1)
    setCurrent(c => (c + 1) % SLIDES.length)
  }, [])

  const startTimer = useCallback(() => {
    clearInterval(timerRef.current)
    if (!reduced) timerRef.current = setInterval(advance, 5000)
  }, [advance, reduced])

  const next = useCallback(() => { setDirection(1); setCurrent(c => (c + 1) % SLIDES.length); startTimer() }, [startTimer])
  const prev = useCallback(() => { setDirection(-1); setCurrent(c => (c - 1 + SLIDES.length) % SLIDES.length); startTimer() }, [startTimer])

  useEffect(() => { startTimer(); return () => clearInterval(timerRef.current) }, [startTimer])

  const handlePanEnd = useCallback((_, info) => {
    const { offset, velocity } = info
    if (Math.abs(offset.x) > Math.abs(offset.y) && (Math.abs(offset.x) > 60 || Math.abs(velocity.x) > 300)) {
      if (offset.x < 0) next(); else prev()
    }
  }, [next, prev])

  return (
    <motion.section
      className="relative h-screen min-h-[600px] overflow-hidden flex items-end"
      style={{ paddingBottom: 'clamp(60px, 8vh, 120px)' }}
      onPanEnd={handlePanEnd}
      aria-roledescription="carousel"
      aria-label="Services showcase"
    >
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
          <div
            className="absolute inset-0"
            style={{ background: 'linear-gradient(165deg, rgba(8,8,8,0.55) 0%, rgba(8,8,8,0.05) 50%, rgba(8,8,8,0.72) 100%)' }}
            aria-hidden="true"
          />
        </motion.div>
      </AnimatePresence>

      {/* Slide counter — top right */}
      <div className="absolute top-24 right-6 md:right-12 z-20 pointer-events-none" aria-hidden="true">
        <AnimatePresence mode="wait">
          <motion.span
            key={current}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
            style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '0.72rem', letterSpacing: '0.18em', color: 'var(--color-accent)' }}
          >
            {String(current + 1).padStart(2, '0')}&nbsp;/&nbsp;{String(SLIDES.length).padStart(2, '0')}
          </motion.span>
        </AnimatePresence>
      </div>

      {/* Text block */}
      <div className="relative z-10 w-full max-w-[1440px] mx-auto px-6 md:px-12">
        <AnimatePresence mode="wait" custom={direction}>
          <motion.div key={current} custom={direction} variants={reduced ? undefined : textVariants} initial="enter" animate="center" exit="exit">
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.22em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 16 }}>
              PopStudios — Mumbai
            </p>
            <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, lineHeight: 0.9, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 20, fontSize: 'clamp(3.5rem, 10vw, 8rem)' }}>
              {SLIDES[current].service}
            </h1>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'rgba(245,245,245,0.65)', marginBottom: 36, fontSize: 'clamp(0.95rem, 1.5vw, 1.15rem)' }}>
              {SLIDES[current].tagline}
            </p>
            <Link
              to="/contact"
              style={{ display: 'inline-flex', alignItems: 'center', gap: 12, fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.8rem', letterSpacing: '0.12em', textTransform: 'uppercase', background: 'var(--color-accent)', color: 'var(--color-black)', padding: '14px 28px', transition: 'background 0.2s' }}
              onMouseEnter={e => e.currentTarget.style.background = '#e04400'}
              onMouseLeave={e => e.currentTarget.style.background = 'var(--color-accent)'}
            >
              Book a Shoot
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </motion.div>
        </AnimatePresence>

        {/* Dots */}
        <div className="flex items-center gap-2 mt-10" role="tablist" aria-label="Slide navigation">
          {SLIDES.map((slide, i) => (
            <button
              key={i}
              role="tab"
              aria-selected={i === current}
              aria-label={slide.service}
              onClick={() => { const dir = i > current ? 1 : -1; setDirection(dir); setCurrent(i); startTimer() }}
              className="focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--color-accent)] rounded-sm p-1"
            >
              <motion.div
                animate={{ width: i === current ? 24 : 6, opacity: i === current ? 1 : 0.35, background: i === current ? 'var(--color-accent)' : 'var(--color-white)' }}
                transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                style={{ height: 2, borderRadius: 2 }}
              />
            </button>
          ))}
        </div>
      </div>

      {/* Arrows */}
      {[
        { fn: prev, label: 'Previous', path: 'M19 12H5M12 5l-7 7 7 7', side: 'left-4 md:left-8' },
        { fn: next, label: 'Next',     path: 'M5 12h14M12 5l7 7-7 7', side: 'right-4 md:right-8' },
      ].map(({ fn, label, path, side }) => (
        <button
          key={label}
          onClick={fn}
          className={`absolute ${side} top-1/2 -translate-y-1/2 z-20 w-10 h-10 flex items-center justify-center border border-white/15 text-white/40 hover:border-white/40 hover:text-white transition-all duration-200 focus-visible:outline-none focus-visible:border-[var(--color-accent)]`}
          aria-label={label}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d={path} />
          </svg>
        </button>
      ))}
    </motion.section>
  )
}

function MarqueeStrip() {
  const items = ['Weddings', 'Portraits', 'Commercial', 'Events', 'Destination', 'Mumbai', 'Pan India', 'Weddings', 'Portraits', 'Commercial', 'Events', 'Destination', 'Mumbai', 'Pan India']
  return (
    <div
      className="overflow-hidden py-4"
      style={{ borderTop: '1px solid var(--color-border)', borderBottom: '1px solid var(--color-border)', background: 'var(--color-surface)' }}
      aria-hidden="true"
    >
      <div className="flex whitespace-nowrap marquee-track">
        {items.concat(items).map((item, i) => (
          <span
            key={i}
            style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: i % 3 === 0 ? 'var(--color-accent)' : 'var(--color-muted-dark)', marginRight: 48 }}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  )
}

function FeaturedWork() {
  const featured = projects.slice(0, 4)
  return (
    <section className="max-w-[1440px] mx-auto px-6 md:px-12 py-20 md:py-32">
      <ScrollReveal>
        <div className="flex items-end justify-between mb-12 md:mb-16">
          <div>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 10 }}>
              Selected Work
            </p>
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2rem, 4vw, 3rem)', lineHeight: 1, letterSpacing: '-0.03em', color: 'var(--color-white)' }}>
              Recent Projects
            </h2>
          </div>
          <Link
            to="/portfolio"
            className="hidden md:flex items-center gap-2 hover:text-[var(--color-white)] transition-colors"
            style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.78rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--color-muted)' }}
          >
            View All
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </ScrollReveal>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-1">
        {featured.map((project, i) => (
          <ScrollReveal key={project.id} delay={i * 0.07}>
            <Link
              to={`/portfolio/${project.id}`}
              className="group block relative overflow-hidden"
              style={{ aspectRatio: project.aspect === 'portrait' ? '4/5' : project.aspect === 'square' ? '1/1' : '3/2', background: 'var(--color-surface)' }}
            >
              <img
                src={project.cover}
                alt={project.coverAlt}
                className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-105"
                loading={i < 2 ? 'eager' : 'lazy'}
              />
              <div
                className="absolute inset-0 flex flex-col justify-end p-6 md:p-8 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                style={{ background: 'linear-gradient(to top, rgba(8,8,8,0.88) 0%, transparent 55%)' }}
              >
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 4 }}>
                  {project.category}
                </p>
                <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.4rem', color: 'var(--color-white)', lineHeight: 1.1 }}>
                  {project.title}
                </h3>
                <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: 'rgba(245,245,245,0.6)', marginTop: 4 }}>
                  {project.location}, {project.year}
                </p>
              </div>
              {/* Always-visible on mobile */}
              <div className="md:hidden absolute bottom-4 left-4">
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.62rem', letterSpacing: '0.15em', textTransform: 'uppercase', color: 'var(--color-accent)' }}>
                  {project.category}
                </p>
                <p style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.1rem', color: 'var(--color-white)' }}>{project.title}</p>
              </div>
            </Link>
          </ScrollReveal>
        ))}
      </div>

      <div className="mt-6 md:hidden text-center">
        <Link
          to="/portfolio"
          className="inline-flex items-center gap-2 hover:text-[var(--color-white)] transition-colors"
          style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.78rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--color-muted)' }}
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
    <section style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(60px, 8vw, 120px)', paddingBottom: 'clamp(60px, 8vw, 120px)' }}>
      <div className="max-w-[1440px] mx-auto px-6 md:px-12">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-20 items-center">
          <ScrollReveal>
            <div className="relative overflow-hidden" style={{ aspectRatio: '4/5', background: 'var(--color-surface)' }}>
              <img
                src="https://images.unsplash.com/photo-1554151228-14d9def656e4?auto=format&fit=crop&w=900&q=80"
                alt="PopStudios photographer at work"
                className="w-full h-full object-cover"
                loading="lazy"
              />
              {/* Orange tag */}
              <div className="absolute top-6 left-6" style={{ background: 'var(--color-accent)', padding: '6px 14px' }}>
                <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 700, fontSize: '0.65rem', letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--color-black)' }}>
                  Mumbai Based
                </span>
              </div>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={0.15} direction="left">
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 20 }}>
              The Studio
            </p>
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2rem, 4vw, 3.2rem)', lineHeight: 1, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 24 }}>
              We don't capture<br />moments — we <em style={{ fontStyle: 'italic' }}>make</em> them.
            </h2>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'var(--color-muted)', lineHeight: 1.8, marginBottom: 16, fontSize: '0.93rem' }}>
              PopStudios is a Mumbai-based photography studio working across weddings, portraits, commercial campaigns, and events. Founded on the belief that every frame should tell a story that needs no caption.
            </p>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'var(--color-muted)', lineHeight: 1.8, marginBottom: 36, fontSize: '0.93rem' }}>
              We've shot from Udaipur to Bangalore, from Bandra rooftops to Rajasthan deserts. The camera changes. The obsession with light doesn't.
            </p>
            <Link
              to="/about"
              className="inline-flex items-center gap-2 hover:text-[var(--color-accent)] transition-colors group"
              style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.8rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-white)' }}
            >
              Our Story
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="group-hover:translate-x-1 transition-transform">
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
  const services = [
    { icon: '01', label: 'Weddings', desc: 'Documentary & fine-art coverage of every ritual', price: 'From ₹75,000' },
    { icon: '02', label: 'Portraits', desc: 'Personal, professional & creative sessions', price: 'From ₹18,000' },
    { icon: '03', label: 'Commercial', desc: 'Campaigns, lookbooks & brand content', price: 'From ₹30,000/day' },
    { icon: '04', label: 'Events', desc: 'Corporate, cultural & private events', price: 'From ₹25,000' },
  ]
  return (
    <section style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(60px, 8vw, 120px)', paddingBottom: 'clamp(60px, 8vw, 120px)' }}>
      <div className="max-w-[1440px] mx-auto px-6 md:px-12">
        <ScrollReveal>
          <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 12 }}>
            Services
          </p>
          <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2rem, 4vw, 3rem)', lineHeight: 1, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 56 }}>
            What we shoot
          </h2>
        </ScrollReveal>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-px" style={{ background: 'var(--color-border)' }}>
          {services.map((s, i) => (
            <ScrollReveal key={s.label} delay={i * 0.08}>
              <div
                className="group transition-colors duration-200"
                style={{ background: 'var(--color-black)', padding: '32px 28px' }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--color-surface)'}
                onMouseLeave={e => e.currentTarget.style.background = 'var(--color-black)'}
              >
                <p style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '0.65rem', letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 20 }}>
                  {s.icon}
                </p>
                <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.3rem', color: 'var(--color-white)', marginBottom: 10, lineHeight: 1.1 }}>
                  {s.label}
                </h3>
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.82rem', color: 'var(--color-muted)', lineHeight: 1.7, marginBottom: 20 }}>
                  {s.desc}
                </p>
                <p style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '0.9rem', color: 'var(--color-accent)' }}>
                  {s.price}
                </p>
              </div>
            </ScrollReveal>
          ))}
        </div>

        <ScrollReveal delay={0.2}>
          <div className="mt-10 flex justify-end">
            <Link
              to="/services"
              className="inline-flex items-center gap-2 hover:text-[var(--color-white)] transition-colors group"
              style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.78rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--color-muted)' }}
            >
              Full Pricing
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="group-hover:translate-x-1 transition-transform">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

function Testimonials() {
  return (
    <section style={{ background: 'var(--color-surface)', paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)', borderTop: '1px solid var(--color-border)' }}>
      <div className="max-w-[1440px] mx-auto px-6 md:px-12">
        <ScrollReveal>
          <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 48 }}>
            What Clients Say
          </p>
        </ScrollReveal>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
          {TESTIMONIALS.map((t, i) => (
            <ScrollReveal key={t.name} delay={i * 0.1}>
              <div>
                <p style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: '1.05rem', color: 'var(--color-white)', lineHeight: 1.65, marginBottom: 20, fontStyle: 'italic' }}>
                  "{t.quote}"
                </p>
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.8rem', color: 'var(--color-accent)', letterSpacing: '0.06em' }}>
                  {t.name}
                </p>
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.75rem', color: 'var(--color-muted)', marginTop: 2 }}>
                  {t.detail}
                </p>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  )
}

function CtaBanner() {
  return (
    <section style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(80px, 10vw, 140px)', paddingBottom: 'clamp(80px, 10vw, 140px)' }}>
      <div className="max-w-[1440px] mx-auto px-6 md:px-12">
        <ScrollReveal>
          <div className="max-w-2xl">
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2.5rem, 6vw, 5rem)', lineHeight: 0.95, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 28 }}>
              Let's make<br />
              something<br />
              <span style={{ color: 'var(--color-accent)' }}>unforgettable.</span>
            </h2>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'var(--color-muted)', lineHeight: 1.7, marginBottom: 40, fontSize: '0.95rem' }}>
              Based in Mumbai. Available across India. We respond within 24 hours.
            </p>
            <Link
              to="/contact"
              style={{ display: 'inline-flex', alignItems: 'center', gap: 12, fontFamily: 'var(--font-sans)', fontWeight: 700, fontSize: '0.82rem', letterSpacing: '0.12em', textTransform: 'uppercase', background: 'var(--color-accent)', color: 'var(--color-black)', padding: '16px 32px', transition: 'background 0.2s' }}
              onMouseEnter={e => e.currentTarget.style.background = '#e04400'}
              onMouseLeave={e => e.currentTarget.style.background = 'var(--color-accent)'}
            >
              Start a Conversation
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
      <MarqueeStrip />
      <FeaturedWork />
      <AboutTeaser />
      <ServicesTeaser />
      <Testimonials />
      <CtaBanner />
    </main>
  )
}
