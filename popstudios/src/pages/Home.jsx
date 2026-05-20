import { useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion, useScroll, useTransform, useReducedMotion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { projects } from '../data/portfolio'

// {/* REPLACE: swap heroVideo src with your own looping .mp4 showreel */}
const HERO_VIDEO_POSTER = 'https://images.unsplash.com/photo-1554048612-b6a482bc67e5?auto=format&fit=crop&w=1920&q=80'

const featured = projects.slice(0, 4)

function HeroSection() {
  const ref = useRef(null)
  const reduced = useReducedMotion()
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] })
  const y = useTransform(scrollYProgress, [0, 1], reduced ? ['0%', '0%'] : ['0%', '25%'])

  return (
    <section ref={ref} className="relative h-screen min-h-[600px] overflow-hidden flex items-end pb-16 md:pb-24">
      {/* Background */}
      <motion.div className="absolute inset-0" style={{ y }}>
        {/* {/* REPLACE: swap this video src with your own showreel .mp4 */}
        <video
          className="absolute inset-0 w-full h-full object-cover"
          autoPlay
          muted
          loop
          playsInline
          poster={HERO_VIDEO_POSTER}
          aria-hidden="true"
        >
          <source src="" type="video/mp4" />
        </video>
        {/* Poster fallback (shown when video not available) */}
        <img
          src={HERO_VIDEO_POSTER}
          alt=""
          className="absolute inset-0 w-full h-full object-cover"
          aria-hidden="true"
        />
        {/* Scrim */}
        <div
          className="absolute inset-0"
          style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.15) 0%, rgba(0,0,0,0.55) 100%)' }}
          aria-hidden="true"
        />
      </motion.div>

      {/* Content */}
      <div className="relative z-10 w-full max-w-[1440px] mx-auto px-8 md:px-16">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
          <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-5">
            London — Worldwide
          </p>
          <h1 className="font-display font-light leading-[0.95] tracking-[-0.03em] text-[var(--color-warm-white)] mb-8"
              style={{ fontSize: 'clamp(3.5rem, 10vw, 8rem)' }}>
            Photography<br />
            <em className="not-italic text-[var(--color-off-white)]">as a fine art</em>
          </h1>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 1.0 }}
          >
            <Link
              to="/portfolio"
              className="inline-flex items-center gap-3 font-sans font-medium text-sm tracking-[0.1em] uppercase text-[var(--color-warm-white)] border border-[var(--color-warm-white)]/40 px-7 py-3.5 hover:bg-[var(--color-warm-white)] hover:text-[var(--color-black)] transition-all duration-300"
            >
              View Work
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          className="absolute bottom-0 right-8 md:right-16 flex flex-col items-center gap-2"
          aria-hidden="true"
        >
          <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--color-warm-white)]/50 rotate-90 origin-center mb-6">
            Scroll
          </span>
          <motion.div
            className="w-px h-12 bg-[var(--color-warm-white)]/30 overflow-hidden"
            aria-hidden="true"
          >
            <motion.div
              className="w-full h-1/2 bg-[var(--color-accent)]"
              animate={{ y: ['-100%', '200%'] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
            />
          </motion.div>
        </motion.div>
      </div>
    </section>
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
              {/* Caption overlay */}
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

              {/* Always-visible minimal label on mobile */}
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
            {/* {/* REPLACE: photographer portrait */}
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
