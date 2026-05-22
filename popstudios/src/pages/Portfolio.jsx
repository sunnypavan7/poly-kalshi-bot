import { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { projects, categories } from '../data/portfolio'

export default function Portfolio() {
  const [activeCategory, setActiveCategory] = useState('All')

  const filtered = activeCategory === 'All'
    ? projects
    : projects.filter(p => p.category === activeCategory)

  return (
    <main className="min-h-screen">
      {/* Header */}
      <section style={{ paddingTop: 'clamp(100px, 12vw, 160px)', paddingBottom: 48, borderBottom: '1px solid var(--color-border)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <ScrollReveal>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 14 }}>
              Portfolio
            </p>
            <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(3rem, 8vw, 6rem)', lineHeight: 0.9, letterSpacing: '-0.03em', color: 'var(--color-white)' }}>
              Selected Work
            </h1>
          </ScrollReveal>
        </div>
      </section>

      {/* Filter bar */}
      <div className="sticky z-30" style={{ top: 68, background: 'rgba(8,8,8,0.92)', backdropFilter: 'blur(12px)', borderBottom: '1px solid var(--color-border)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <div className="flex items-center gap-6 overflow-x-auto py-4" style={{ scrollbarWidth: 'none' }}>
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontWeight: 600,
                  fontSize: '0.72rem',
                  letterSpacing: '0.14em',
                  textTransform: 'uppercase',
                  whiteSpace: 'nowrap',
                  paddingBottom: 6,
                  borderBottom: `2px solid ${activeCategory === cat ? 'var(--color-accent)' : 'transparent'}`,
                  color: activeCategory === cat ? 'var(--color-white)' : 'var(--color-muted)',
                  transition: 'color 0.15s, border-color 0.15s',
                }}
                onMouseEnter={e => { if (activeCategory !== cat) e.currentTarget.style.color = 'var(--color-white)' }}
                onMouseLeave={e => { if (activeCategory !== cat) e.currentTarget.style.color = 'var(--color-muted)' }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Gallery grid */}
      <section className="max-w-[1440px] mx-auto px-6 md:px-12 py-10 md:py-14">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeCategory}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1, transition: { duration: 0.3 } }}
            exit={{ opacity: 0, transition: { duration: 0.2 } }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-1"
          >
            {filtered.map((project, i) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0, transition: { delay: i * 0.05, duration: 0.45, ease: [0.16, 1, 0.3, 1] } }}
              >
                <Link
                  to={`/portfolio/${project.id}`}
                  className="group block relative overflow-hidden"
                  style={{
                    aspectRatio: project.aspect === 'portrait' ? '4/5' : project.aspect === 'square' ? '1/1' : '3/2',
                    background: 'var(--color-surface)',
                  }}
                >
                  <img
                    src={project.cover}
                    alt={project.coverAlt}
                    className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-105"
                    loading="lazy"
                  />
                  <div
                    className="absolute inset-0 flex flex-col justify-end p-5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                    style={{ background: 'linear-gradient(to top, rgba(8,8,8,0.85) 0%, transparent 50%)' }}
                  >
                    <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.62rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 4 }}>
                      {project.category}
                    </span>
                    <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.2rem', color: 'var(--color-white)', lineHeight: 1.15 }}>
                      {project.title}
                    </h2>
                    <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'rgba(245,245,245,0.55)', marginTop: 3 }}>
                      {project.location} · {project.year}
                    </p>
                  </div>
                </Link>
              </motion.div>
            ))}
          </motion.div>
        </AnimatePresence>

        {filtered.length === 0 && (
          <div style={{ padding: '80px 0', textAlign: 'center' }}>
            <p style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: '1.4rem', color: 'var(--color-muted)' }}>
              No projects in this category yet.
            </p>
          </div>
        )}
      </section>
    </main>
  )
}
