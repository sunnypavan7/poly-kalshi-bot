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
      <section className="pt-36 md:pt-44 pb-12 md:pb-16 border-b border-[var(--color-border)]">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <ScrollReveal>
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-4">
              Portfolio
            </p>
            <h1 className="font-display font-light text-5xl md:text-7xl leading-none tracking-[-0.03em] text-[var(--color-warm-white)]">
              Selected Work
            </h1>
          </ScrollReveal>
        </div>
      </section>

      {/* Filter bar */}
      <section className="sticky top-[72px] z-30 bg-[var(--color-black)]/90 backdrop-blur-sm border-b border-[var(--color-border)]">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <div className="flex items-center gap-8 overflow-x-auto py-4 scrollbar-none">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                className={`font-sans text-xs tracking-[0.14em] uppercase whitespace-nowrap transition-colors duration-200 pb-1 border-b ${
                  activeCategory === cat
                    ? 'text-[var(--color-warm-white)] border-[var(--color-accent)]'
                    : 'text-[var(--color-muted)] border-transparent hover:text-[var(--color-warm-white)] hover:border-[var(--color-muted-dark)]'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Gallery grid */}
      <section className="max-w-[1440px] mx-auto px-8 md:px-16 py-12 md:py-16">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeCategory}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1, transition: { duration: 0.35 } }}
            exit={{ opacity: 0, transition: { duration: 0.2 } }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-1"
          >
            {filtered.map((project, i) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0, transition: { delay: i * 0.05, duration: 0.5, ease: [0.16, 1, 0.3, 1] } }}
              >
                <Link
                  to={`/portfolio/${project.id}`}
                  className="group block relative overflow-hidden bg-[var(--color-surface)]"
                  style={{ aspectRatio: project.aspect === 'portrait' ? '4/5' : project.aspect === 'square' ? '1/1' : '3/2' }}
                >
                  <img
                    src={project.cover}
                    alt={project.coverAlt}
                    className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-105"
                    loading="lazy"
                  />
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-6"
                    style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 55%)' }}
                  >
                    <span className="font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-accent)] mb-1">
                      {project.category}
                    </span>
                    <h2 className="font-display font-light text-xl text-[var(--color-warm-white)] leading-tight">
                      {project.title}
                    </h2>
                    <p className="font-sans text-xs text-[var(--color-warm-white)]/60 mt-1">
                      {project.location} · {project.year}
                    </p>
                  </div>
                </Link>
              </motion.div>
            ))}
          </motion.div>
        </AnimatePresence>

        {filtered.length === 0 && (
          <div className="py-32 text-center">
            <p className="font-display text-2xl text-[var(--color-muted)]">No projects in this category yet.</p>
          </div>
        )}
      </section>
    </main>
  )
}
