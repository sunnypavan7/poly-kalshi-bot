import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Lightbox from '../components/Lightbox'
import ScrollReveal from '../components/ScrollReveal'
import { GALLERY, GALLERY_CATS } from '../data/estate'

export default function Gallery() {
  const [cat, setCat]           = useState('All')
  const [lbIndex, setLbIndex]   = useState(null)

  const filtered = cat === 'All' ? GALLERY : GALLERY.filter(g => g.cat === cat)

  const open  = i => setLbIndex(i)
  const close = ()  => setLbIndex(null)
  const prev  = ()  => setLbIndex(i => (i - 1 + filtered.length) % filtered.length)
  const next  = ()  => setLbIndex(i => (i + 1) % filtered.length)

  const layouts = ['span8', 'span4', 'span5', 'span7', 'span4', 'span4', 'span4', 'span6', 'span6']
  const getSpan = i => layouts[i % layouts.length] || 'span4'

  return (
    <>
      {/* Hero */}
      <div className="page-hero" style={{ '--bg': 'url(https://images.unsplash.com/photo-1566073771259-6a8506099945?w=1800&q=82&auto=format)' }}>
        <div className="page-hero-bg" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1566073771259-6a8506099945?w=1800&q=82&auto=format)' }} />
        <div className="page-hero-scrim" />
        <div className="page-hero-cnt">
          <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.16,1,0.3,1], delay: 0.3 }}>Visual Archive</motion.p>
          <motion.h1 initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, ease: [0.16,1,0.3,1], delay: 0.45 }}>The <em>Gallery</em></motion.h1>
        </div>
      </div>

      {/* Gallery */}
      <section className="section-pad">
        <div className="section-inner">
          {/* Filter tabs */}
          <div className="filter-tabs">
            {GALLERY_CATS.map(c => (
              <button key={c} className={`filter-tab${cat === c ? ' active' : ''}`} onClick={() => setCat(c)}>{c}</button>
            ))}
          </div>

          {/* Grid */}
          <motion.div className="gal-grid full" layout>
            <AnimatePresence>
              {filtered.map((img, i) => (
                <motion.div
                  key={img.id}
                  className={`gal-item ${getSpan(i)} cursor-grow`}
                  layout
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.96 }}
                  transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1], delay: i * 0.03 }}
                  onClick={() => open(i)}
                  style={{ cursor: 'none' }}
                >
                  <img src={img.src} alt={img.alt} loading="lazy" />
                  <div className="gal-overlay">
                    <span className="gal-label">{img.label}</span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        </div>
      </section>

      {/* Lightbox */}
      <AnimatePresence>
        {lbIndex !== null && (
          <Lightbox
            images={filtered}
            index={lbIndex}
            onClose={close}
            onPrev={prev}
            onNext={next}
          />
        )}
      </AnimatePresence>
    </>
  )
}
