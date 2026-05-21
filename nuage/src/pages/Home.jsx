import { useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { AMENITIES, GALLERY, TESTIMONIALS } from '../data/estate'

/* ─── SVG ICONS ─── */
function Icon({ type }) {
  const d = {
    pool:    <path d="M3 12h18M12 3c0 5-3 8-3 8s3 1 3 9m0-17c0 5 3 8 3 8s-3 1-3 9"/>,
    spa:     <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>,
    cinema:  <><rect x="2" y="7" width="20" height="15" rx="2"/><polyline points="17,2 12,7 7,2"/></>,
    wine:    <><line x1="12" y1="2" x2="12" y2="22"/><path d="M5 5a7 7 0 0 0 7 7 7 7 0 0 0 7-7"/></>,
    team:    <><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></>,
    heli:    <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>,
    ski:     <><path d="M3 17l4-4 4 4 4-8 4 4"/><path d="M3 21h18"/></>,
    kitchen: <><path d="M3 3h18v4H3z"/><path d="M3 7v14"/><path d="M21 7v14"/><path d="M3 14h18"/></>,
    library: <><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></>,
  }
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      {d[type]}
    </svg>
  )
}

/* ─── HERO ─── */
function Hero() {
  const ref = useRef(null)
  const { scrollY } = useScroll()
  const y = useTransform(scrollY, [0, 700], [0, 200])

  const words = ['House', 'on', 'the']
  const em = 'Clouds'

  return (
    <section ref={ref} style={{ position: 'relative', height: '100vh', minHeight: 680, overflow: 'hidden', display: 'flex', alignItems: 'flex-end', paddingBottom: 72 }}>
      {/* Parallax BG */}
      <motion.div
        style={{
          position: 'absolute', inset: 0, y,
          backgroundImage: 'url(https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=2000&q=85&auto=format)',
          backgroundSize: 'cover', backgroundPosition: 'center',
        }}
      />
      {/* Scrim */}
      <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(160deg, rgba(0,0,0,0.08) 0%, rgba(0,0,0,0.15) 40%, rgba(0,0,0,0.62) 100%)' }} />

      {/* Content */}
      <div style={{ position: 'relative', zIndex: 2, padding: '0 60px', width: '100%' }}>
        <motion.p
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.5 }}
          style={{ fontFamily: 'var(--font-b)', fontSize: '0.68rem', fontWeight: 500, letterSpacing: '0.22em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.58)', marginBottom: 18 }}
        >
          Private Estate &nbsp;·&nbsp; Swiss Alps &nbsp;·&nbsp; Est. 1998
        </motion.p>

        <h1 style={{ fontFamily: 'var(--font-d)', fontWeight: 300, lineHeight: 1.0, letterSpacing: '-0.02em', color: 'white', fontSize: 'clamp(3.8rem, 8.5vw, 7.5rem)', marginBottom: 26 }}>
          {words.map((w, i) => (
            <motion.span
              key={w}
              initial={{ opacity: 0, y: 28 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.7 + i * 0.1 }}
              style={{ display: 'inline-block', marginRight: '0.28em' }}
            >
              {w}
            </motion.span>
          ))}
          <br />
          <motion.em
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.0, ease: [0.16, 1, 0.3, 1], delay: 1.0 }}
            style={{ fontStyle: 'italic', display: 'inline-block' }}
          >
            {em}
          </motion.em>
        </h1>

        <motion.p
          initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 1.1 }}
          style={{ fontSize: '0.9rem', fontWeight: 300, lineHeight: 1.75, color: 'rgba(255,255,255,0.75)', maxWidth: 380 }}
        >
          An intimate sanctuary poised above the valley floor, where morning light spills through mountain peaks and time moves at its own pace.
        </motion.p>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        transition={{ delay: 1.6, duration: 0.8 }}
        style={{ position: 'absolute', right: 60, bottom: 44, display: 'flex', alignItems: 'center', gap: 14 }}
      >
        <span style={{ fontSize: '0.6rem', fontWeight: 500, letterSpacing: '0.22em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.45)', writingMode: 'vertical-rl' }}>Discover</span>
        <div style={{ width: 1, height: 56, background: 'rgba(255,255,255,0.22)', position: 'relative', overflow: 'hidden' }}>
          <motion.div
            style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '100%', background: 'var(--gold)' }}
            animate={{ y: ['−100%', '100%'] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear', delay: 2 }}
          />
        </div>
      </motion.div>

      {/* Bottom gold rule */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 1, background: 'linear-gradient(to right, transparent, rgba(197,160,104,0.35), transparent)' }} />
    </section>
  )
}

/* ─── STATS BAR ─── */
function StatsBar() {
  const stats = [
    { n: '8',         l: 'Bedrooms' },
    { n: '16',        l: 'Guests'   },
    { n: <>2,400<sup style={{ fontSize: '1.1rem' }}>m</sup></>, l: 'Altitude' },
    { n: '12',        l: 'Hectares' },
  ]
  return (
    <div className="stats-bar">
      <div className="stats-group">
        {stats.map((s, i) => (
          <ScrollReveal key={s.l} delay={i * 0.08}>
            <div className="stat-n">{s.n}</div>
            <div className="stat-l">{s.l}</div>
          </ScrollReveal>
        ))}
      </div>
      <ScrollReveal delay={0.3}>
        <p className="stats-quote">"Where the earth meets the sky, and every moment becomes a memory."</p>
      </ScrollReveal>
    </div>
  )
}

/* ─── DISCOVER ─── */
function Discover() {
  return (
    <section style={{ padding: '110px 60px', maxWidth: 1400, margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80, alignItems: 'center' }}>
      <ScrollReveal>
        <blockquote style={{ fontFamily: 'var(--font-d)', fontSize: 'clamp(1.7rem, 2.7vw, 2.5rem)', fontWeight: 300, fontStyle: 'italic', lineHeight: 1.35, color: 'var(--dark)', borderLeft: '2px solid var(--gold)', paddingLeft: 32 }}>
          "To stay at Nuage is to understand what it means to be truly removed from the world — yet want for nothing."
        </blockquote>
      </ScrollReveal>
      <div>
        <ScrollReveal delay={0.1}>
          <p style={{ fontSize: '0.94rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)', marginBottom: 18 }}>
            Perched at 2,400 metres above sea level in the heart of the Swiss Alps, Nuage Estate offers an unparalleled private retreat for those who seek seclusion, grandeur, and a profound connection with the natural world.
          </p>
        </ScrollReveal>
        <ScrollReveal delay={0.2}>
          <p style={{ fontSize: '0.94rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)', marginBottom: 32 }}>
            The estate encompasses a fully staffed 8-bedroom residence, private ski access, a heated outdoor infinity pool overlooking three valleys, and a spa that rivals the world's finest.
          </p>
        </ScrollReveal>
        <ScrollReveal delay={0.3}>
          <Link to="/estate" className="btn-txt">
            Explore the Estate
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          </Link>
        </ScrollReveal>
      </div>
    </section>
  )
}

/* ─── ESTATE FEATURES ─── */
function EstateFeatures() {
  const items = [
    { num: '01', title: 'Architecture', body: 'Conceived by Pritzker laureate Jean-Marc Pellerin, the main house is a dialogue between raw Alpine stone and floor-to-ceiling glass — respecting the mountain\'s permanence while embracing its panoramic, ever-shifting light.', img: 'https://images.unsplash.com/photo-1613977257592-4871e5fcd7c4?w=1200&q=82&auto=format', alt: 'Exterior of the Nuage Estate', link: '/estate' },
    { num: '02', title: 'Interiors',     body: 'Curated over two decades by Maison Artefact in Brussels, the interiors blend museum-quality art with handwoven textiles, reclaimed Alpine timber, and bespoke furniture. Each of the eight suites is a world unto itself.', img: 'https://images.unsplash.com/photo-1615529328331-f8917597711f?w=1200&q=82&auto=format', alt: 'Master suite interior', link: '/estate' },
    { num: '03', title: 'The Grounds',  body: 'Twelve hectares of private land include manicured gardens, a heated infinity pool, a cedar-clad spa pavilion, two groomed ski runs, and a helipad for seamless arrival from Geneva or Zurich.', img: 'https://images.unsplash.com/photo-1571896349842-33c89424de2d?w=1200&q=82&auto=format', alt: 'Heated infinity pool', link: '/experiences' },
  ]

  return (
    <section className="estate">
      {items.map((item, i) => (
        <div className="feat-pair" key={item.num}>
          <ScrollReveal direction="scale" className="feat-img">
            <img src={item.img} alt={item.alt} loading="lazy" />
          </ScrollReveal>
          <div className="feat-body">
            <ScrollReveal><div className="feat-num">{item.num}</div></ScrollReveal>
            <ScrollReveal delay={0.1}><h3>{item.title}</h3></ScrollReveal>
            <ScrollReveal delay={0.2}><p>{item.body}</p></ScrollReveal>
            <ScrollReveal delay={0.3}>
              <Link to={item.link} className="btn-txt">
                Discover more
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
              </Link>
            </ScrollReveal>
          </div>
        </div>
      ))}
    </section>
  )
}

/* ─── AMENITIES ─── */
function AmenitiesPreview() {
  return (
    <section className="section-pad bg-dark" id="amenities">
      <div className="section-inner">
        <ScrollReveal className="section-hd">
          <p className="eyebrow">World-Class Facilities</p>
          <h2 className="section-title on-dark">Every <em>luxury</em>,<br />considered.</h2>
          <div className="gold-rule" />
        </ScrollReveal>
        <div className="amen-grid" style={{ marginTop: 64 }}>
          {AMENITIES.map((a, i) => (
            <ScrollReveal key={a.name} delay={(i % 3) * 0.08} className="amen-card">
              <div className="amen-ico"><Icon type={a.icon} /></div>
              <div className="amen-name">{a.name}</div>
              <div className="amen-desc">{a.desc}</div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ─── GALLERY PREVIEW ─── */
function GalleryPreview() {
  const preview = GALLERY.slice(0, 6)
  const spans = ['span7', 'span5', 'span5', 'span4', 'span4', 'span4']

  return (
    <section className="section-pad">
      <div className="section-inner">
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', marginBottom: 64 }}>
          <ScrollReveal>
            <p className="eyebrow">Visual Journey</p>
            <h2 className="section-title">The <em>estate</em>,<br />in full.</h2>
            <div className="gold-rule" />
          </ScrollReveal>
          <ScrollReveal delay={0.2}>
            <Link to="/gallery" className="btn-txt">
              Full Gallery
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </Link>
          </ScrollReveal>
        </div>
        <div className="gal-grid">
          {preview.map((img, i) => (
            <ScrollReveal key={img.id} direction="scale" delay={(i % 3) * 0.07} className={`gal-item ${spans[i]}`}>
              <img src={img.src} alt={img.alt} loading="lazy" />
              <div className="gal-overlay"><span className="gal-label">{img.label}</span></div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ─── TESTIMONIALS ─── */
function Testimonials() {
  const [cur, setCur] = useState(0)

  return (
    <section className="section-pad bg-dark" style={{ textAlign: 'center' }}>
      <div className="section-inner">
        <ScrollReveal className="section-hd center">
          <p className="eyebrow">Guest Voices</p>
          <h2 className="section-title on-dark">Words from<br /><em>those who know.</em></h2>
          <div className="gold-rule" style={{ margin: '20px auto 0' }} />
        </ScrollReveal>

        <div style={{ maxWidth: 780, margin: '64px auto 0' }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={cur}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            >
              <p className="testi-q">"{TESTIMONIALS[cur].quote}"</p>
              <p className="testi-author">{TESTIMONIALS[cur].author}</p>
              <p className="testi-org">{TESTIMONIALS[cur].origin}</p>
            </motion.div>
          </AnimatePresence>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 44 }}>
          {TESTIMONIALS.map((_, i) => (
            <button key={i} className={`testi-dot${i === cur ? ' active' : ''}`} onClick={() => setCur(i)} aria-label={`Testimonial ${i + 1}`} />
          ))}
        </div>
      </div>
    </section>
  )
}

/* ─── CTA ─── */
function CTA() {
  return (
    <section style={{ padding: '100px 60px', borderTop: '1px solid var(--stone)' }}>
      <div style={{ maxWidth: 1400, margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80, alignItems: 'center' }}>
        <ScrollReveal>
          <p className="eyebrow">Begin Your Journey</p>
          <h2 className="section-title" style={{ marginTop: 12, marginBottom: 28 }}>
            Reserve <em>Nuage</em><br />for your escape.
          </h2>
          <p style={{ fontSize: '0.9rem', fontWeight: 300, lineHeight: 1.82, color: 'var(--muted)', maxWidth: 400, marginBottom: 36 }}>
            Minimum stay of seven nights. Our reservations team will respond within 24 hours to discuss availability and bespoke arrangements for your visit.
          </p>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            <Link to="/reserve" className="btn-pri dark-fill">Reserve Now</Link>
            <Link to="/estate" className="btn-pri outline">Explore Estate</Link>
          </div>
        </ScrollReveal>
        <ScrollReveal direction="scale" delay={0.15}>
          <div style={{ overflow: 'hidden', aspectRatio: '4/3' }}>
            <img src="https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=900&q=80&auto=format" alt="The Nuage Estate illuminated at dusk" loading="lazy" style={{ width: '100%', height: '100%', objectFit: 'cover', transition: 'transform 0.9s var(--ease-cin)' }} className="cursor-grow" />
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}

/* ─── PAGE ─── */
export default function Home() {
  return (
    <main>
      <Hero />
      <StatsBar />
      <Discover />
      <EstateFeatures />
      <AmenitiesPreview />
      <GalleryPreview />
      <Testimonials />
      <CTA />
    </main>
  )
}
