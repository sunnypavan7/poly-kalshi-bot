import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { EXPERIENCES } from '../data/estate'

export default function Experiences() {
  return (
    <main>
      {/* Hero */}
      <div className="page-hero">
        <div className="page-hero-bg" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1605540436563-5bca919ae766?w=1800&q=82&auto=format)' }} />
        <div className="page-hero-scrim" />
        <div className="page-hero-cnt">
          <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.16,1,0.3,1], delay: 0.3 }}>Curated Activities</motion.p>
          <motion.h1 initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, ease: [0.16,1,0.3,1], delay: 0.45 }}>
            <em>Experiences</em>
          </motion.h1>
        </div>
      </div>

      {/* Intro */}
      <section className="section-pad">
        <div className="section-inner" style={{ maxWidth: 760, margin: '0 auto', textAlign: 'center' }}>
          <ScrollReveal>
            <p className="eyebrow">The Art of Remarkable Days</p>
            <h2 className="section-title" style={{ marginTop: 12, marginBottom: 24 }}>
              Curated entirely<br /><em>around you.</em>
            </h2>
            <div className="gold-rule" style={{ margin: '0 auto 32px' }} />
            <p style={{ fontSize: '0.93rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)' }}>
              Every activity at Nuage is designed to feel effortless. Our estate team handles all logistics, guides, and reservations — you simply arrive and experience.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Experience cards — 3-column grid, each with full detail */}
      <section className="section-pad bg-warm" style={{ paddingTop: 0 }}>
        <div className="section-inner">
          <div className="exp-grid">
            {EXPERIENCES.map((e, i) => (
              <ScrollReveal key={e.title} delay={(i % 3) * 0.1} direction="scale" className="exp-card cursor-grow">
                <img src={e.img} alt={e.title} loading="lazy" />
                <div className="exp-content">
                  <div className="exp-tag">{e.tag}</div>
                  <div className="exp-title">{e.title}</div>
                  <div className="exp-desc">{e.desc}</div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Seasonal guide */}
      <section className="section-pad">
        <div className="section-inner">
          <ScrollReveal className="section-hd">
            <p className="eyebrow">Season by Season</p>
            <h2 className="section-title">There is never<br />a wrong <em>time.</em></h2>
            <div className="gold-rule" />
          </ScrollReveal>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 2, marginTop: 64 }}>
            {[
              { season: 'Winter', months: 'Dec–Mar', highlights: ['Powder skiing', 'Snowshoeing', 'Ice sculpting', 'Fondue evenings', 'Northern lights viewing'] },
              { season: 'Spring', months: 'Apr–May', highlights: ['Wildflower hikes', 'Waterfall tours', 'Wine cellar evenings', 'Photography walks', 'Bird watching'] },
              { season: 'Summer', months: 'Jun–Sep', highlights: ['High-altitude trails', 'Wild swimming', 'Rock climbing', 'Paragliding', 'Botanical foraging'] },
              { season: 'Autumn', months: 'Oct–Nov', highlights: ['Larch forest walks', 'Truffle hunting', 'Harvest dinners', 'Spa retreats', 'Stargazing season'] },
            ].map((s, i) => (
              <ScrollReveal key={s.season} delay={i * 0.08}>
                <div style={{ padding: '40px 32px', background: i % 2 === 0 ? 'var(--warm)' : 'var(--cream)', height: '100%' }}>
                  <p style={{ fontFamily: 'var(--font-b)', fontSize: '0.62rem', fontWeight: 500, letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--gold)', marginBottom: 8 }}>{s.months}</p>
                  <h3 style={{ fontFamily: 'var(--font-d)', fontSize: '1.8rem', fontWeight: 300, color: 'var(--dark)', marginBottom: 24, lineHeight: 1.1 }}>{s.season}</h3>
                  <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {s.highlights.map(h => (
                      <li key={h} style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: '0.82rem', fontWeight: 300, color: 'var(--muted)' }}>
                        <span style={{ width: 16, height: 1, background: 'var(--gold)', flexShrink: 0, display: 'block' }} />
                        {h}
                      </li>
                    ))}
                  </ul>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-pad bg-dark" style={{ textAlign: 'center' }}>
        <ScrollReveal>
          <p className="eyebrow">Plan Your Stay</p>
          <h2 className="section-title on-dark" style={{ marginTop: 12, marginBottom: 28 }}>Tell us what you dream of.<br /><em>We'll arrange the rest.</em></h2>
          <Link to="/reserve" className="btn-pri ghost-light">Begin Enquiry</Link>
        </ScrollReveal>
      </section>
    </main>
  )
}
