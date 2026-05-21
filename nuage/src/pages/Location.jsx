import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { LOCATION_FACTS } from '../data/estate'

export default function Location() {
  return (
    <main>
      {/* Hero */}
      <div className="page-hero" style={{ height: '60vh' }}>
        <div className="page-hero-bg" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=1800&q=82&auto=format)' }} />
        <div className="page-hero-scrim" />
        <div className="page-hero-cnt">
          <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.16,1,0.3,1], delay: 0.3 }}>Setting & Access</motion.p>
          <motion.h1 initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, ease: [0.16,1,0.3,1], delay: 0.45 }}>
            Above <em>Zermatt</em>
          </motion.h1>
        </div>
      </div>

      {/* Overview */}
      <section className="section-pad">
        <div className="section-inner">
          <div className="loc-grid">
            <ScrollReveal direction="scale" className="loc-img">
              <img src="https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=1000&q=80&auto=format" alt="Aerial view of the Swiss Alps and Mattertal valley" loading="lazy" />
            </ScrollReveal>
            <div>
              <ScrollReveal>
                <p className="eyebrow">A World Apart</p>
                <h2 className="section-title" style={{ marginTop: 12 }}>Mattertal,<br /><em>Switzerland.</em></h2>
                <div className="gold-rule" />
              </ScrollReveal>
              <ScrollReveal delay={0.15}>
                <p style={{ fontSize: '0.93rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)', marginTop: 28, marginBottom: 28 }}>
                  Nuage Estate sits at 2,400 metres in the Mattertal valley with direct sight lines to the Matterhorn. The location combines near-absolute seclusion with discreet proximity to Zermatt's village and its constellation of Michelin-starred dining.
                </p>
              </ScrollReveal>
              <ul className="loc-list">
                {LOCATION_FACTS.map((f, i) => (
                  <ScrollReveal key={f} delay={0.15 + i * 0.07} style={{ display: 'contents' }}>
                    <li>{f}</li>
                  </ScrollReveal>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Full-bleed landscape image */}
      <div style={{ height: '50vh', overflow: 'hidden', position: 'relative' }}>
        <img
          src="https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=2000&q=80&auto=format"
          alt="Mountain peaks above the Nuage Estate"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          loading="lazy"
        />
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.25)' }}>
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontFamily: 'var(--font-d)', fontSize: 'clamp(2rem, 5vw, 4rem)', fontWeight: 300, color: 'white', fontStyle: 'italic', letterSpacing: '-0.01em' }}>
              "Above the world."
            </p>
          </div>
        </div>
      </div>

      {/* Getting here */}
      <section className="section-pad bg-warm">
        <div className="section-inner">
          <ScrollReveal className="section-hd">
            <p className="eyebrow">Getting Here</p>
            <h2 className="section-title">Seamless <em>arrival</em>,<br />however you travel.</h2>
            <div className="gold-rule" />
          </ScrollReveal>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2, marginTop: 64 }}>
            {[
              { icon: '✈', mode: 'By Air', title: 'Helicopter', detail: 'The estate\'s helipad accepts most light helicopters. Geneva International to estate: 35 minutes. Zurich: 40 minutes. Sion Airport: 12 minutes. Our concierge coordinates charter and scheduling.' },
              { icon: '🚗', mode: 'By Road', title: 'Private Transfer', detail: 'Chauffeur-driven luxury transfer from Geneva, Zurich, or Sion airports. The road to Zermatt ends at Täsch; from there a private cable car brings you directly to the estate gate in 8 minutes.' },
              { icon: '🚆', mode: 'By Rail', title: 'Train & Cable Car', detail: 'The Matterhorn Gotthard Railway from Visp or Brig to Zermatt is one of the great scenic rail journeys in Europe. From Zermatt station, the estate cable car departs every 20 minutes.' },
            ].map((m, i) => (
              <ScrollReveal key={m.mode} delay={i * 0.1}>
                <div style={{ padding: '48px 36px', background: 'var(--cream)', height: '100%' }}>
                  <p style={{ fontFamily: 'var(--font-b)', fontSize: '0.62rem', fontWeight: 500, letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--gold)', marginBottom: 12 }}>{m.mode}</p>
                  <h3 style={{ fontFamily: 'var(--font-d)', fontSize: '1.7rem', fontWeight: 400, color: 'var(--dark)', marginBottom: 16 }}>{m.title}</h3>
                  <p style={{ fontSize: '0.85rem', fontWeight: 300, lineHeight: 1.82, color: 'var(--muted)' }}>{m.detail}</p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Nearby */}
      <section className="section-pad">
        <div className="section-inner">
          <ScrollReveal className="section-hd">
            <p className="eyebrow">The Surroundings</p>
            <h2 className="section-title">Zermatt & <em>beyond.</em></h2>
            <div className="gold-rule" />
          </ScrollReveal>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80, alignItems: 'center', marginTop: 64 }}>
            <ScrollReveal direction="scale">
              <div style={{ overflow: 'hidden', aspectRatio: '4/3' }}>
                <img src="https://images.unsplash.com/photo-1604014237800-1c9102c219da?w=900&q=80&auto=format" alt="Zermatt village in the valley below the estate" loading="lazy" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
            </ScrollReveal>
            <div>
              <ScrollReveal delay={0.1}>
                <p style={{ fontSize: '0.93rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)', marginBottom: 28 }}>
                  Zermatt is one of the world's great mountain destinations — car-free, impossibly picturesque, and home to two three-Michelin-starred restaurants. The town sits 8 minutes below the estate by private cable car.
                </p>
                <p style={{ fontSize: '0.93rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)', marginBottom: 36 }}>
                  Beyond Zermatt, the wider Valais canton offers wine villages, thermal spas, and a landscape that has inspired writers and artists for centuries. The estate team curates all day trips and reservations.
                </p>
                <Link to="/experiences" className="btn-txt">
                  Explore Experiences
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                </Link>
              </ScrollReveal>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-pad bg-dark" style={{ textAlign: 'center' }}>
        <ScrollReveal>
          <p className="eyebrow">Your Journey Begins</p>
          <h2 className="section-title on-dark" style={{ marginTop: 12, marginBottom: 28 }}>
            Reserve Nuage.<br /><em>Arrive above the world.</em>
          </h2>
          <Link to="/reserve" className="btn-pri ghost-light">Begin Enquiry</Link>
        </ScrollReveal>
      </section>
    </main>
  )
}
