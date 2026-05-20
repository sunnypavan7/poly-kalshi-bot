import { useState } from 'react'
import { motion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'

export default function Reserve() {
  const [sent, setSent] = useState(false)
  const [form, setForm] = useState({ first: '', last: '', email: '', phone: '', arrival: '', departure: '', guests: '', type: '', requests: '' })

  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }))
  const submit = e => { e.preventDefault(); setSent(true) }

  return (
    <main>
      {/* Hero */}
      <div className="page-hero" style={{ height: '50vh' }}>
        <div className="page-hero-bg" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=1800&q=82&auto=format)' }} />
        <div className="page-hero-scrim" />
        <div className="page-hero-cnt">
          <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.16,1,0.3,1], delay: 0.3 }}>Begin Your Journey</motion.p>
          <motion.h1 initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, ease: [0.16,1,0.3,1], delay: 0.45 }}>
            Reserve <em>Nuage</em>
          </motion.h1>
        </div>
      </div>

      {/* Main form section */}
      <section className="section-pad">
        <div className="section-inner" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80, alignItems: 'start' }}>

          {/* Left — info */}
          <div style={{ position: 'sticky', top: 100 }}>
            <ScrollReveal>
              <p className="eyebrow">Reservations</p>
              <h2 className="section-title" style={{ marginTop: 12 }}>Your <em>private</em><br />Alpine escape.</h2>
              <div className="gold-rule" />
            </ScrollReveal>

            <ScrollReveal delay={0.15}>
              <p style={{ fontSize: '0.9rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)', marginTop: 28, marginBottom: 40 }}>
                Minimum stay of seven nights. Our reservations team will respond within 24 hours to discuss availability, seasonal pricing, and bespoke arrangements for your visit.
              </p>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <div style={{ borderTop: '1px solid var(--stone)', paddingTop: 32, display: 'flex', flexDirection: 'column', gap: 20 }}>
                {[
                  { label: 'Reservations', value: 'reservations@nuage-estate.com' },
                  { label: 'Telephone', value: '+41 22 555 0100' },
                  { label: 'Minimum Stay', value: '7 nights' },
                  { label: 'Maximum Guests', value: '16 persons' },
                ].map(i => (
                  <div key={i.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 16 }}>
                    <span style={{ fontSize: '0.62rem', fontWeight: 500, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--muted)' }}>{i.label}</span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 300, color: 'var(--dark)' }}>{i.value}</span>
                  </div>
                ))}
              </div>
            </ScrollReveal>

            <ScrollReveal delay={0.25}>
              <div style={{ marginTop: 32, overflow: 'hidden', aspectRatio: '4/3' }}>
                <img src="https://images.unsplash.com/photo-1566073771259-6a8506099945?w=800&q=80&auto=format" alt="Aerial view of the Nuage Estate" loading="lazy" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
            </ScrollReveal>
          </div>

          {/* Right — form */}
          <ScrollReveal delay={0.1}>
            {sent ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                style={{ padding: '60px 0', textAlign: 'center' }}
              >
                <div style={{ width: 48, height: 48, borderRadius: '50%', background: 'var(--sage)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px' }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
                <p className="eyebrow">Enquiry Received</p>
                <h2 className="section-title" style={{ marginTop: 12, marginBottom: 16, fontSize: '2.4rem' }}>Thank you,<br /><em>{form.first || 'dear guest'}.</em></h2>
                <p style={{ fontSize: '0.9rem', fontWeight: 300, color: 'var(--muted)', lineHeight: 1.75, maxWidth: 400, margin: '0 auto' }}>
                  Our reservations team will be in touch within 24 hours to discuss your stay and begin crafting your Nuage experience.
                </p>
              </motion.div>
            ) : (
              <form onSubmit={submit}>
                <div className="form-row">
                  <div className="form-grp">
                    <label className="form-label">First Name</label>
                    <input className="form-ctrl" type="text" placeholder="James" value={form.first} onChange={set('first')} required />
                  </div>
                  <div className="form-grp">
                    <label className="form-label">Last Name</label>
                    <input className="form-ctrl" type="text" placeholder="Whitmore" value={form.last} onChange={set('last')} required />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-grp">
                    <label className="form-label">Email Address</label>
                    <input className="form-ctrl" type="email" placeholder="james@example.com" value={form.email} onChange={set('email')} required />
                  </div>
                  <div className="form-grp">
                    <label className="form-label">Phone (optional)</label>
                    <input className="form-ctrl" type="tel" placeholder="+44 20 7946 0958" value={form.phone} onChange={set('phone')} />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-grp">
                    <label className="form-label">Arrival Date</label>
                    <input className="form-ctrl" type="date" value={form.arrival} onChange={set('arrival')} required />
                  </div>
                  <div className="form-grp">
                    <label className="form-label">Departure Date</label>
                    <input className="form-ctrl" type="date" value={form.departure} onChange={set('departure')} required />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-grp">
                    <label className="form-label">Number of Guests</label>
                    <select className="form-ctrl" value={form.guests} onChange={set('guests')} required>
                      <option value="">Select</option>
                      {[2,4,6,8,10,12,14,16].map(n => <option key={n} value={n}>{n} guests</option>)}
                    </select>
                  </div>
                  <div className="form-grp">
                    <label className="form-label">Visit Type</label>
                    <select className="form-ctrl" value={form.type} onChange={set('type')}>
                      <option value="">Select</option>
                      <option>Family holiday</option>
                      <option>Couples retreat</option>
                      <option>Corporate offsite</option>
                      <option>Celebration</option>
                      <option>Wellness retreat</option>
                    </select>
                  </div>
                </div>

                <div className="form-grp">
                  <label className="form-label">Special Requests</label>
                  <textarea className="form-ctrl" placeholder="Tell us how we may make your stay truly exceptional — dietary requirements, celebrations, experiences you have in mind…" value={form.requests} onChange={set('requests')} style={{ height: 100 }} />
                </div>

                <button type="submit" className="btn-pri dark-fill" style={{ marginTop: 8, width: '100%', textAlign: 'center' }}>
                  Send Enquiry
                </button>

                <p style={{ fontSize: '0.7rem', fontWeight: 300, color: 'var(--muted)', marginTop: 16, lineHeight: 1.6 }}>
                  By submitting this form you agree to our privacy policy. We never share your details with third parties.
                </p>
              </form>
            )}
          </ScrollReveal>
        </div>
      </section>

      {/* Trust signals */}
      <section className="section-pad bg-warm" style={{ paddingTop: 60, paddingBottom: 60 }}>
        <div className="section-inner">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 40, textAlign: 'center' }}>
            {[
              { n: '24h',    l: 'Response time' },
              { n: '15+',    l: 'Years of excellence' },
              { n: '100%',   l: 'Private & exclusive' },
              { n: '5★',     l: 'Guest rating' },
            ].map((s, i) => (
              <ScrollReveal key={s.l} delay={i * 0.08}>
                <div style={{ fontFamily: 'var(--font-d)', fontSize: '2.4rem', fontWeight: 300, color: 'var(--gold)', letterSpacing: '-0.02em' }}>{s.n}</div>
                <div style={{ fontSize: '0.62rem', fontWeight: 500, letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--muted)', marginTop: 6 }}>{s.l}</div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>
    </main>
  )
}
