import { Link } from 'react-router-dom'
import ScrollReveal from '../components/ScrollReveal'
import { packages, PROCESS, TESTIMONIALS } from '../data/services'

function PackageCard({ pkg, index }) {
  return (
    <ScrollReveal delay={index * 0.1}>
      <div
        className="group flex flex-col h-full transition-colors duration-200"
        style={{ border: '1px solid var(--color-border)', padding: '36px 32px' }}
        onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--color-accent)'}
        onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--color-border)'}
      >
        <div style={{ marginBottom: 28 }}>
          <p style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '0.62rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 16 }}>
            0{index + 1}
          </p>
          <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '1.8rem', color: 'var(--color-white)', marginBottom: 8, lineHeight: 1 }}>
            {pkg.name}
          </h2>
          <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.85rem', color: 'var(--color-muted)', lineHeight: 1.6 }}>
            {pkg.tagline}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 28 }}>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '1.8rem', color: 'var(--color-accent)' }}>
            {pkg.price}
          </span>
          <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'var(--color-muted)', letterSpacing: '0.04em' }}>
            / {pkg.duration}
          </span>
        </div>

        <ul style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 28, flex: 1 }}>
          {pkg.includes.map((item) => (
            <li key={item} style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
              <span style={{ marginTop: 8, width: 16, height: 1.5, background: 'var(--color-accent)', flexShrink: 0, display: 'block' }} aria-hidden="true" />
              <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.82rem', color: 'var(--color-muted)', lineHeight: 1.65 }}>{item}</span>
            </li>
          ))}
        </ul>

        <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'var(--color-muted-dark)', lineHeight: 1.65, fontStyle: 'italic', borderTop: '1px solid var(--color-border)', paddingTop: 20, marginBottom: 24 }}>
          {pkg.note}
        </p>

        <Link
          to="/contact"
          style={{ display: 'block', fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.78rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-white)', border: '1px solid var(--color-border)', padding: '12px 0', textAlign: 'center', transition: 'all 0.2s' }}
          onMouseEnter={e => { e.currentTarget.style.background = 'var(--color-accent)'; e.currentTarget.style.borderColor = 'var(--color-accent)'; e.currentTarget.style.color = 'var(--color-black)' }}
          onMouseLeave={e => { e.currentTarget.style.background = ''; e.currentTarget.style.borderColor = 'var(--color-border)'; e.currentTarget.style.color = 'var(--color-white)' }}
        >
          Inquire
        </Link>
      </div>
    </ScrollReveal>
  )
}

export default function Services() {
  return (
    <main>
      {/* Header */}
      <section style={{ paddingTop: 'clamp(100px, 12vw, 160px)', paddingBottom: 64, borderBottom: '1px solid var(--color-border)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-end">
            <ScrollReveal>
              <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 14 }}>
                Services
              </p>
              <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(3rem, 8vw, 6rem)', lineHeight: 0.9, letterSpacing: '-0.03em', color: 'var(--color-white)' }}>
                Packages &amp;<br />Pricing
              </h1>
            </ScrollReveal>
            <ScrollReveal delay={0.15} direction="left">
              <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '1rem', color: 'var(--color-muted)', lineHeight: 1.8, maxWidth: 380 }}>
                Transparent pricing, zero hidden charges. Every project starts with a conversation — these are starting points, not limits.
              </p>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* Packages */}
      <section className="max-w-[1440px] mx-auto px-6 md:px-12" style={{ paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-1 md:gap-4">
          {packages.map((pkg, i) => (
            <PackageCard key={pkg.id} pkg={pkg} index={i} />
          ))}
        </div>
      </section>

      {/* Process */}
      <section style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <ScrollReveal>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 12 }}>
              Process
            </p>
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2rem, 4vw, 3rem)', lineHeight: 1, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 56 }}>
              How it works
            </h2>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 md:gap-10">
            {PROCESS.map((step, i) => (
              <ScrollReveal key={step.n} delay={i * 0.08}>
                <div>
                  <p style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '3rem', color: 'var(--color-border)', marginBottom: 20, lineHeight: 1 }}>
                    {step.n}
                  </p>
                  <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.15rem', color: 'var(--color-white)', marginBottom: 10 }}>
                    {step.title}
                  </h3>
                  <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.82rem', color: 'var(--color-muted)', lineHeight: 1.75 }}>
                    {step.body}
                  </p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section style={{ background: 'var(--color-surface)', borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <ScrollReveal>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 40 }}>
              Client Voices
            </p>
          </ScrollReveal>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
            {TESTIMONIALS.map((t, i) => (
              <ScrollReveal key={t.name} delay={i * 0.1}>
                <p style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: '1rem', color: 'var(--color-white)', lineHeight: 1.65, marginBottom: 16, fontStyle: 'italic' }}>
                  "{t.quote}"
                </p>
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.78rem', color: 'var(--color-accent)' }}>
                  {t.name}
                </p>
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.72rem', color: 'var(--color-muted)', marginTop: 2 }}>
                  {t.detail}
                </p>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(80px, 10vw, 120px)', paddingBottom: 'clamp(80px, 10vw, 120px)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12 text-center">
          <ScrollReveal>
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2rem, 5vw, 4rem)', lineHeight: 1, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 20 }}>
              Ready to book?
            </h2>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'var(--color-muted)', fontSize: '0.93rem', marginBottom: 36 }}>
              We respond within 24 hours. WhatsApp or email — your choice.
            </p>
            <Link
              to="/contact"
              style={{ display: 'inline-flex', alignItems: 'center', gap: 12, fontFamily: 'var(--font-sans)', fontWeight: 700, fontSize: '0.82rem', letterSpacing: '0.12em', textTransform: 'uppercase', background: 'var(--color-accent)', color: 'var(--color-black)', padding: '16px 36px', transition: 'background 0.2s' }}
              onMouseEnter={e => e.currentTarget.style.background = '#e04400'}
              onMouseLeave={e => e.currentTarget.style.background = 'var(--color-accent)'}
            >
              Get in Touch
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </ScrollReveal>
        </div>
      </section>
    </main>
  )
}
