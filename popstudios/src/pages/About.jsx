import { Link } from 'react-router-dom'
import ScrollReveal from '../components/ScrollReveal'

const clients = [
  'Fabindia', 'Tanishq', 'Nykaa Fashion', 'YourStory Media',
  'Rolling Stone India', 'Vogue India', 'Hindustan Unilever', 'CRED',
]

const philosophy = [
  {
    heading: 'Light is the material',
    body: 'Every location, timing, and equipment decision begins and ends with light. We don\'t manufacture it. We find it and wait for it.',
  },
  {
    heading: 'Less, but better',
    body: 'We take 25–30 weddings and a limited number of commercial projects a year. Fewer commissions means more time and attention per frame.',
  },
  {
    heading: 'Documentary instinct',
    body: 'Even in a fully planned shoot, we\'re looking for the unguarded moment. The best photographs look like they couldn\'t have been planned.',
  },
]

export default function About() {
  return (
    <main>
      {/* Header */}
      <section style={{ paddingTop: 'clamp(100px, 12vw, 160px)', paddingBottom: 64, borderBottom: '1px solid var(--color-border)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-end">
            <ScrollReveal>
              <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 14 }}>
                About
              </p>
              <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(3rem, 8vw, 6rem)', lineHeight: 0.9, letterSpacing: '-0.03em', color: 'var(--color-white)' }}>
                The Studio
              </h1>
            </ScrollReveal>
            <ScrollReveal delay={0.15} direction="left">
              <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '1rem', color: 'var(--color-muted)', lineHeight: 1.8, maxWidth: 420 }}>
                PopStudios is a Mumbai-based photography studio working at the intersection of documentary and fine art — obsessively attentive to light, honest about emotion.
              </p>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* Founder bio */}
      <section className="max-w-[1440px] mx-auto px-6 md:px-12" style={{ paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12 md:gap-16 items-start">
          <ScrollReveal className="md:col-span-5">
            <div className="relative overflow-hidden" style={{ aspectRatio: '3/4', background: 'var(--color-surface)' }}>
              <img
                src="https://images.unsplash.com/photo-1554151228-14d9def656e4?auto=format&fit=crop&w=900&q=80"
                alt="Rohan Kapoor, founder photographer in his Bandra studio"
                className="w-full h-full object-cover"
                loading="eager"
              />
              <div className="absolute bottom-0 left-0 right-0" style={{ padding: '24px 24px', background: 'linear-gradient(to top, rgba(8,8,8,0.9) 0%, transparent 100%)' }}>
                <p style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1rem', color: 'var(--color-white)' }}>Rohan Kapoor</p>
                <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.75rem', color: 'var(--color-accent)', marginTop: 2 }}>Founder & Lead Photographer</p>
              </div>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={0.15} direction="left" className="md:col-span-6 md:col-start-7 md:pt-10">
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 18 }}>
              Founder
            </p>
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(1.8rem, 3vw, 2.4rem)', color: 'var(--color-white)', lineHeight: 1.1, marginBottom: 24 }}>
              "A photograph should make you feel something before you understand it."
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'var(--color-muted)', lineHeight: 1.8, fontSize: '0.9rem' }}>
              <p>
                Rohan Kapoor started PopStudios in 2017 after years working as a photojournalist for publications across South Asia. The move to commercial and wedding work wasn't a compromise — it was a different canvas for the same obsession.
              </p>
              <p>
                Based in Bandra, Mumbai, he's shot weddings from Kerala backwaters to Rajasthan forts, portraits of founders, musicians, and athletes, and commercial campaigns for brands building something real.
              </p>
              <p>
                The studio takes on 25–30 weddings per year and a small number of commercial and portrait projects — chosen for the quality of the collaboration, not just the brief.
              </p>
            </div>

            <div style={{ marginTop: 32, display: 'flex', gap: 24 }}>
              <a
                href="https://instagram.com/popstudios"
                target="_blank"
                rel="noopener noreferrer"
                style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', transition: 'color 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--color-white)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--color-muted)'}
              >
                @popstudios
              </a>
              <a
                href="mailto:hello@popstudios.in"
                style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', transition: 'color 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--color-white)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--color-muted)'}
              >
                Email
              </a>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* Philosophy */}
      <section style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <ScrollReveal>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 12 }}>
              Philosophy
            </p>
            <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2rem, 4vw, 3rem)', lineHeight: 1, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 56 }}>
              How we work
            </h2>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-10 md:gap-14">
            {philosophy.map((item, i) => (
              <ScrollReveal key={item.heading} delay={i * 0.1}>
                <div>
                  <p style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '0.62rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 16 }}>
                    0{i + 1}
                  </p>
                  <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.25rem', color: 'var(--color-white)', marginBottom: 12 }}>
                    {item.heading}
                  </h3>
                  <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.85rem', color: 'var(--color-muted)', lineHeight: 1.8 }}>
                    {item.body}
                  </p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Studio wide shot */}
      <section className="max-w-[1440px] mx-auto px-6 md:px-12" style={{ paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <ScrollReveal>
          <div className="overflow-hidden" style={{ aspectRatio: '16/7', background: 'var(--color-surface)' }}>
            <img
              src="https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1920&q=80"
              alt="PopStudios Bandra workspace and equipment"
              className="w-full h-full object-cover"
              loading="lazy"
            />
          </div>
        </ScrollReveal>
      </section>

      {/* Clients */}
      <section style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <ScrollReveal>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 40 }}>
              Brands &amp; Publications
            </p>
          </ScrollReveal>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-8 gap-y-5">
            {clients.map((name, i) => (
              <ScrollReveal key={name} delay={i * 0.05}>
                <p
                  style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: '1rem', color: 'var(--color-muted)', cursor: 'default', transition: 'color 0.15s' }}
                  onMouseEnter={e => e.currentTarget.style.color = 'var(--color-white)'}
                  onMouseLeave={e => e.currentTarget.style.color = 'var(--color-muted)'}
                >
                  {name}
                </p>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <div className="max-w-[1440px] mx-auto px-6 md:px-12">
          <ScrollReveal>
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
              <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2rem, 5vw, 3.5rem)', color: 'var(--color-white)', lineHeight: 1, letterSpacing: '-0.03em' }}>
                Let's make something.
              </h2>
              <div className="flex gap-3">
                <Link
                  to="/portfolio"
                  style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.78rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--color-muted)', border: '1px solid var(--color-border)', padding: '12px 20px', transition: 'all 0.15s', whiteSpace: 'nowrap' }}
                  onMouseEnter={e => { e.currentTarget.style.color = 'var(--color-white)'; e.currentTarget.style.borderColor = 'var(--color-muted-dark)' }}
                  onMouseLeave={e => { e.currentTarget.style.color = 'var(--color-muted)'; e.currentTarget.style.borderColor = 'var(--color-border)' }}
                >
                  View Work
                </Link>
                <Link
                  to="/contact"
                  style={{ fontFamily: 'var(--font-sans)', fontWeight: 700, fontSize: '0.78rem', letterSpacing: '0.1em', textTransform: 'uppercase', background: 'var(--color-accent)', color: 'var(--color-black)', padding: '12px 20px', transition: 'background 0.15s', whiteSpace: 'nowrap' }}
                  onMouseEnter={e => e.currentTarget.style.background = '#e04400'}
                  onMouseLeave={e => e.currentTarget.style.background = 'var(--color-accent)'}
                >
                  Get in Touch
                </Link>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </section>
    </main>
  )
}
