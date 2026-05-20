import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'

const press = [
  'Vogue Italia', 'AnOther Magazine', 'British Journal of Photography',
  'National Geographic', 'Wallpaper*', 'i-D',
]

const clients = [
  'Zaha Hadid Architects', 'Aesop', 'Dior Beauty', 'NET-A-PORTER',
  'The Guardian', 'Christie\'s', 'Soho House',
]

const philosophy = [
  {
    heading: 'Light first',
    body: 'Every decision — location, timing, equipment — begins with a conversation about light. We don\'t manufacture it; we find it.',
  },
  {
    heading: 'Unhurried process',
    body: 'We take fewer commissions than we could. This lets us give each project the time it deserves. Good photography is patient.',
  },
  {
    heading: 'Documentary instinct',
    body: 'Even in controlled studio work, we\'re searching for the unguarded moment. The best images look like they couldn\'t have been planned.',
  },
]

export default function About() {
  return (
    <main>
      {/* Header */}
      <section className="pt-36 md:pt-44 pb-20 border-b border-[var(--color-border)]">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-end">
            <ScrollReveal>
              <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-4">
                About
              </p>
              <h1 className="font-display font-light text-5xl md:text-7xl leading-none tracking-[-0.03em] text-[var(--color-warm-white)]">
                The Studio
              </h1>
            </ScrollReveal>
            <ScrollReveal delay={0.15} direction="left">
              <p className="font-sans font-light text-lg text-[var(--color-muted)] leading-relaxed">
                PopStudios is a photography studio founded in London. We work at the intersection of documentary photography and fine art — unhurried, collaborative, and obsessively attentive to light.
              </p>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* Portrait + bio */}
      <section className="max-w-[1440px] mx-auto px-8 md:px-16 py-24 md:py-36">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12 md:gap-16 items-start">
          <ScrollReveal className="md:col-span-5">
            {/* {/* REPLACE: lead photographer portrait */}
            <div className="aspect-[3/4] overflow-hidden bg-[var(--color-surface)]">
              <img
                src="https://images.unsplash.com/photo-1554151228-14d9def656e4?auto=format&fit=crop&w=900&q=80"
                alt="Lead photographer in natural window light"
                className="w-full h-full object-cover"
                loading="eager"
              />
            </div>
          </ScrollReveal>

          <ScrollReveal delay={0.15} direction="left" className="md:col-span-6 md:col-start-7 md:pt-12">
            <h2 className="font-display font-light text-3xl md:text-4xl text-[var(--color-warm-white)] leading-tight mb-8">
              Alex PopCameron,<br />
              <em className="text-[var(--color-muted)]">Founder &amp; Lead Photographer</em>
            </h2>
            <div className="space-y-5 font-sans font-light text-[var(--color-muted)] leading-relaxed">
              <p>
                Alex began working as a photographer after a decade in documentary filmmaking. The transition brought a filmmaker's instinct for narrative to still images — a way of thinking about time, sequence, and what lies between decisive moments.
              </p>
              <p>
                Based between London and wherever the work requires, Alex has photographed on six continents and in every lighting condition that nature and architecture can produce. The throughline is always the same: find the light.
              </p>
              <p>
                PopStudios takes on between 20 and 30 commissions per year — editorial, commercial, portrait, and wedding — chosen for the quality of the collaboration as much as the brief.
              </p>
            </div>

            <div className="mt-10 flex gap-6">
              <a
                href="https://instagram.com"
                target="_blank"
                rel="noopener noreferrer"
                className="font-sans text-xs tracking-[0.14em] uppercase text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors"
              >
                Instagram
              </a>
              <a
                href="mailto:hello@popstudios.com"
                className="font-sans text-xs tracking-[0.14em] uppercase text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors"
              >
                Email
              </a>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* Philosophy */}
      <section className="border-t border-[var(--color-border)] py-24 md:py-36">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <ScrollReveal>
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-4">
              Philosophy
            </p>
            <h2 className="font-display font-light text-4xl md:text-5xl text-[var(--color-warm-white)] leading-tight tracking-[-0.02em] mb-16 md:mb-20">
              How we work
            </h2>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 md:gap-16">
            {philosophy.map((item, i) => (
              <ScrollReveal key={item.heading} delay={i * 0.1}>
                <div>
                  <p className="font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted-dark)] mb-4">
                    0{i + 1}
                  </p>
                  <h3 className="font-display font-light text-2xl text-[var(--color-warm-white)] mb-4">
                    {item.heading}
                  </h3>
                  <p className="font-sans font-light text-sm text-[var(--color-muted)] leading-relaxed">
                    {item.body}
                  </p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Studio image */}
      <section className="max-w-[1440px] mx-auto px-8 md:px-16 pb-24 md:pb-36">
        <ScrollReveal>
          {/* {/* REPLACE: studio/workspace photograph */}
          <div className="aspect-[16/7] overflow-hidden bg-[var(--color-surface)]">
            <img
              src="https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1920&q=80"
              alt="Studio workspace — large format printing tables and light boards"
              className="w-full h-full object-cover"
              loading="lazy"
            />
          </div>
        </ScrollReveal>
      </section>

      {/* Press */}
      <section className="border-t border-[var(--color-border)] py-20 md:py-28">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <ScrollReveal>
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-12">
              Press &amp; Publications
            </p>
          </ScrollReveal>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8 mb-20">
            {press.map((name, i) => (
              <ScrollReveal key={name} delay={i * 0.07}>
                <p className="font-display font-light text-lg text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors duration-200 cursor-default">
                  {name}
                </p>
              </ScrollReveal>
            ))}
          </div>

          <div className="border-t border-[var(--color-border)] pt-16">
            <ScrollReveal>
              <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-12">
                Selected Clients
              </p>
            </ScrollReveal>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-x-8 gap-y-4">
              {clients.map((name, i) => (
                <ScrollReveal key={name} delay={i * 0.05}>
                  <p className="font-sans font-light text-sm text-[var(--color-muted)]">{name}</p>
                </ScrollReveal>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-[var(--color-border)] py-24 md:py-32">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <ScrollReveal>
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
              <h2 className="font-display font-light text-4xl md:text-5xl text-[var(--color-warm-white)] leading-tight tracking-[-0.02em]">
                Ready to begin?
              </h2>
              <div className="flex gap-4">
                <Link
                  to="/portfolio"
                  className="font-sans text-sm font-medium tracking-[0.1em] uppercase text-[var(--color-muted)] border border-[var(--color-border)] px-6 py-3 hover:text-[var(--color-warm-white)] hover:border-[var(--color-muted-dark)] transition-colors duration-200"
                >
                  View Work
                </Link>
                <Link
                  to="/contact"
                  className="font-sans text-sm font-medium tracking-[0.1em] uppercase bg-[var(--color-warm-white)] text-[var(--color-black)] px-6 py-3 hover:bg-[var(--color-accent)] transition-colors duration-200"
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
