import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { packages, addOns } from '../data/services'

function PackageCard({ pkg, index }) {
  return (
    <ScrollReveal delay={index * 0.1}>
      <div className="border border-[var(--color-border)] p-8 md:p-10 group hover:border-[var(--color-accent)] transition-colors duration-300 flex flex-col h-full">
        <div className="mb-8">
          <p className="font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted-dark)] mb-4">
            0{index + 1}
          </p>
          <h2 className="font-display font-light text-3xl text-[var(--color-warm-white)] mb-2">
            {pkg.name}
          </h2>
          <p className="font-sans font-light text-sm text-[var(--color-muted)] leading-relaxed">
            {pkg.tagline}
          </p>
        </div>

        <div className="flex items-baseline gap-2 mb-8">
          <span className="font-display font-light text-3xl text-[var(--color-accent)]">
            {pkg.price}
          </span>
          <span className="font-sans text-xs text-[var(--color-muted)] tracking-[0.06em]">
            / {pkg.duration}
          </span>
        </div>

        <ul className="space-y-3 mb-8 flex-1">
          {pkg.includes.map((item) => (
            <li key={item} className="flex items-start gap-3">
              <span className="mt-1.5 w-3 h-px bg-[var(--color-accent)] shrink-0" aria-hidden="true" />
              <span className="font-sans font-light text-sm text-[var(--color-muted)] leading-relaxed">{item}</span>
            </li>
          ))}
        </ul>

        <p className="font-sans text-xs text-[var(--color-muted-dark)] leading-relaxed italic border-t border-[var(--color-border)] pt-6 mb-8">
          {pkg.note}
        </p>

        <Link
          to="/contact"
          className="font-sans text-sm font-medium tracking-[0.1em] uppercase text-[var(--color-warm-white)] border border-[var(--color-border)] group-hover:border-[var(--color-accent)] px-6 py-3 text-center hover:bg-[var(--color-accent)] hover:text-[var(--color-black)] hover:border-[var(--color-accent)] transition-all duration-300 block"
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
      <section className="pt-36 md:pt-44 pb-20 border-b border-[var(--color-border)]">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-end">
            <ScrollReveal>
              <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-4">
                Services
              </p>
              <h1 className="font-display font-light text-5xl md:text-7xl leading-none tracking-[-0.03em] text-[var(--color-warm-white)]">
                Packages &amp;<br />Pricing
              </h1>
            </ScrollReveal>
            <ScrollReveal delay={0.15} direction="left">
              <p className="font-sans font-light text-lg text-[var(--color-muted)] leading-relaxed max-w-md">
                All projects begin with a conversation. These packages are starting points; most commissions are tailored to the specific brief.
              </p>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* Packages */}
      <section className="max-w-[1440px] mx-auto px-8 md:px-16 py-24 md:py-36">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-1 md:gap-4">
          {packages.map((pkg, i) => (
            <PackageCard key={pkg.id} pkg={pkg} index={i} />
          ))}
        </div>
      </section>

      {/* Add-ons */}
      <section className="border-t border-[var(--color-border)] py-20 md:py-28">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <ScrollReveal>
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-4">
              Add-ons
            </p>
            <h2 className="font-display font-light text-4xl text-[var(--color-warm-white)] leading-tight mb-14 md:mb-16">
              Optional extras
            </h2>
          </ScrollReveal>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-[var(--color-border)]">
            {addOns.map((item, i) => (
              <ScrollReveal key={item.name} delay={i * 0.08}>
                <div className="bg-[var(--color-black)] p-8 hover:bg-[var(--color-surface)] transition-colors duration-300">
                  <h3 className="font-display font-light text-xl text-[var(--color-warm-white)] mb-2">
                    {item.name}
                  </h3>
                  <p className="font-sans font-light text-sm text-[var(--color-muted)] leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Process */}
      <section className="border-t border-[var(--color-border)] py-20 md:py-28">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16">
          <ScrollReveal>
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-4">
              Process
            </p>
            <h2 className="font-display font-light text-4xl text-[var(--color-warm-white)] leading-tight mb-14 md:mb-16">
              How it works
            </h2>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 md:gap-12">
            {[
              { n: '01', title: 'Inquiry', body: 'Fill in the contact form. We\'ll respond within 48 hours to arrange an initial conversation.' },
              { n: '02', title: 'Brief', body: 'We spend time understanding the project, the light conditions, and what a successful outcome looks like.' },
              { n: '03', title: 'Shoot', body: 'The production itself. We work methodically, leaving room for the unexpected.' },
              { n: '04', title: 'Delivery', body: 'Edited finals in your agreed formats, within the agreed timeline. No rush fees, no surprises.' },
            ].map((step, i) => (
              <ScrollReveal key={step.n} delay={i * 0.1}>
                <div>
                  <p className="font-display text-5xl font-light text-[var(--color-border)] mb-6">{step.n}</p>
                  <h3 className="font-display font-light text-xl text-[var(--color-warm-white)] mb-3">{step.title}</h3>
                  <p className="font-sans font-light text-sm text-[var(--color-muted)] leading-relaxed">{step.body}</p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-[var(--color-border)] py-24 md:py-32">
        <div className="max-w-[1440px] mx-auto px-8 md:px-16 text-center">
          <ScrollReveal>
            <h2 className="font-display font-light text-4xl md:text-6xl text-[var(--color-warm-white)] leading-tight tracking-[-0.02em] mb-8">
              Ready to begin?
            </h2>
            <p className="font-sans font-light text-[var(--color-muted)] mb-10 max-w-md mx-auto">
              Limited availability. We take fewer commissions than we could.
            </p>
            <Link
              to="/contact"
              className="inline-flex items-center gap-3 font-sans font-medium text-sm tracking-[0.1em] uppercase bg-[var(--color-accent)] text-[var(--color-black)] px-8 py-4 hover:bg-[var(--color-warm-white)] transition-colors duration-300"
            >
              Start a Conversation
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </ScrollReveal>
        </div>
      </section>
    </main>
  )
}
