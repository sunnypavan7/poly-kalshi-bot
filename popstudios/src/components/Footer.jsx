import { Link } from 'react-router-dom'
import ScrollReveal from './ScrollReveal'

export default function Footer() {
  const year = new Date().getFullYear()

  return (
    <footer className="border-t border-[var(--color-border)] bg-[var(--color-black)]">
      <div className="max-w-[1440px] mx-auto px-8 md:px-16 py-16 md:py-20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 md:gap-8">
          <ScrollReveal>
            <Link
              to="/"
              className="font-display font-light text-2xl tracking-[0.15em] uppercase text-[var(--color-warm-white)] hover:text-[var(--color-accent)] transition-colors duration-300 block mb-4"
            >
              Pop<span className="text-[var(--color-accent)]">Studios</span>
            </Link>
            <p className="text-[var(--color-muted)] text-sm leading-relaxed max-w-xs">
              Gallery-grade photography.<br />
              London &amp; worldwide.
            </p>
          </ScrollReveal>

          <ScrollReveal delay={0.1}>
            <p className="text-xs tracking-[0.12em] uppercase text-[var(--color-muted-dark)] mb-5">Navigation</p>
            <nav className="flex flex-col gap-3">
              {[
                { to: '/portfolio', label: 'Portfolio' },
                { to: '/about', label: 'About' },
                { to: '/services', label: 'Services' },
                { to: '/contact', label: 'Contact' },
              ].map(({ to, label }) => (
                <Link
                  key={to}
                  to={to}
                  className="text-sm text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors duration-200"
                >
                  {label}
                </Link>
              ))}
            </nav>
          </ScrollReveal>

          <ScrollReveal delay={0.2}>
            <p className="text-xs tracking-[0.12em] uppercase text-[var(--color-muted-dark)] mb-5">Contact</p>
            <div className="flex flex-col gap-3">
              <a
                href="mailto:hello@popstudios.com"
                className="text-sm text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors duration-200"
              >
                hello@popstudios.com
              </a>
              <a
                href="https://instagram.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors duration-200"
              >
                @popstudios
              </a>
            </div>
          </ScrollReveal>
        </div>

        <div className="mt-16 pt-8 border-t border-[var(--color-border)] flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <p className="text-xs text-[var(--color-muted)] tracking-[0.06em]">
            © {year} PopStudios. All rights reserved.
          </p>
          <p className="text-xs text-[var(--color-muted-dark)] tracking-[0.06em]">
            London, UK
          </p>
        </div>
      </div>
    </footer>
  )
}
