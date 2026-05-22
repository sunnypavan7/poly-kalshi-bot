import { Link } from 'react-router-dom'
import ScrollReveal from './ScrollReveal'

export default function Footer() {
  const year = new Date().getFullYear()

  return (
    <footer style={{ borderTop: '1px solid var(--color-border)', background: 'var(--color-black)' }}>
      <div className="max-w-[1440px] mx-auto px-6 md:px-12 py-16 md:py-20">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 md:gap-8">

          <ScrollReveal className="md:col-span-2">
            <Link
              to="/"
              className="block mb-5"
              style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.5rem', letterSpacing: '-0.01em', textTransform: 'uppercase', color: 'var(--color-white)' }}
              aria-label="PopStudios home"
            >
              Pop<span style={{ color: 'var(--color-accent)' }}>Studios</span>
            </Link>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.9rem', lineHeight: 1.7, color: 'var(--color-muted)', maxWidth: 280 }}>
              Mumbai-based photography studio.<br />
              Weddings, portraits, commercial &amp; events.<br />
              Available across India.
            </p>
            <div className="flex gap-5 mt-6">
              <a href="https://instagram.com/popstudios" target="_blank" rel="noopener noreferrer"
                style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-muted)' }}
                className="hover:text-[var(--color-white)] transition-colors"
              >Instagram</a>
              <a href="https://wa.me/919876543210" target="_blank" rel="noopener noreferrer"
                style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-muted)' }}
                className="hover:text-[var(--color-white)] transition-colors"
              >WhatsApp</a>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={0.1}>
            <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 20 }}>
              Explore
            </p>
            <nav className="flex flex-col gap-3">
              {[
                { to: '/portfolio', label: 'Portfolio' },
                { to: '/services', label: 'Services' },
                { to: '/about', label: 'About' },
                { to: '/contact', label: 'Contact' },
              ].map(({ to, label }) => (
                <Link
                  key={to}
                  to={to}
                  style={{ fontFamily: 'var(--font-sans)', fontSize: '0.85rem', color: 'var(--color-muted)' }}
                  className="hover:text-[var(--color-white)] transition-colors duration-150"
                >
                  {label}
                </Link>
              ))}
            </nav>
          </ScrollReveal>

          <ScrollReveal delay={0.2}>
            <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 20 }}>
              Contact
            </p>
            <div className="flex flex-col gap-3">
              <a href="mailto:hello@popstudios.in"
                style={{ fontFamily: 'var(--font-sans)', fontSize: '0.85rem', color: 'var(--color-muted)' }}
                className="hover:text-[var(--color-white)] transition-colors"
              >
                hello@popstudios.in
              </a>
              <a href="tel:+919876543210"
                style={{ fontFamily: 'var(--font-sans)', fontSize: '0.85rem', color: 'var(--color-muted)' }}
                className="hover:text-[var(--color-white)] transition-colors"
              >
                +91 98765 43210
              </a>
              <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.82rem', color: 'var(--color-muted)', lineHeight: 1.6, marginTop: 4 }}>
                Bandra West, Mumbai<br />
                Available pan-India
              </p>
            </div>
          </ScrollReveal>

        </div>

        <div className="mt-16 pt-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-3"
          style={{ borderTop: '1px solid var(--color-border)' }}>
          <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-muted-dark)', letterSpacing: '0.04em' }}>
            © {year} PopStudios. All rights reserved.
          </p>
          <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-muted-dark)', letterSpacing: '0.04em' }}>
            Mumbai, India
          </p>
        </div>
      </div>
    </footer>
  )
}
