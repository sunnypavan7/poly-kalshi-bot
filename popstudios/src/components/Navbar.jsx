import { useState, useEffect } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

const links = [
  { to: '/portfolio', label: 'Portfolio' },
  { to: '/about', label: 'About' },
  { to: '/services', label: 'Services' },
  { to: '/contact', label: 'Contact' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    setMenuOpen(false)
  }, [location.pathname])

  useEffect(() => {
    document.body.style.overflow = menuOpen ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [menuOpen])

  return (
    <>
      <motion.header
        className="fixed top-0 left-0 right-0 z-50 h-[72px] flex items-center px-8 md:px-12 transition-all"
        animate={{
          backgroundColor: scrolled ? 'rgba(10,10,10,0.88)' : 'rgba(10,10,10,0)',
          backdropFilter: scrolled ? 'blur(12px)' : 'blur(0px)',
        }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
      >
        <div className="w-full flex items-center justify-between">
          <Link
            to="/"
            className="font-display font-light tracking-[0.15em] text-[var(--color-warm-white)] text-lg uppercase hover:text-[var(--color-accent)] transition-colors duration-300"
            aria-label="PopStudios — home"
          >
            Pop<span className="text-[var(--color-accent)]">Studios</span>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-10" aria-label="Main navigation">
            {links.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `font-sans text-sm font-medium tracking-[0.08em] uppercase transition-colors duration-200 relative group ${
                    isActive ? 'text-[var(--color-accent)]' : 'text-[var(--color-warm-white)] hover:text-[var(--color-accent)]'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    {label}
                    <span
                      className="absolute -bottom-1 left-0 h-px bg-[var(--color-accent)] transition-all duration-300"
                      style={{ width: isActive ? '100%' : '0%' }}
                    />
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          {/* Mobile menu button */}
          <button
            onClick={() => setMenuOpen(v => !v)}
            className="md:hidden flex flex-col gap-[5px] w-6 h-5 justify-center items-end"
            aria-expanded={menuOpen}
            aria-controls="mobile-menu"
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          >
            <motion.span
              className="block h-px bg-[var(--color-warm-white)] w-full origin-center"
              animate={menuOpen ? { rotate: 45, y: 6 } : { rotate: 0, y: 0 }}
              transition={{ duration: 0.25 }}
            />
            <motion.span
              className="block h-px bg-[var(--color-warm-white)] w-4"
              animate={menuOpen ? { opacity: 0, scaleX: 0 } : { opacity: 1, scaleX: 1 }}
              transition={{ duration: 0.2 }}
            />
            <motion.span
              className="block h-px bg-[var(--color-warm-white)] w-full origin-center"
              animate={menuOpen ? { rotate: -45, y: -6 } : { rotate: 0, y: 0 }}
              transition={{ duration: 0.25 }}
            />
          </button>
        </div>
      </motion.header>

      {/* Mobile full-screen menu */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            id="mobile-menu"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } }}
            exit={{ opacity: 0, y: -20, transition: { duration: 0.3, ease: 'easeIn' } }}
            className="fixed inset-0 z-40 bg-[var(--color-black)] flex flex-col justify-center px-8"
            role="dialog"
            aria-modal="true"
          >
            <nav className="flex flex-col gap-8" aria-label="Mobile navigation">
              {links.map(({ to, label }, i) => (
                <motion.div
                  key={to}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0, transition: { delay: i * 0.08 + 0.1, duration: 0.4, ease: [0.16, 1, 0.3, 1] } }}
                >
                  <NavLink
                    to={to}
                    className={({ isActive }) =>
                      `font-display font-light text-5xl leading-none tracking-tight hover:text-[var(--color-accent)] transition-colors duration-200 ${
                        isActive ? 'text-[var(--color-accent)]' : 'text-[var(--color-warm-white)]'
                      }`
                    }
                  >
                    {label}
                  </NavLink>
                </motion.div>
              ))}
            </nav>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, transition: { delay: 0.5, duration: 0.4 } }}
              className="absolute bottom-12 left-8 right-8 flex justify-between items-end"
            >
              <a
                href="https://instagram.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs tracking-[0.12em] uppercase text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors"
              >
                Instagram
              </a>
              <a
                href="mailto:hello@popstudios.com"
                className="text-xs tracking-[0.12em] uppercase text-[var(--color-muted)] hover:text-[var(--color-warm-white)] transition-colors"
              >
                hello@popstudios.com
              </a>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
