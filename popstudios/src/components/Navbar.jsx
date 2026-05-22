import { useState, useEffect } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

const links = [
  { to: '/portfolio', label: 'Portfolio' },
  { to: '/services', label: 'Services' },
  { to: '/about', label: 'About' },
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

  useEffect(() => { setMenuOpen(false) }, [location.pathname])

  useEffect(() => {
    document.body.style.overflow = menuOpen ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [menuOpen])

  return (
    <>
      <motion.header
        className="fixed top-0 left-0 right-0 z-50 h-[68px] flex items-center px-6 md:px-12"
        animate={{
          backgroundColor: scrolled ? 'rgba(8,8,8,0.92)' : 'rgba(8,8,8,0)',
          borderBottomColor: scrolled ? 'rgba(30,30,30,0.8)' : 'rgba(30,30,30,0)',
          borderBottomWidth: '1px',
          backdropFilter: scrolled ? 'blur(16px)' : 'blur(0px)',
        }}
        transition={{ duration: 0.25, ease: 'easeOut' }}
      >
        <div className="w-full flex items-center justify-between">
          <Link
            to="/"
            className="font-display font-700 tracking-tight text-xl uppercase text-[var(--color-white)] hover:text-[var(--color-accent)] transition-colors duration-200"
            style={{ fontFamily: 'var(--font-display)', fontWeight: 700 }}
            aria-label="PopStudios — home"
          >
            Pop<span style={{ color: 'var(--color-accent)' }}>Studios</span>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-8" aria-label="Main navigation">
            {links.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `font-sans text-[0.78rem] font-500 tracking-[0.1em] uppercase transition-colors duration-150 ${
                    isActive ? 'text-[var(--color-accent)]' : 'text-[var(--color-muted)] hover:text-[var(--color-white)]'
                  }`
                }
                style={{ fontFamily: 'var(--font-sans)', fontWeight: 500 }}
              >
                {label}
              </NavLink>
            ))}
            <Link
              to="/contact"
              className="font-sans text-[0.72rem] font-600 tracking-[0.12em] uppercase px-5 py-2.5 border border-[var(--color-accent)] text-[var(--color-accent)] hover:bg-[var(--color-accent)] hover:text-[var(--color-black)] transition-all duration-200"
              style={{ fontFamily: 'var(--font-sans)', fontWeight: 600 }}
            >
              Book Now
            </Link>
          </nav>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMenuOpen(v => !v)}
            className="md:hidden flex flex-col gap-[5px] w-6 h-5 justify-center items-end"
            aria-expanded={menuOpen}
            aria-controls="mobile-menu"
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          >
            <motion.span
              className="block h-px bg-[var(--color-white)] w-full origin-center"
              animate={menuOpen ? { rotate: 45, y: 6 } : { rotate: 0, y: 0 }}
              transition={{ duration: 0.22 }}
            />
            <motion.span
              className="block h-px bg-[var(--color-white)] w-3.5"
              animate={menuOpen ? { opacity: 0, scaleX: 0 } : { opacity: 1, scaleX: 1 }}
              transition={{ duration: 0.18 }}
            />
            <motion.span
              className="block h-px bg-[var(--color-white)] w-full origin-center"
              animate={menuOpen ? { rotate: -45, y: -6 } : { rotate: 0, y: 0 }}
              transition={{ duration: 0.22 }}
            />
          </button>
        </div>
      </motion.header>

      {/* Mobile menu */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            id="mobile-menu"
            initial={{ opacity: 0, clipPath: 'inset(0 0 100% 0)' }}
            animate={{ opacity: 1, clipPath: 'inset(0 0 0% 0)', transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] } }}
            exit={{ opacity: 0, clipPath: 'inset(0 0 100% 0)', transition: { duration: 0.3, ease: [0.76, 0, 0.24, 1] } }}
            className="fixed inset-0 z-40 bg-[var(--color-black)] flex flex-col justify-center px-8"
            role="dialog"
            aria-modal="true"
          >
            <nav className="flex flex-col gap-6" aria-label="Mobile navigation">
              {links.map(({ to, label }, i) => (
                <motion.div
                  key={to}
                  initial={{ opacity: 0, x: -24 }}
                  animate={{ opacity: 1, x: 0, transition: { delay: i * 0.07 + 0.12, duration: 0.4, ease: [0.16, 1, 0.3, 1] } }}
                >
                  <NavLink
                    to={to}
                    className={({ isActive }) =>
                      `block font-display font-700 text-5xl leading-none tracking-tight transition-colors duration-150 ${
                        isActive ? 'text-[var(--color-accent)]' : 'text-[var(--color-white)] hover:text-[var(--color-accent)]'
                      }`
                    }
                    style={{ fontFamily: 'var(--font-display)', fontWeight: 700 }}
                  >
                    {label}
                  </NavLink>
                </motion.div>
              ))}
            </nav>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, transition: { delay: 0.48, duration: 0.4 } }}
              className="absolute bottom-12 left-8 right-8 flex justify-between items-end"
            >
              <a
                href="https://instagram.com/popstudios"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs tracking-[0.12em] uppercase text-[var(--color-muted)] hover:text-[var(--color-white)] transition-colors"
                style={{ fontFamily: 'var(--font-sans)' }}
              >
                @popstudios
              </a>
              <a
                href="mailto:hello@popstudios.in"
                className="text-xs tracking-[0.12em] uppercase text-[var(--color-muted)] hover:text-[var(--color-white)] transition-colors"
                style={{ fontFamily: 'var(--font-sans)' }}
              >
                hello@popstudios.in
              </a>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
