import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

const LINKS = [
  { label: 'The Estate',   to: '/estate'      },
  { label: 'Gallery',      to: '/gallery'     },
  { label: 'Location',     to: '/location'    },
  { label: 'Experiences',  to: '/experiences' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [mobOpen, setMobOpen]   = useState(false)
  const location = useLocation()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => { setMobOpen(false) }, [location.pathname])

  const logoColor  = scrolled ? 'var(--dark)'  : 'white'
  const linkColor  = scrolled ? 'var(--brown)' : 'rgba(255,255,255,0.82)'
  const hamColor   = scrolled ? 'var(--dark)'  : 'white'

  return (
    <>
      <header className={`nav${scrolled ? ' scrolled' : ''}`}>
        <Link to="/" style={{ fontFamily: 'var(--font-d)', fontWeight: 300, fontSize: '1.55rem', letterSpacing: '0.22em', textTransform: 'uppercase', color: logoColor, transition: 'color 0.45s' }}>
          Nuage
        </Link>

        {/* Desktop links */}
        <nav className="nav-links" style={{ display: 'flex', gap: 40, listStyle: 'none', alignItems: 'center' }}>
          {LINKS.map(l => (
            <li key={l.to}>
              <Link
                to={l.to}
                className={`nav-link${location.pathname === l.to ? ' active' : ''}`}
                style={{ color: linkColor }}
              >
                {l.label}
              </Link>
            </li>
          ))}
        </nav>

        <Link to="/reserve" className={`btn-pri ${scrolled ? 'outline' : 'ghost-light'}`} style={{ fontSize: '0.68rem' }} aria-label="Reserve the estate">
          Reserve
        </Link>

        {/* Hamburger */}
        <button
          className="nav-ham"
          onClick={() => setMobOpen(o => !o)}
          aria-label="Toggle menu"
          aria-expanded={mobOpen}
          style={{ display: 'none' }}
        >
          <motion.span animate={{ rotate: mobOpen ? 45 : 0, y: mobOpen ? 7 : 0, background: hamColor }} transition={{ duration: 0.28 }} />
          <motion.span animate={{ opacity: mobOpen ? 0 : 1, background: hamColor }} transition={{ duration: 0.28 }} />
          <motion.span animate={{ rotate: mobOpen ? -45 : 0, y: mobOpen ? -7 : 0, background: hamColor }} transition={{ duration: 0.28 }} />
        </button>
      </header>

      {/* Mobile overlay menu */}
      <AnimatePresence>
        {mobOpen && (
          <motion.div
            className="mobile-menu"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.25, 0.1, 0.0, 1.0] }}
          >
            <motion.ul
              initial="hidden"
              animate="show"
              variants={{ show: { transition: { staggerChildren: 0.07 } } }}
            >
              {[...LINKS, { label: 'Reserve', to: '/reserve' }].map((l, i) => (
                <motion.li
                  key={l.to}
                  variants={{ hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } } }}
                >
                  <Link to={l.to} className="mobile-menu-link">{l.label}</Link>
                </motion.li>
              ))}
            </motion.ul>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
