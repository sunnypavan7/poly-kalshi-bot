import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'

const projectTypes = ['Wedding', 'Portrait', 'Commercial', 'Events', 'Other']
const budgetRanges = ['Under ₹20,000', '₹20,000–50,000', '₹50,000–1,00,000', '₹1,00,000+', "Let's discuss"]

const inputStyle = (focused, field) => ({
  width: '100%',
  background: 'transparent',
  border: 'none',
  borderBottom: `1px solid ${focused === field ? 'var(--color-accent)' : 'var(--color-border)'}`,
  color: 'var(--color-white)',
  fontFamily: 'var(--font-sans)',
  fontWeight: 300,
  fontSize: '1rem',
  padding: '10px 4px 10px 0',
  outline: 'none',
  transition: 'border-color 0.2s',
})

export default function Contact() {
  const [form, setForm] = useState({ name: '', email: '', phone: '', projectType: '', budget: '', message: '', timeline: '' })
  const [submitted, setSubmitted] = useState(false)
  const [focused, setFocused] = useState(null)

  const set = (field) => (e) => setForm(f => ({ ...f, [field]: e.target.value }))
  const handleSubmit = (e) => { e.preventDefault(); setSubmitted(true) }

  return (
    <main>
      {/* Header */}
      <section className="relative overflow-hidden" style={{ paddingTop: 'clamp(100px, 12vw, 160px)', paddingBottom: 64, borderBottom: '1px solid var(--color-border)' }}>
        <img
          src="https://images.unsplash.com/photo-1469334031218-e382a71b716b?auto=format&fit=crop&w=1920&q=15"
          alt=""
          className="absolute inset-0 w-full h-full object-cover"
          style={{ opacity: 0.07 }}
          aria-hidden="true"
        />
        <div className="relative z-10 max-w-[1440px] mx-auto px-6 md:px-12">
          <ScrollReveal>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 500, fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-accent)', marginBottom: 14 }}>
              Contact
            </p>
            <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(3rem, 8vw, 6rem)', lineHeight: 0.9, letterSpacing: '-0.03em', color: 'var(--color-white)', marginBottom: 20 }}>
              Let's talk.
            </h1>
            <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '1rem', color: 'var(--color-muted)', maxWidth: 380, lineHeight: 1.7 }}>
              Tell us about your project. We read every inquiry personally and respond within 24 hours.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Form + info */}
      <section className="max-w-[1440px] mx-auto px-6 md:px-12" style={{ paddingTop: 'clamp(60px, 8vw, 100px)', paddingBottom: 'clamp(60px, 8vw, 100px)' }}>
        <div className="grid grid-cols-1 md:grid-cols-12 gap-16 md:gap-24">

          {/* Form */}
          <div className="md:col-span-7">
            <AnimatePresence mode="wait">
              {submitted ? (
                <motion.div
                  key="success"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] } }}
                  style={{ paddingTop: 48, paddingBottom: 48 }}
                >
                  <div style={{ width: 40, height: 2, background: 'var(--color-accent)', marginBottom: 28 }} aria-hidden="true" />
                  <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 'clamp(2rem, 4vw, 3rem)', color: 'var(--color-white)', marginBottom: 14, lineHeight: 1 }}>
                    Thank you, {form.name.split(' ')[0]}.
                  </h2>
                  <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'var(--color-muted)', lineHeight: 1.7, fontSize: '0.93rem' }}>
                    We've received your message and will be in touch within 24 hours.<br />
                    WhatsApp works too if you're in a hurry.
                  </p>
                </motion.div>
              ) : (
                <motion.form
                  key="form"
                  onSubmit={handleSubmit}
                  style={{ display: 'flex', flexDirection: 'column', gap: 36 }}
                  noValidate
                >
                  <ScrollReveal>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      {[
                        { id: 'name', label: 'Full Name *', type: 'text', placeholder: 'Your name', required: true, autoComplete: 'name' },
                        { id: 'email', label: 'Email *', type: 'email', placeholder: 'your@email.com', required: true, autoComplete: 'email' },
                      ].map(({ id, label, type, placeholder, required, autoComplete }) => (
                        <div key={id}>
                          <label style={{ display: 'block', fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 8 }} htmlFor={id}>
                            {label}
                          </label>
                          <input
                            id={id}
                            type={type}
                            required={required}
                            value={form[id]}
                            onChange={set(id)}
                            onFocus={() => setFocused(id)}
                            onBlur={() => setFocused(null)}
                            style={inputStyle(focused, id)}
                            placeholder={placeholder}
                            autoComplete={autoComplete}
                          />
                        </div>
                      ))}
                    </div>
                  </ScrollReveal>

                  <ScrollReveal delay={0.05}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div>
                        <label style={{ display: 'block', fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 8 }} htmlFor="phone">
                          Phone (WhatsApp)
                        </label>
                        <input
                          id="phone"
                          type="tel"
                          value={form.phone}
                          onChange={set('phone')}
                          onFocus={() => setFocused('phone')}
                          onBlur={() => setFocused(null)}
                          style={inputStyle(focused, 'phone')}
                          placeholder="+91 98765 43210"
                          autoComplete="tel"
                        />
                      </div>
                      <div>
                        <label style={{ display: 'block', fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 8 }} htmlFor="projectType">
                          Project Type
                        </label>
                        <select
                          id="projectType"
                          value={form.projectType}
                          onChange={set('projectType')}
                          onFocus={() => setFocused('projectType')}
                          onBlur={() => setFocused(null)}
                          style={{ ...inputStyle(focused, 'projectType'), background: 'var(--color-black)', cursor: 'pointer' }}
                        >
                          <option value="">Select type</option>
                          {projectTypes.map(t => <option key={t} value={t}>{t}</option>)}
                        </select>
                      </div>
                    </div>
                  </ScrollReveal>

                  <ScrollReveal delay={0.1}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div>
                        <label style={{ display: 'block', fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 8 }} htmlFor="budget">
                          Budget Range
                        </label>
                        <select
                          id="budget"
                          value={form.budget}
                          onChange={set('budget')}
                          onFocus={() => setFocused('budget')}
                          onBlur={() => setFocused(null)}
                          style={{ ...inputStyle(focused, 'budget'), background: 'var(--color-black)', cursor: 'pointer' }}
                        >
                          <option value="">Approximate budget</option>
                          {budgetRanges.map(b => <option key={b} value={b}>{b}</option>)}
                        </select>
                      </div>
                      <div>
                        <label style={{ display: 'block', fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 8 }} htmlFor="timeline">
                          Timeline / Date
                        </label>
                        <input
                          id="timeline"
                          type="text"
                          value={form.timeline}
                          onChange={set('timeline')}
                          onFocus={() => setFocused('timeline')}
                          onBlur={() => setFocused(null)}
                          style={inputStyle(focused, 'timeline')}
                          placeholder="e.g. Feb 2026 wedding, flexible"
                        />
                      </div>
                    </div>
                  </ScrollReveal>

                  <ScrollReveal delay={0.12}>
                    <div>
                      <label style={{ display: 'block', fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 8 }} htmlFor="message">
                        Tell us about the project *
                      </label>
                      <textarea
                        id="message"
                        required
                        rows={5}
                        value={form.message}
                        onChange={set('message')}
                        onFocus={() => setFocused('message')}
                        onBlur={() => setFocused(null)}
                        style={{ ...inputStyle(focused, 'message'), resize: 'none' }}
                        placeholder="What are you planning, and what matters most about how it's captured?"
                      />
                    </div>
                  </ScrollReveal>

                  <ScrollReveal delay={0.15}>
                    <button
                      type="submit"
                      style={{ display: 'inline-flex', alignItems: 'center', gap: 12, fontFamily: 'var(--font-sans)', fontWeight: 700, fontSize: '0.82rem', letterSpacing: '0.12em', textTransform: 'uppercase', background: 'var(--color-accent)', color: 'var(--color-black)', padding: '16px 32px', border: 'none', cursor: 'pointer', transition: 'background 0.2s' }}
                      onMouseEnter={e => e.currentTarget.style.background = '#e04400'}
                      onMouseLeave={e => e.currentTarget.style.background = 'var(--color-accent)'}
                    >
                      Send Message
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                      </svg>
                    </button>
                  </ScrollReveal>
                </motion.form>
              )}
            </AnimatePresence>
          </div>

          {/* Info sidebar */}
          <aside className="md:col-span-4 md:col-start-9">
            <ScrollReveal delay={0.2} direction="left">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
                {[
                  { label: 'Email', value: 'hello@popstudios.in', href: 'mailto:hello@popstudios.in' },
                  { label: 'WhatsApp', value: '+91 98765 43210', href: 'https://wa.me/919876543210' },
                  { label: 'Instagram', value: '@popstudios', href: 'https://instagram.com/popstudios' },
                ].map(({ label, value, href }) => (
                  <div key={label}>
                    <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.62rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 8 }}>
                      {label}
                    </p>
                    <a
                      href={href}
                      target={href.startsWith('http') ? '_blank' : undefined}
                      rel={href.startsWith('http') ? 'noopener noreferrer' : undefined}
                      style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, color: 'var(--color-white)', fontSize: '0.9rem', transition: 'color 0.15s' }}
                      onMouseEnter={e => e.currentTarget.style.color = 'var(--color-accent)'}
                      onMouseLeave={e => e.currentTarget.style.color = 'var(--color-white)'}
                    >
                      {value}
                    </a>
                  </div>
                ))}

                <div>
                  <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.62rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 8 }}>
                    Studio
                  </p>
                  <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.88rem', color: 'var(--color-muted)', lineHeight: 1.7 }}>
                    Bandra West, Mumbai<br />
                    Available pan-India
                  </p>
                </div>

                <div>
                  <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '0.62rem', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-muted-dark)', marginBottom: 8 }}>
                    Response Time
                  </p>
                  <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.88rem', color: 'var(--color-muted)' }}>
                    Within 24 hours
                  </p>
                </div>

                <div style={{ paddingTop: 20, borderTop: '1px solid var(--color-border)' }}>
                  <p style={{ fontFamily: 'var(--font-sans)', fontWeight: 300, fontSize: '0.78rem', color: 'var(--color-muted-dark)', lineHeight: 1.75, fontStyle: 'italic' }}>
                    We take 25–30 weddings per year and limited commercial projects. Inquiry early for peak season dates (Oct–Feb).
                  </p>
                </div>
              </div>
            </ScrollReveal>
          </aside>
        </div>
      </section>
    </main>
  )
}
