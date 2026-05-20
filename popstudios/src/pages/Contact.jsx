import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'

const projectTypes = ['Editorial', 'Portrait', 'Wedding', 'Commercial', 'Travel', 'Other']
const budgetRanges = ['Under £2,000', '£2,000–5,000', '£5,000–10,000', '£10,000+', 'Let\'s discuss']

export default function Contact() {
  const [form, setForm] = useState({
    name: '', email: '', projectType: '', budget: '', message: '', timeline: '',
  })
  const [submitted, setSubmitted] = useState(false)
  const [focused, setFocused] = useState(null)

  const set = (field) => (e) => setForm(f => ({ ...f, [field]: e.target.value }))

  const handleSubmit = (e) => {
    e.preventDefault()
    setSubmitted(true)
  }

  const inputClass = (field) => `
    w-full bg-transparent border-0 border-b text-[var(--color-warm-white)] font-sans font-light text-base py-3 pr-2
    placeholder:text-[var(--color-muted-dark)] transition-colors duration-200 outline-none
    ${focused === field ? 'border-[var(--color-accent)]' : 'border-[var(--color-border)] hover:border-[var(--color-muted-dark)]'}
  `

  return (
    <main>
      {/* Header */}
      <section className="relative pt-36 md:pt-44 pb-20 overflow-hidden border-b border-[var(--color-border)]">
        {/* {/* REPLACE: subtle background image for contact page */}
        <img
          src="https://images.unsplash.com/photo-1554048612-b6a482bc67e5?auto=format&fit=crop&w=1920&q=20"
          alt=""
          className="absolute inset-0 w-full h-full object-cover opacity-10"
          aria-hidden="true"
          loading="eager"
        />
        <div className="relative z-10 max-w-[1440px] mx-auto px-8 md:px-16">
          <ScrollReveal>
            <p className="font-sans text-xs tracking-[0.2em] uppercase text-[var(--color-accent)] mb-4">
              Contact
            </p>
            <h1 className="font-display font-light text-5xl md:text-7xl leading-none tracking-[-0.03em] text-[var(--color-warm-white)] mb-6">
              Start a<br />conversation.
            </h1>
            <p className="font-sans font-light text-lg text-[var(--color-muted)] max-w-md">
              Tell us about your project. We read every inquiry personally and respond within 48 hours.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Form + info */}
      <section className="max-w-[1440px] mx-auto px-8 md:px-16 py-24 md:py-36">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-16 md:gap-24">

          {/* Form */}
          <div className="md:col-span-7">
            <AnimatePresence mode="wait">
              {submitted ? (
                <motion.div
                  key="success"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] } }}
                  className="py-20"
                >
                  <div className="w-12 h-px bg-[var(--color-accent)] mb-8" aria-hidden="true" />
                  <h2 className="font-display font-light text-4xl text-[var(--color-warm-white)] mb-4">
                    Thank you, {form.name.split(' ')[0]}.
                  </h2>
                  <p className="font-sans font-light text-[var(--color-muted)] leading-relaxed mb-2">
                    We've received your inquiry.
                  </p>
                  <p className="font-sans font-light text-[var(--color-muted)] leading-relaxed">
                    We'll be in touch within 48 hours.
                  </p>
                </motion.div>
              ) : (
                <motion.form
                  key="form"
                  onSubmit={handleSubmit}
                  className="space-y-10"
                  noValidate
                >
                  <ScrollReveal>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div>
                        <label className="block font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted)] mb-2" htmlFor="name">
                          Name *
                        </label>
                        <input
                          id="name"
                          type="text"
                          required
                          value={form.name}
                          onChange={set('name')}
                          onFocus={() => setFocused('name')}
                          onBlur={() => setFocused(null)}
                          className={inputClass('name')}
                          placeholder="Your full name"
                          autoComplete="name"
                        />
                      </div>
                      <div>
                        <label className="block font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted)] mb-2" htmlFor="email">
                          Email *
                        </label>
                        <input
                          id="email"
                          type="email"
                          required
                          value={form.email}
                          onChange={set('email')}
                          onFocus={() => setFocused('email')}
                          onBlur={() => setFocused(null)}
                          className={inputClass('email')}
                          placeholder="your@email.com"
                          autoComplete="email"
                        />
                      </div>
                    </div>
                  </ScrollReveal>

                  <ScrollReveal delay={0.05}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div>
                        <label className="block font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted)] mb-2" htmlFor="projectType">
                          Project Type
                        </label>
                        <select
                          id="projectType"
                          value={form.projectType}
                          onChange={set('projectType')}
                          onFocus={() => setFocused('projectType')}
                          onBlur={() => setFocused(null)}
                          className={inputClass('projectType') + ' cursor-pointer'}
                          style={{ background: 'var(--color-black)' }}
                        >
                          <option value="">Select a type</option>
                          {projectTypes.map(t => <option key={t} value={t}>{t}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="block font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted)] mb-2" htmlFor="budget">
                          Budget Range
                        </label>
                        <select
                          id="budget"
                          value={form.budget}
                          onChange={set('budget')}
                          onFocus={() => setFocused('budget')}
                          onBlur={() => setFocused(null)}
                          className={inputClass('budget') + ' cursor-pointer'}
                          style={{ background: 'var(--color-black)' }}
                        >
                          <option value="">Approximate budget</option>
                          {budgetRanges.map(b => <option key={b} value={b}>{b}</option>)}
                        </select>
                      </div>
                    </div>
                  </ScrollReveal>

                  <ScrollReveal delay={0.1}>
                    <div>
                      <label className="block font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted)] mb-2" htmlFor="timeline">
                        Timeline
                      </label>
                      <input
                        id="timeline"
                        type="text"
                        value={form.timeline}
                        onChange={set('timeline')}
                        onFocus={() => setFocused('timeline')}
                        onBlur={() => setFocused(null)}
                        className={inputClass('timeline')}
                        placeholder="e.g. September 2025, flexible, ASAP"
                      />
                    </div>
                  </ScrollReveal>

                  <ScrollReveal delay={0.12}>
                    <div>
                      <label className="block font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted)] mb-2" htmlFor="message">
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
                        className={inputClass('message') + ' resize-none'}
                        placeholder="What are you making, and why does it matter?"
                      />
                    </div>
                  </ScrollReveal>

                  <ScrollReveal delay={0.15}>
                    <button
                      type="submit"
                      className="font-sans font-medium text-sm tracking-[0.1em] uppercase bg-[var(--color-warm-white)] text-[var(--color-black)] px-10 py-4 hover:bg-[var(--color-accent)] transition-colors duration-300 flex items-center gap-3"
                    >
                      Send Inquiry
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                      </svg>
                    </button>
                  </ScrollReveal>
                </motion.form>
              )}
            </AnimatePresence>
          </div>

          {/* Studio info */}
          <aside className="md:col-span-4 md:col-start-9">
            <ScrollReveal delay={0.2} direction="left">
              <div className="space-y-10">
                <div>
                  <p className="font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted-dark)] mb-3">
                    Email
                  </p>
                  <a
                    href="mailto:hello@popstudios.com"
                    className="font-sans font-light text-[var(--color-warm-white)] hover:text-[var(--color-accent)] transition-colors"
                  >
                    hello@popstudios.com
                  </a>
                </div>
                <div>
                  <p className="font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted-dark)] mb-3">
                    Instagram
                  </p>
                  <a
                    href="https://instagram.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans font-light text-[var(--color-warm-white)] hover:text-[var(--color-accent)] transition-colors"
                  >
                    @popstudios
                  </a>
                </div>
                <div>
                  <p className="font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted-dark)] mb-3">
                    Studio
                  </p>
                  <p className="font-sans font-light text-sm text-[var(--color-muted)] leading-relaxed">
                    Hackney, London<br />
                    Available worldwide
                  </p>
                </div>
                <div>
                  <p className="font-sans text-[10px] tracking-[0.2em] uppercase text-[var(--color-muted-dark)] mb-3">
                    Response time
                  </p>
                  <p className="font-sans font-light text-sm text-[var(--color-muted)]">
                    Within 48 hours
                  </p>
                </div>

                <div className="pt-6 border-t border-[var(--color-border)]">
                  <p className="font-sans font-light text-xs text-[var(--color-muted-dark)] leading-relaxed italic">
                    We take on 20–30 commissions per year. If the project resonates, we make it work.
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
