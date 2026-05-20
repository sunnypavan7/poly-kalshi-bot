import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import ScrollReveal from '../components/ScrollReveal'
import { ROOMS, AMENITIES } from '../data/estate'

function Icon({ type }) {
  const d = {
    pool:    <path d="M3 12h18M12 3c0 5-3 8-3 8s3 1 3 9m0-17c0 5 3 8 3 8s-3 1-3 9"/>,
    spa:     <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>,
    cinema:  <><rect x="2" y="7" width="20" height="15" rx="2"/><polyline points="17,2 12,7 7,2"/></>,
    wine:    <><line x1="12" y1="2" x2="12" y2="22"/><path d="M5 5a7 7 0 0 0 7 7 7 7 0 0 0 7-7"/></>,
    team:    <><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></>,
    heli:    <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>,
    ski:     <><path d="M3 17l4-4 4 4 4-8 4 4"/><path d="M3 21h18"/></>,
    kitchen: <><path d="M3 3h18v4H3z"/><path d="M3 7v14"/><path d="M21 7v14"/><path d="M3 14h18"/></>,
    library: <><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></>,
  }
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      {d[type]}
    </svg>
  )
}

export default function Estate() {
  return (
    <main>
      {/* Hero */}
      <div className="page-hero">
        <div className="page-hero-bg" style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1613977257592-4871e5fcd7c4?w=1800&q=82&auto=format)' }} />
        <div className="page-hero-scrim" />
        <div className="page-hero-cnt">
          <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.16,1,0.3,1], delay: 0.3 }}>The Residence</motion.p>
          <motion.h1 initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, ease: [0.16,1,0.3,1], delay: 0.45 }}>
            The <em>Estate</em>
          </motion.h1>
        </div>
      </div>

      {/* Intro */}
      <section className="section-pad">
        <div className="section-inner" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80, alignItems: 'center' }}>
          <ScrollReveal>
            <p className="eyebrow">About the Property</p>
            <h2 className="section-title" style={{ marginTop: 12 }}>Architecture<br />& <em>Interiors</em></h2>
            <div className="gold-rule" />
          </ScrollReveal>
          <ScrollReveal delay={0.15}>
            <p style={{ fontSize: '0.93rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)', marginBottom: 18 }}>
              Conceived by Pritzker laureate Jean-Marc Pellerin, the main house is a dialogue between raw Alpine stone and floor-to-ceiling glass. The structure was built to last centuries, yet every detail speaks to contemporary ease.
            </p>
            <p style={{ fontSize: '0.93rem', fontWeight: 300, lineHeight: 1.85, color: 'var(--muted)' }}>
              Curated over two decades by Maison Artefact in Brussels, the interiors blend museum-quality art with handwoven textiles, reclaimed Alpine timber, and bespoke furniture. Eight suites, each its own universe.
            </p>
          </ScrollReveal>
        </div>
      </section>

      {/* Alternating estate features */}
      <section className="estate">
        {[
          { num: '01', title: 'The Architecture', body: 'The main house responds to the mountain with a 2,400m vocabulary — massive stone walls that absorb the Alpine cold, then vast glass planes that surrender to panorama. Pellerin\'s masterstroke is the floating roof terrace: 320m² of heated decking hovering above the valley.', img: 'https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=1200&q=82&auto=format', alt: 'Estate exterior' },
          { num: '02', title: 'The Interiors',    body: 'Every material is local or traceable. The reclaimed fir comes from a sawmill eight kilometres north. The stone is Evolène limestone. The textiles are woven by a fourth-generation family in Herisau. Art is museum-loaned and rotated seasonally.', img: 'https://images.unsplash.com/photo-1615529328331-f8917597711f?w=1200&q=82&auto=format', alt: 'Master suite' },
          { num: '03', title: 'The Pool & Spa',   body: 'The 22-metre infinity pool is fed by an Alpine spring and heated to 30°C year-round. The cedar spa pavilion below houses a hammam, cold plunge, two treatment rooms, and a sauna carved from a single century-old fir.', img: 'https://images.unsplash.com/photo-1571896349842-33c89424de2d?w=1200&q=82&auto=format', alt: 'Infinity pool' },
          { num: '04', title: 'The Grounds',      body: 'Twelve hectares run from the formal gardens to a stand of larch at the treeline. Two groomed ski runs descend from the estate\'s upper boundary. In summer they become wildflower meadows. The helipad sits discretely behind the north wing.', img: 'https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=1200&q=82&auto=format', alt: 'Alpine grounds' },
        ].map((item, i) => (
          <div className="feat-pair" key={item.num}>
            <ScrollReveal direction="scale" className="feat-img">
              <img src={item.img} alt={item.alt} loading="lazy" />
            </ScrollReveal>
            <div className="feat-body">
              <ScrollReveal><div className="feat-num">{item.num}</div></ScrollReveal>
              <ScrollReveal delay={0.1}><h3>{item.title}</h3></ScrollReveal>
              <ScrollReveal delay={0.2}><p>{item.body}</p></ScrollReveal>
            </div>
          </div>
        ))}
      </section>

      {/* Rooms */}
      <section className="section-pad bg-warm">
        <div className="section-inner">
          <ScrollReveal className="section-hd">
            <p className="eyebrow">Accommodation</p>
            <h2 className="section-title">Eight <em>suites</em>,<br />one world.</h2>
            <div className="gold-rule" />
          </ScrollReveal>
          <div className="room-grid" style={{ marginTop: 64 }}>
            {ROOMS.map((r, i) => (
              <ScrollReveal key={r.name} delay={i * 0.08} direction="scale" className="room-card">
                <img src={r.img} alt={`${r.name} at Nuage Estate`} loading="lazy" />
                <div className="room-info">
                  <div className="room-meta">{r.floor} &nbsp;·&nbsp; {r.size} &nbsp;·&nbsp; {r.guests} guests</div>
                  <div className="room-name">{r.name}</div>
                  <div className="room-desc">{r.desc}</div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Amenities */}
      <section className="section-pad bg-dark" id="amenities">
        <div className="section-inner">
          <ScrollReveal className="section-hd">
            <p className="eyebrow">Facilities</p>
            <h2 className="section-title on-dark">Every <em>luxury</em>,<br />considered.</h2>
            <div className="gold-rule" />
          </ScrollReveal>
          <div className="amen-grid" style={{ marginTop: 64 }}>
            {AMENITIES.map((a, i) => (
              <ScrollReveal key={a.name} delay={(i % 3) * 0.07} className="amen-card">
                <div className="amen-ico"><Icon type={a.icon} /></div>
                <div className="amen-name">{a.name}</div>
                <div className="amen-desc">{a.desc}</div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-pad" style={{ borderTop: '1px solid var(--stone)', textAlign: 'center' }}>
        <ScrollReveal>
          <p className="eyebrow">Ready?</p>
          <h2 className="section-title" style={{ marginTop: 12, marginBottom: 28 }}>Reserve the <em>Estate</em></h2>
          <Link to="/reserve" className="btn-pri dark-fill">Begin Enquiry</Link>
        </ScrollReveal>
      </section>
    </main>
  )
}
