# PopStudios — Luxury Photography Brand Website

A production-ready React/Vite website for a luxury photography studio. Built with Framer Motion animations, a full design system, and 6 fully built pages.

## Stack

- **React 19** + **Vite 8**
- **Tailwind CSS v4** (via `@tailwindcss/vite`)
- **Framer Motion** — scroll reveals, page transitions, parallax, lightbox
- **React Router v7** — client-side routing with animated page transitions
- **Google Fonts** — Cormorant Garamond (display) + DM Sans (body)

## Pages

| Route | Description |
|---|---|
| `/` | Home — cinematic hero, featured work, about teaser, services teaser, CTA |
| `/portfolio` | Filterable gallery grid (Editorial, Portrait, Wedding, Commercial, Travel) |
| `/portfolio/:id` | Single project case study with parallax images and lightbox |
| `/about` | Photographer bio, philosophy, press/clients |
| `/services` | Packages, pricing, add-ons, process |
| `/contact` | Inquiry form with animated success state |

## Running locally

```bash
cd popstudios
npm install
npm run dev
```

Open `http://localhost:5173`

## Production build

```bash
npm run build
npm run preview
```

## Replacing placeholder media

All placeholder images come from Unsplash and are marked with comments. Search for `REPLACE:` in the source:

```bash
grep -r "REPLACE:" src/
```

### Key replacements

| File | What to replace |
|---|---|
| `src/pages/Home.jsx` | Hero video (`heroVideo` src) and hero poster image |
| `src/pages/About.jsx` | Photographer portrait, studio workspace photo |
| `src/pages/Contact.jsx` | Contact page background image |
| `src/data/portfolio.js` | All `cover` and `images[].src` URLs |

For the hero video, drop a looping `.mp4` in `public/` and update the `<source src="">` in `Home.jsx`.

## Design system

See `DESIGN.md` for the full design system: color tokens, typography scale, motion vocabulary, and component specs.

All design tokens are CSS custom properties in `src/index.css` under `@theme {}`.

## Accessibility

- `prefers-reduced-motion` respected via `useReducedMotion()` (Framer Motion)
- Lightbox: `role="dialog"`, `aria-modal`, keyboard navigation (←/→/Esc)
- Video heroes: `aria-hidden="true"`, poster fallback
- Focus rings: `2px solid var(--color-accent)` on all interactive elements
- All images have descriptive alt text
