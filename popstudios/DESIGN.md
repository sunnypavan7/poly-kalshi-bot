# PopStudios — Design System
> Single source of truth for all UI, motion, and component decisions.

---

## 1. Brand Positioning

PopStudios is a **gallery-grade luxury photography studio**. The interface is a frame, never a distraction. Every design decision asks: *does this make the photography feel more elevated?*

Tone: editorial restraint, cinematic confidence, quiet authority.
Anti-patterns: centered hero templates, stock-looking layouts, decorative gradients, busy backgrounds.

---

## 2. Color System

### Rationale
Near-monochrome palette — maximum focus on photography. A single accent in **muted champagne gold** (`#C5A068`) is chosen over oxblood because it references darkroom chemistry, film edges, and luxury print materials without reading as aggressive. It's used only for micro-accents and hover states, never as a fill color.

```
--color-black:       #0A0A0A   /* primary background, text on light */
--color-surface:     #111111   /* card/overlay backgrounds */
--color-border:      #222222   /* subtle dividers */
--color-muted-dark:  #3A3A3A   /* secondary text on dark */
--color-muted:       #888888   /* captions, metadata */
--color-warm-white:  #F7F5F2   /* primary text on dark, light bg */
--color-off-white:   #EDEAE5   /* secondary light surface */
--color-accent:      #C5A068   /* champagne gold — use sparingly */
--color-accent-dim:  #8A6E44   /* gold on light backgrounds */
```

**Usage rules:**
- Dark-mode-first: backgrounds are `--color-black`, text is `--color-warm-white`
- Gold accent appears on: links on hover, active nav states, cursor trails, thin rule dividers, price figures
- Never use accent as a button fill — use white text on near-black, or outlined

---

## 3. Typography

### Rationale
**Cormorant Garamond** (display) is a Didone-inspired serif with extreme thick/thin contrast — the typographic equivalent of a large-format print. It's free, premium-feeling, and uncommon enough to avoid template associations.

**DM Sans** (body/UI) is a clean, optically-corrected grotesque. It's modern without feeling tech-startup. Pairs beautifully with Cormorant's drama.

### Scale (Major Third — 1.250)
```
--text-xs:    0.64rem   / 10.2px  — metadata, captions
--text-sm:    0.8rem    / 12.8px  — UI labels, tags
--text-base:  1rem      / 16px    — body copy
--text-lg:    1.25rem   / 20px    — lead text, large labels
--text-xl:    1.563rem  / 25px    — subheadings
--text-2xl:   1.953rem  / 31.2px  — section headings
--text-3xl:   2.441rem  / 39px    — page titles
--text-4xl:   3.052rem  / 48.8px  — hero sub-headings
--text-5xl:   3.815rem  / 61px    — hero display
--text-6xl:   6rem      / 96px    — wordmark / full-bleed display
```

### Font Weights & Usage
```
Cormorant Garamond
  300 (Light)     — hero wordmark, editorial display
  400 (Regular)   — large sub-headings
  600 (SemiBold)  — section titles

DM Sans
  300 (Light)     — large body/lead paragraphs
  400 (Regular)   — body copy
  500 (Medium)    — nav links, button labels, UI
  700 (Bold)      — rarely; labels, tags only
```

### Letter Spacing
```
--tracking-tight:  -0.03em   — large display type only
--tracking-normal:  0em
--tracking-wide:    0.08em   — all-caps UI labels, nav, tags
--tracking-wider:   0.15em   — wordmark
```

### Line Height
```
--leading-none:   1
--leading-tight:  1.1   — display headings
--leading-snug:   1.3   — subheadings
--leading-normal: 1.6   — body
--leading-loose:  1.9   — captions, fine print
```

---

## 4. Spacing & Layout

### Grid
- **Desktop**: 12-column, `max-width: 1440px`, 120px outer margin, 24px gutters
- **Tablet**: 8-column, 48px outer margin, 20px gutters
- **Mobile**: 4-column, 24px outer margin, 16px gutters

### Spacing Scale (4px base)
```
--space-1:    4px
--space-2:    8px
--space-3:    12px
--space-4:    16px
--space-5:    20px
--space-6:    24px
--space-8:    32px
--space-10:   40px
--space-12:   48px
--space-16:   64px
--space-20:   80px
--space-24:   96px
--space-32:   128px
--space-40:   160px
--space-48:   192px
```

### Layout Rules
- Section vertical padding: `--space-32` to `--space-48`
- Hero sections: 100vh minimum
- Text columns max-width: 680px (readability)
- Gallery items: no fixed max-width
- Nav height: 72px (desktop), 60px (mobile)
- Never center-align body text blocks

---

## 5. Motion & Animation

### Rationale
**Framer Motion** is chosen over GSAP for this project because:
1. React-native scroll-driven API (`useScroll`, `useTransform`)
2. `LazyMotion` for bundle splitting
3. First-class `prefers-reduced-motion` support via `useReducedMotion()`
4. Simpler maintenance for a component-driven codebase

GSAP would be preferred if we needed timeline-scrubbed scroll sequences at scale; for a portfolio site, Framer Motion is sufficient and lighter.

### Easing Curves
```
--ease-luxury:    cubic-bezier(0.25, 0.1, 0.0, 1.0)   /* slow in, abrupt stop — editorial */
--ease-reveal:    cubic-bezier(0.16, 1, 0.3, 1)        /* fast out, overshoot settle */
--ease-cinematic: cubic-bezier(0.76, 0, 0.24, 1)       /* symmetric, slow in/out */
```

### Duration Scale
```
--dur-instant:  80ms    — hover color flips
--dur-fast:     150ms   — nav state changes
--dur-normal:   300ms   — most UI transitions
--dur-slow:     600ms   — page element reveals
--dur-cinematic: 1200ms — hero transitions, page wipes
--dur-epic:     2000ms  — opening sequence only
```

### Motion Vocabulary

| Motion | Behavior | When |
|--------|----------|------|
| **Fade-up reveal** | `y: 40px → 0, opacity: 0 → 1` | All section content on scroll entry |
| **Stagger reveal** | Children stagger `100ms` apart | Grid items, nav links, lists |
| **Scale-in** | `scale: 0.96 → 1, opacity: 0 → 1` | Cards, images entering viewport |
| **Parallax drift** | Background `y` at `0.3×` scroll rate | Hero image layers |
| **Cinematic wipe** | Full-screen `--color-black` panel sweeps off | Page transitions |
| **Nav blur** | `backdrop-filter: blur(0 → 12px)` + bg opacity | Nav on scroll |
| **Cursor expand** | Custom cursor scales `1 → 2.5×` on image hover | Gallery images |
| **Gold line draw** | SVG stroke-dashoffset reveal | Dividers, decorative lines |

### Reduced-Motion Fallback
All animation values gate behind `useReducedMotion()`. When true:
- Parallax disabled
- Fade durations drop to `80ms`
- Page transitions become instant opacity cross-fades
- Video backgrounds show poster stills

---

## 6. Components

### Navigation
- Fixed position, full-width
- Default state: `background: transparent`, white text (for hero use)
- Scrolled state: `background: rgba(10,10,10,0.85)`, `backdrop-filter: blur(12px)`
- Logo left, links right (desktop); hamburger right (mobile)
- Mobile menu: full-screen dark overlay with staggered link entries
- Active link: thin gold underline (`--color-accent`)

### Hero
- `100vh`, full-bleed
- Layer order (back → front): video/image → dark scrim → content
- Scrim: `linear-gradient(to bottom, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0.5) 100%)`
- Video: `autoPlay muted loop playsInline`, `object-fit: cover`
- Poster: a high-quality still matching first video frame
- Wordmark: Cormorant Garamond 300, `--text-6xl`, `--tracking-wider`, white

### Gallery Grid
- CSS Grid with `auto-fill`, variable `minmax` per breakpoint
- Items: overflow hidden, `border-radius: 0` (clean, gallery-wall feel)
- Hover: `scale(1.03)` on inner image, `opacity` reveal of caption overlay
- Masonry variant: CSS columns or JS-measured absolute positioning
- Lightbox: full-screen, keyboard navigable (←/→/Esc), drag/swipe, counter

### Cards (Services/About)
- No border-radius (editorial) or max `4px` only
- Border: `1px solid --color-border`
- Background: `--color-surface`
- Hover: border color shifts to `--color-accent`

### Buttons
```
Primary:   bg transparent, border 1px --color-warm-white, text --color-warm-white
           Hover: bg --color-warm-white, text --color-black
           
Secondary: no border, text only with animated underline grow
           
CTA large: Cormorant Garamond 400, --text-xl, uppercase, --tracking-wide
```

### Form Elements
- Input: borderless except bottom border, `--color-muted-dark`
- Focus: bottom border shifts to `--color-accent`, no outline box
- Label: floats up on focus (animated)

---

## 7. Imagery Rules

- **All photography**: `object-fit: cover`, never stretch
- **Aspect ratios in galleries**: intentional variety — 3:2, 4:5, 1:1, 16:9 mixed
- **Loading**: blur-up placeholder (low-res base64 → full-res), lazy `loading="lazy"`
- **Alt text**: always descriptive, never "image" or filename
- **Placeholder source**: `images.unsplash.com` with specific photo IDs keyed to luxury/photography aesthetic
- **Replace marker**: Every placeholder has a comment `{/* REPLACE: description of shot */}`

---

## 8. Page Transitions

Implemented as a full-screen `--color-black` overlay positioned fixed, z-index 9999.

1. **Exit**: overlay slides in from bottom-right, covering the page (`dur-cinematic`)
2. **Route change**: fires after overlay covers screen
3. **Enter**: overlay slides out to top-left, revealing new page (`dur-cinematic`)

On first load: overlay fades out from full opacity over `dur-epic`.

---

## 9. Iconography

No icon libraries. SVG inline only. Set limited to:
- Hamburger / close (nav)
- Arrow right (CTAs)
- Arrow left/right (lightbox nav)
- X (lightbox close)
- Instagram, email (footer)

All strokes, 1.5px weight, white, no fills.

---

## 10. Accessibility

- All interactive elements: visible focus ring (`outline: 2px solid --color-accent, offset 2px`)
- Color contrast: warm-white on black = 16.1:1 (AAA). Gold on black = 4.9:1 (AA large text only — never used for body copy)
- Lightbox: traps focus, `role="dialog"`, `aria-modal`, Esc closes
- Video heroes: `aria-hidden="true"` (decorative), poster shown when autoplay blocked
- Reduced motion: `@media (prefers-reduced-motion: reduce)` CSS layer + JS hook
- Nav mobile: `aria-expanded`, `aria-controls`, focus trap in open state
