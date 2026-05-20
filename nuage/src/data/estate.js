export const GALLERY = [
  { id: 1,  src: 'https://images.unsplash.com/photo-1566073771259-6a8506099945?w=1400&q=80&auto=format', alt: 'Aerial view of the Nuage Estate and infinity pool',      label: 'Aerial View',    cat: 'Exterior' },
  { id: 2,  src: 'https://images.unsplash.com/photo-1613977257592-4871e5fcd7c4?w=1000&q=80&auto=format', alt: 'Main residence with heated pool',                        label: 'The Residence',  cat: 'Exterior' },
  { id: 3,  src: 'https://images.unsplash.com/photo-1615529328331-f8917597711f?w=1000&q=80&auto=format', alt: 'Master bedroom suite with mountain views',               label: 'Master Suite',   cat: 'Interior' },
  { id: 4,  src: 'https://images.unsplash.com/photo-1571896349842-33c89424de2d?w=1000&q=80&auto=format', alt: 'Heated infinity pool at golden hour',                   label: 'Infinity Pool',  cat: 'Pool'     },
  { id: 5,  src: 'https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=1000&q=80&auto=format', alt: 'Great room with open fireplace',                        label: 'Great Room',     cat: 'Interior' },
  { id: 6,  src: 'https://images.unsplash.com/photo-1604014237800-1c9102c219da?w=1000&q=80&auto=format', alt: 'South terrace overlooking the valley',                  label: 'The Terrace',    cat: 'Exterior' },
  { id: 7,  src: 'https://images.unsplash.com/photo-1578683010236-d716f9a3f461?w=1000&q=80&auto=format', alt: 'Private dining room with candlelight',                  label: 'Dining Room',    cat: 'Interior' },
  { id: 8,  src: 'https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=1000&q=80&auto=format', alt: 'Alpine peaks at sunrise above the estate',              label: 'Mountain Views', cat: 'Views'    },
  { id: 9,  src: 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=1000&q=80&auto=format', alt: 'Winter summit views from the terrace',                  label: 'Winter Peaks',   cat: 'Views'    },
  { id: 10, src: 'https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?w=1000&q=80&auto=format', alt: 'Calacatta marble bathroom with soaking tub',            label: 'Marble Bath',    cat: 'Interior' },
  { id: 11, src: 'https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=1000&q=80&auto=format', alt: 'Pool terrace at dusk',                                  label: 'Pool at Dusk',   cat: 'Pool'     },
  { id: 12, src: 'https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=1000&q=80&auto=format', alt: 'Estate exterior illuminated at night',                      label: 'Night View',     cat: 'Exterior' },
]

export const GALLERY_CATS = ['All', 'Exterior', 'Interior', 'Pool', 'Views']

export const ROOMS = [
  {
    name: 'Cloud Suite',
    size: '120 m²', guests: 2, floor: 'Third Floor',
    desc: 'The flagship suite occupies the entire top floor. 270° glazing frames the Matterhorn by day; at night, the sky belongs entirely to you. Private sauna, terrace with outdoor soaking tub.',
    img:  'https://images.unsplash.com/photo-1615529328331-f8917597711f?w=900&q=80&auto=format',
  },
  {
    name: 'Alpine Suite I',
    size: '88 m²', guests: 2, floor: 'Second Floor',
    desc: 'East-facing, in reclaimed Alpine fir and polished concrete. King bed, en-suite hammam, and direct garden access. Watch the sun rise over the ridge from your private loggia.',
    img:  'https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?w=900&q=80&auto=format',
  },
  {
    name: 'Alpine Suite II',
    size: '82 m²', guests: 2, floor: 'Second Floor',
    desc: 'West-facing — catching last light as it sweeps the valley floor. Hand-loomed linen, Bulthaup kitchenette, and a deep Japanese soaking tub set in Pietra Serena stone.',
    img:  'https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=900&q=80&auto=format',
  },
  {
    name: 'Valley Suite',
    size: '95 m²', guests: 4, floor: 'Ground Floor',
    desc: 'A connected double suite — bedroom plus sitting room, shared south terrace, and two full en-suites in Calacatta marble. Ideal for families or travelling companions.',
    img:  'https://images.unsplash.com/photo-1578683010236-d716f9a3f461?w=900&q=80&auto=format',
  },
]

export const AMENITIES = [
  { name: 'Infinity Pool',    desc: 'Heated year-round, 22 metres. Uninterrupted views across three valleys.',           icon: 'pool'    },
  { name: 'Spa & Wellness',   desc: 'Hammam, two treatment rooms, cold plunge, and Alpine fir sauna.',                   icon: 'spa'     },
  { name: 'Private Cinema',   desc: '12-seat room, 4K laser projection, Dolby Atmos, and a 3,000-title library.',        icon: 'cinema'  },
  { name: 'Wine Cellar',      desc: '2,200 bottles — Grand Cru Burgundy, Premier Cru Bordeaux, rare Swiss domaines.',    icon: 'wine'    },
  { name: 'Estate Team',      desc: 'Private chef, two sous chefs, house manager, and a 24-hour butler on call.',        icon: 'team'    },
  { name: 'Private Helipad',  desc: 'Geneva in 35 minutes. Full transfer coordination arranged on request.',             icon: 'heli'    },
  { name: 'Ski Access',       desc: 'Two groomed private runs + direct access to 360 km of Zermatt pistes.',             icon: 'ski'     },
  { name: "Chef's Kitchen",   desc: 'Full Gaggenau professional suite. Private cooking classes available.',               icon: 'kitchen' },
  { name: 'Library',          desc: '800-volume curated collection with a rare Alpine cartography archive.',              icon: 'library' },
]

export const EXPERIENCES = [
  { tag: 'Winter',        title: 'Private Skiing',    desc: 'Begin each morning on two groomed private runs before the valley wakes. Your dedicated ski guide and full equipment fitting is arranged on arrival.', img: 'https://images.unsplash.com/photo-1605540436563-5bca919ae766?w=800&q=80&auto=format' },
  { tag: 'Year-Round',    title: 'Alpine Trails',     desc: 'The estate intersects four heritage hiking routes. Guided expeditions from 2 to 12 hours, tailored entirely to your group\'s pace and ambition.', img: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80&auto=format' },
  { tag: 'Daily',         title: 'Private Dining',    desc: 'Three chefs rotate through seasonal tasting menus, wood-fired Alpine dishes, and on-request à la carte service — indoors, on the terrace, or at altitude.', img: 'https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=800&q=80&auto=format' },
  { tag: 'Nightly',       title: 'Stargazing',        desc: 'At 2,400m with zero light pollution, the Milky Way is a nightly spectacle. The estate\'s Swarovski telescope and resident astronomer complete the experience.', img: 'https://images.unsplash.com/photo-1419242902214-272b3f66ee7a?w=800&q=80&auto=format' },
  { tag: 'On Request',    title: 'Helicopter Tours',  desc: 'Circumnavigate the Monte Rosa massif or transfer to Monaco for dinner. The estate\'s dedicated EC135 is available with 24 hours notice.', img: 'https://images.unsplash.com/photo-1524168272322-bf73616d9cb5?w=800&q=80&auto=format' },
  { tag: 'Spring–Autumn', title: 'Alpine Spa Day',    desc: 'A full-day programme combining guided foraging, outdoor hydrotherapy, bespoke treatments, and a botanically-sourced menu prepared by the chef.', img: 'https://images.unsplash.com/photo-1544161515-4ab6ce6db874?w=800&q=80&auto=format' },
]

export const TESTIMONIALS = [
  { quote: 'Nuage redefined what a private estate can be. The view at dawn — clouds below, the Matterhorn above — is an image I will carry for the rest of my life.', author: 'The Whitmore Family', origin: 'London, United Kingdom' },
  { quote: 'Every member of the team anticipated needs we hadn\'t yet thought of. The level of service was, quite simply, incomparable to anywhere I have ever stayed.', author: 'Antoine Lefebvre', origin: 'Paris, France' },
  { quote: 'We\'ve stayed at the world\'s finest properties. Nuage is something else entirely — elemental, quiet, perfect. The silence alone is worth the journey.', author: 'Mr. & Mrs. Vandenberg', origin: 'New York, United States' },
  { quote: 'Our chef prepared a different tasting menu each evening. By the third night we had forgotten the rest of the world entirely. That is rare.', author: 'Isabelle Chen', origin: 'Hong Kong' },
]

export const LOCATION_FACTS = [
  '35 min by private helicopter from Geneva International',
  '8 min to Zermatt village via private cable car',
  'Direct access to 360 km of ski pistes',
  '3 Michelin-starred restaurants within 15 minutes',
  'Private snow groomer for two on-estate ski runs',
  '55 min drive to Visp railway station',
]
