import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-top">
        <div>
          <div className="f-brand">Nuage</div>
          <p className="f-tagline">A private Alpine estate above the clouds. Available for exclusive rental to those who seek the extraordinary.</p>
        </div>

        <div className="f-col">
          <h4>The Estate</h4>
          <ul>
            <li><Link to="/estate">Overview</Link></li>
            <li><Link to="/estate#amenities">Amenities</Link></li>
            <li><Link to="/gallery">Gallery</Link></li>
            <li><Link to="/location">Location</Link></li>
          </ul>
        </div>

        <div className="f-col">
          <h4>Experiences</h4>
          <ul>
            <li><Link to="/experiences">Skiing</Link></li>
            <li><Link to="/experiences">Fine Dining</Link></li>
            <li><Link to="/experiences">Wellness</Link></li>
            <li><Link to="/experiences">Exploration</Link></li>
          </ul>
        </div>

        <div className="f-col">
          <h4>Contact</h4>
          <ul>
            <li><Link to="/reserve">Reserve</Link></li>
            <li><a href="mailto:reservations@nuage-estate.com">Email Us</a></li>
            <li><a href="tel:+41225550100">+41 22 555 0100</a></li>
            <li><a href="#press">Press Enquiries</a></li>
          </ul>
        </div>
      </div>

      <div className="footer-bottom">
        <p className="f-copy">© 2026 Nuage Estate. All rights reserved. Mattertal, Valais, Switzerland.</p>
        <div className="f-social">
          <a href="#ig" aria-label="Instagram">Instagram</a>
          <a href="#vi" aria-label="Vimeo">Vimeo</a>
          <a href="#pi" aria-label="Pinterest">Pinterest</a>
        </div>
      </div>
    </footer>
  )
}
