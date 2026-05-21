import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Cursor from './components/Cursor'
import Loader from './components/Loader'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Estate from './pages/Estate'
import Gallery from './pages/Gallery'
import Experiences from './pages/Experiences'
import Location from './pages/Location'
import Reserve from './pages/Reserve'

function AnimatedRoutes() {
  const location = useLocation()
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/"            element={<Home />}        />
        <Route path="/estate"      element={<Estate />}      />
        <Route path="/gallery"     element={<Gallery />}     />
        <Route path="/experiences" element={<Experiences />} />
        <Route path="/location"    element={<Location />}    />
        <Route path="/reserve"     element={<Reserve />}     />
        <Route path="*"            element={<Home />}        />
      </Routes>
    </AnimatePresence>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Cursor />
      <Loader />
      <Navbar />
      <AnimatedRoutes />
      <Footer />
    </BrowserRouter>
  )
}
