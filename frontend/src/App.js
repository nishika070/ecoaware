import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Home from "./pages/Home";
import AQI from "./pages/AQI";
import Temperature from "./pages/Temperature";
import Policies from "./pages/Policies";
import Contact from "./pages/Contact";
import Navbar from "./components/Navbar";

function App() {
  return (
    <Router>
      <Navbar />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/aqi" element={<AQI />} />
        <Route path="/temperature" element={<Temperature />} />
        <Route path="/policies" element={<Policies />} />
        <Route path="/contact" element={<Contact />} />
      </Routes>

    </Router>
  );
}

export default App;