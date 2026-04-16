import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState } from "react";

import Home from "./pages/Home";
import AQI from "./pages/AQI";
import Temperature from "./pages/Temperature";
import Policies from "./pages/Policies";
import Contact from "./pages/Contact";
import Navbar from "./components/Navbar";

const locations = [
  { name: "Central Delhi", lat: 28.6139, lon: 77.2090 },
  { name: "Rohini", lat: 28.7495, lon: 77.0565 },
  { name: "Dwarka", lat: 28.5921, lon: 77.0460 },
  { name: "Saket", lat: 28.5245, lon: 77.2066 },
  { name: "Anand Vihar", lat: 28.6460, lon: 77.3150 }
];

function App() {

  const [selectedLocation, setSelectedLocation] = useState(locations[0]);

  return (
    <Router>

      <Navbar />

      <Routes>

        <Route
          path="/"
          element={
            <Home
              selectedLocation={selectedLocation}
              setSelectedLocation={setSelectedLocation}
              locations={locations}
            />
          }
        />

        <Route
          path="/temperature"
          element={
            <Temperature
              selectedLocation={selectedLocation}
            />
          }
        />

        <Route path="/aqi" element={<AQI />} />
        <Route path="/policies" element={<Policies />} />
        <Route path="/contact" element={<Contact />} />

      </Routes>

    </Router>
  );
}

export default App;