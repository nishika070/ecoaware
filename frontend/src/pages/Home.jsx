import { useEffect, useState } from "react";

import DashboardCard from "../components/DashboardCard";
import AQIChart from "../components/AQIChart";
import AdvisoryPanel from "../components/AdvisoryPanel";

import "../styles/pages/Home.css";

function Home({ selectedLocation, setSelectedLocation, locations }) {

  const [temperature, setTemperature] = useState(null);
  const [aqi, setAqi] = useState(null);
  const [pm25, setPm25] = useState(null);

  useEffect(() => {

    async function fetchData() {

      if (!selectedLocation) return;

      try {

        const weatherRes = await fetch(
          `https://api.open-meteo.com/v1/forecast?latitude=${selectedLocation.lat}&longitude=${selectedLocation.lon}&current_weather=true`
        );

        const weatherData = await weatherRes.json();
        setTemperature(weatherData?.current_weather?.temperature);

        const pollutionRes = await fetch(
          `https://api.openweathermap.org/data/2.5/air_pollution?lat=${selectedLocation.lat}&lon=${selectedLocation.lon}&appid=YOUR_API_KEY`
        );

        const pollutionData = await pollutionRes.json();

        setAqi(pollutionData?.list?.[0]?.main?.aqi);
        setPm25(pollutionData?.list?.[0]?.components?.pm2_5);

      } catch (error) {
        console.error("API error:", error);
      }

    }

    fetchData();

    const interval = setInterval(fetchData, 600000);

    return () => clearInterval(interval);

  }, [selectedLocation]);

  return (

    <main className="page-shell">

      <div className="container home-page">

        <section className="hero panel">

          <p className="chip chip-warning">
            Delhi pilot region | Live environmental intelligence
          </p>

          <h1>
            One platform for AQI, temperature, trend analysis, and health guidance
          </h1>

        </section>

        <div className="location-selector">

          <label>Select Delhi Area:</label>

          <select
            value={selectedLocation?.name}
            onChange={(e) => {
              const loc = locations.find(l => l.name === e.target.value);
              setSelectedLocation(loc);
            }}
          >

            {locations.map((loc) => (
              <option key={loc.name} value={loc.name}>
                {loc.name}
              </option>
            ))}

          </select>

        </div>

        <section className="grid grid-4 metrics-grid">

          <DashboardCard
            title={`AQI (${selectedLocation?.name})`}
            value={aqi ? aqi : "Loading"}
            note="Live AQI data"
          />

          <DashboardCard
            title="Temperature"
            value={temperature ? `${temperature} °C` : "Loading"}
            note="Live weather data"
          />

          <DashboardCard
            title="PM2.5"
            value={pm25 ? `${pm25} µg/m³` : "Loading"}
            note="Fine particulate concentration"
          />

          <DashboardCard
            title="Heat Risk"
            value={temperature && temperature > 35 ? "High" : "Moderate"}
            note="Temperature threshold based"
          />

        </section>

        <section className="grid grid-2 section-split">

          <div className="panel chart-panel">
            <h2>Weekly AQI and temperature movement</h2>
            <AQIChart />
          </div>

          <AdvisoryPanel />

        </section>

      </div>

    </main>

  );
}

export default Home;