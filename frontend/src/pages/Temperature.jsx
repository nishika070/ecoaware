import "../styles/pages/Temperature.css";
import { useEffect, useState } from "react";
import { getWeather } from "../services/weatherService";

function Temperature({ selectedLocation }) {

  const [temperature, setTemperature] = useState(null);
  const [wind, setWind] = useState(null);

  useEffect(() => {

    async function fetchWeather(){

      if (!selectedLocation) return;

      const data = await getWeather(
        selectedLocation.lat,
        selectedLocation.lon
      );

      setTemperature(data.temperature);
      setWind(data.windspeed);

    }

    fetchWeather();

    const interval = setInterval(fetchWeather, 600000);

    return () => clearInterval(interval);

  }, [selectedLocation]);

  return (

    <main className="page-shell">

      <div className="container temp-page">

        <h1>Temperature & Heat Stress Analysis</h1>

        <h2>Current conditions for {selectedLocation?.name}</h2>

        <p className="highlight">

          {temperature
            ? `${temperature} °C with wind ${wind} km/h`
            : "Loading weather data..."}

        </p>

      </div>

    </main>

  );
}

export default Temperature;