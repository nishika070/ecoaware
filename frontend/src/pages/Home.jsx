import DashboardCard from "../components/DashboardCard";
import AQIChart from "../components/AQIChart";
import AdvisoryPanel from "../components/AdvisoryPanel";

import "../styles/pages/Home.css";

const kpiCards = [
  { title: "Current AQI", value: "182", note: "Delhi average | Unhealthy" },
  { title: "Temperature", value: "34 C", note: "Feels like 37 C" },
  { title: "PM2.5", value: "118 ug/m3", note: "3x above safe level" },
  { title: "Heat Risk", value: "High", note: "2 PM to 5 PM window" }
];

const drivingFactors = [
  {
    factor: "Construction Dust",
    impact: "+21% PM10 spike",
    action: "Dust barriers and strict site watering"
  },
  {
    factor: "Industrial Cluster",
    impact: "+16% NO2 concentration",
    action: "Targeted emission audits + temporary curbs"
  },
  {
    factor: "Vehicle Congestion",
    impact: "+12% roadside AQI",
    action: "Odd-even style controls in severe windows"
  },
  {
    factor: "Green Cover Loss",
    impact: "Higher local heat island",
    action: "Ward-level afforestation targets"
  }
];

const trendRows = [
  {
    window: "1986-1995",
    temp: "29.2 C",
    aqi: "Baseline low",
    note: "Lower urban pressure"
  },
  {
    window: "1996-2005",
    temp: "30.0 C",
    aqi: "Moderate rise",
    note: "Industrial growth acceleration"
  },
  {
    window: "2006-2015",
    temp: "31.1 C",
    aqi: "High variability",
    note: "Rapid construction and traffic"
  },
  {
    window: "2016-2025",
    temp: "32.4 C",
    aqi: "Frequently unhealthy",
    note: "Persistent PM2.5 episodes"
  }
];

function Home() {
  return (
    <main className="page-shell">
      <div className="container home-page">
        <section className="hero panel">
          <p className="chip chip-warning">Delhi pilot region | Live environmental intelligence</p>
          <h1>One platform for AQI, temperature, trend analysis, and health guidance</h1>
          <p>
            EcoAware combines 40-year climate history with current AQI and pollution feeds to predict risk, explain causes,
            and recommend public-health plus policy actions.
          </p>
        </section>

        <section className="grid grid-4 metrics-grid">
          {kpiCards.map((item) => (
            <DashboardCard key={item.title} title={item.title} value={item.value} note={item.note} />
          ))}
        </section>

        <section className="grid grid-2 section-split">
          <div className="panel chart-panel">
            <h2>Weekly AQI and temperature movement</h2>
            <AQIChart />
          </div>
          <AdvisoryPanel />
        </section>

        <section className="panel">
          <div className="section-head">
            <h2>40-year trend checkpoints</h2>
            <span className="chip chip-safe">Historic + recent data view</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Period</th>
                <th>Avg Temperature</th>
                <th>AQI Pattern</th>
                <th>Observation</th>
              </tr>
            </thead>
            <tbody>
              {trendRows.map((row) => (
                <tr key={row.window}>
                  <td>{row.window}</td>
                  <td>{row.temp}</td>
                  <td>{row.aqi}</td>
                  <td>{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="panel">
          <div className="section-head">
            <h2>Likely source factors for policy action</h2>
            <span className="chip chip-danger">Action-oriented diagnostics</span>
          </div>
          <div className="grid grid-2 factor-grid">
            {drivingFactors.map((item) => (
              <article key={item.factor} className="factor-card">
                <h3>{item.factor}</h3>
                <p><strong>Impact:</strong> {item.impact}</p>
                <p><strong>Suggested action:</strong> {item.action}</p>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

export default Home;
