import "../styles/pages/AQI.css";

const stationRows = [
  { station: "Anand Vihar", aqi: 254, pm25: 167, pm10: 246, status: "Very Poor" },
  { station: "RK Puram", aqi: 188, pm25: 114, pm10: 210, status: "Unhealthy" },
  { station: "Punjabi Bagh", aqi: 176, pm25: 102, pm10: 198, status: "Unhealthy" },
  { station: "Lodhi Road", aqi: 132, pm25: 78, pm10: 144, status: "Moderate" }
];

function AQI() {
  return (
    <main className="page-shell">
      <div className="container aqi-page">
        <section className="page-heading">
          <h1>Delhi AQI Intelligence</h1>
          <p>
            Station-wise AQI, PM2.5, PM10 indicators and priority zones for rapid intervention.
          </p>
        </section>

        <section className="panel">
          <table>
            <thead>
              <tr>
                <th>Station</th>
                <th>AQI</th>
                <th>PM2.5 (ug/m3)</th>
                <th>PM10 (ug/m3)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {stationRows.map((row) => (
                <tr key={row.station}>
                  <td>{row.station}</td>
                  <td>{row.aqi}</td>
                  <td>{row.pm25}</td>
                  <td>{row.pm10}</td>
                  <td>{row.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="panel alert-strip">
          <h2>Policy trigger suggestion</h2>
          <p>
            Anand Vihar and RK Puram belt mein strict dust suppression aur heavy-vehicle timing restrictions consider karna chahiye.
          </p>
        </section>
      </div>
    </main>
  );
}

export default AQI;
