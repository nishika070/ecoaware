import "../styles/pages/Temperature.css";

const seasonalOutlook = [
  { period: "Mar-Apr", avg: "32 C", risk: "Moderate" },
  { period: "May-Jun", avg: "39 C", risk: "Very High" },
  { period: "Jul-Aug", avg: "34 C", risk: "High humidity stress" },
  { period: "Sep-Oct", avg: "33 C", risk: "Moderate-High" }
];

function Temperature() {
  return (
    <main className="page-shell">
      <div className="container temp-page">
        <section className="page-heading">
          <h1>Temperature & Heat Stress Analysis</h1>
          <p>
            Historical warming trend and short-term heat alerts with area-level recommendations for Delhi.
          </p>
        </section>

        <section className="panel grid grid-2">
          <div>
            <h2>Current condition</h2>
            <p className="highlight">34 C with a 37 C feels-like temperature and elevated dehydration risk.</p>
          </div>
          <div>
            <h2>Recommended response</h2>
            <p className="highlight">Public hydration points, shaded waiting zones, and midday outdoor work limits.</p>
          </div>
        </section>

        <section className="panel">
          <h2>Seasonal heat outlook</h2>
          <table>
            <thead>
              <tr>
                <th>Period</th>
                <th>Avg Temp</th>
                <th>Heat Risk</th>
              </tr>
            </thead>
            <tbody>
              {seasonalOutlook.map((item) => (
                <tr key={item.period}>
                  <td>{item.period}</td>
                  <td>{item.avg}</td>
                  <td>{item.risk}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </main>
  );
}

export default Temperature;
