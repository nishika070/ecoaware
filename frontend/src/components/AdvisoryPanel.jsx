import "../styles/components/AdvisoryPanel.css";

const recommendations = [
  "11 AM - 5 PM ke beech prolonged outdoor activity avoid karein",
  "Sensitive groups (children, elderly, asthma patients) N95 use karein",
  "Hydration high rakhein aur light breathable clothing pehnein",
  "Schools, RWAs, offices ko indoor air quality checks activate karne chahiye"
];

function AdvisoryPanel() {
  return (
    <aside className="advisory panel">
      <div className="advisory-head">
        <h2>Health Advisory Engine</h2>
        <span className="chip chip-danger">Heat + AQI Combined Risk: High</span>
      </div>

      <p className="advisory-subtext">
        Current model indicates elevated heat-stress probability in dense traffic and low-green-cover zones of Delhi.
      </p>

      <ul>
        {recommendations.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </aside>
  );
}

export default AdvisoryPanel;
