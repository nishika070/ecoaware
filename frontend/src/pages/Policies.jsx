import "../styles/pages/Policies.css";

const policyItems = [
  {
    title: "Construction intensity tracker",
    detail: "Detect sanctioned vs active construction load and trigger dust compliance inspections."
  },
  {
    title: "Industrial hotspot index",
    detail: "Overlay AQI spikes with industrial emission zones for targeted shutdown windows."
  },
  {
    title: "Deforestation pressure map",
    detail: "Prioritize urban afforestation wards where heat and pollution jointly increase."
  },
  {
    title: "Traffic emission corridors",
    detail: "Identify peak-hour road segments needing rerouting and stricter fuel checks."
  }
];

function Policies() {
  return (
    <main className="page-shell">
      <div className="container policy-page">
        <section className="page-heading">
          <h1>Government & Policy Insights</h1>
          <p>
            Data-backed intervention framework for district administration and city pollution-control actions.
          </p>
        </section>

        <section className="grid grid-2">
          {policyItems.map((item) => (
            <article key={item.title} className="panel policy-card">
              <h2>{item.title}</h2>
              <p>{item.detail}</p>
            </article>
          ))}
        </section>
      </div>
    </main>
  );
}

export default Policies;
