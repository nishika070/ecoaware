import "../styles/pages/Contact.css";

function Contact() {
  return (
    <main className="page-shell">
      <div className="container contact-page">
        <section className="page-heading">
          <h1>Contact & Collaboration</h1>
          <p>For datasets, integrations, and public-health collaboration, connect with the EcoAware team.</p>
        </section>

        <section className="panel contact-grid">
          <div>
            <h2>Email</h2>
            <p>ecoaware.delhi@project.org</p>
          </div>
          <div>
            <h2>Response SLA</h2>
            <p>Within 24 hours for critical pollution alerts.</p>
          </div>
          <div>
            <h2>Data Partnerships</h2>
            <p>Open to AQI feeds, health datasets, and municipal records APIs.</p>
          </div>
        </section>
      </div>
    </main>
  );
}

export default Contact;
