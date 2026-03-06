import DashboardCard from "../components/DashboardCard";
import AQIChart from "../components/AQIChart";
import AdvisoryPanel from "../components/AdvisoryPanel";

import "../styles/pages/Home.css";

function Home() {

  return (
    <div className="dashboard">

      <h1>EcoAware Dashboard</h1>

      <div className="cards">

        <DashboardCard title="AQI" value="120"/>
        <DashboardCard title="Temperature" value="32°C"/>
        <DashboardCard title="PM2.5" value="85"/>
        <DashboardCard title="PM10" value="140"/>

      </div>

      <AQIChart/>

      <AdvisoryPanel/>

    </div>
  );
}

export default Home;