import { Link } from "react-router-dom";
import "../styles/components/Navbar.css";

function Navbar(){

  return(

    <nav className="navbar">

      <div className="logo">EcoAware</div>

      <div className="nav-links">
        <Link to="/">Home</Link>
        <Link to="/aqi">AQI</Link>
        <Link to="/temperature">Temperature</Link>
        <Link to="/policies">Policies</Link>
        <Link to="/contact">Contact</Link>
      </div>

    </nav>

  )

}

export default Navbar