import { Link, NavLink } from "react-router-dom";
import "../styles/components/Navbar.css";

function Navbar() {
  return (
    <nav className="navbar">
      <div className="container navbar-inner">
        <Link to="/" className="logo" aria-label="EcoAware Home">
          <span className="logo-badge">EA</span>
          EcoAware Delhi
        </Link>

        <div className="nav-links">
          <NavLink to="/">Home</NavLink>
          <NavLink to="/aqi">AQI</NavLink>
          <NavLink to="/temperature">Temperature</NavLink>
          <NavLink to="/policies">Policy Insights</NavLink>
          <NavLink to="/contact">Contact</NavLink>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
