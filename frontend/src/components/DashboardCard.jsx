import "../styles/components/Card.css";

function DashboardCard({ title, value, note }) {
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="card-value">{value}</div>
      {note ? <p className="card-note">{note}</p> : null}
    </div>
  );
}

export default DashboardCard;
