import "../styles/components/Card.css";

function DashboardCard({title,value}){

  return(
    <div className="card">

      <div className="card-title">{title}</div>
      <div className="card-value">{value}</div>

    </div>
  )

}

export default DashboardCard