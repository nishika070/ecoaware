import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import { Line } from "react-chartjs-2";
import "../styles/components/Chart.css";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Title, Tooltip, Legend);

function AQIChart() {
  const data = {
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    datasets: [
      {
        label: "AQI",
        data: [168, 174, 188, 182, 176, 192, 185],
        borderColor: "#b03a2e",
        backgroundColor: "rgba(176, 58, 46, 0.18)",
        yAxisID: "y",
        tension: 0.35
      },
      {
        label: "Temperature (C)",
        data: [32, 33, 34, 35, 34, 33, 34],
        borderColor: "#1f8f5f",
        backgroundColor: "rgba(31, 143, 95, 0.16)",
        yAxisID: "y1",
        tension: 0.35
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top"
      }
    },
    scales: {
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: "AQI"
        }
      },
      y1: {
        position: "right",
        grid: {
          drawOnChartArea: false
        },
        title: {
          display: true,
          text: "Temperature (C)"
        }
      }
    }
  };

  return (
    <div className="chart-wrap">
      <Line data={data} options={options} />
    </div>
  );
}

export default AQIChart;
