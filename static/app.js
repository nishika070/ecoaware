function renderHomeChart() {
  if (!window.ecoAwareHomeChart || typeof Chart === "undefined") return;

  const canvas = document.getElementById("homeTrendChart");
  if (!canvas) return;

  const data = window.ecoAwareHomeChart;

  new Chart(canvas, {
    type: "line",
    data: {
      labels: [...data.labels, "Tomorrow"],
      datasets: [
        {
          label: "AQI",
          data: [...data.aqi_values, data.prediction],
          borderColor: "#b03a2e",
          backgroundColor: "rgba(176, 58, 46, 0.18)",
          tension: 0.35,
          fill: true,
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  });
}

function renderTemperatureChart() {
  if (!window.ecoAwareTemperatureChart || typeof Chart === "undefined") return;

  const canvas = document.getElementById("temperaturePatternChart");
  if (!canvas) return;

  const data = window.ecoAwareTemperatureChart;
  if (!Array.isArray(data.labels) || data.labels.length === 0) return;

  new Chart(canvas, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Max Temp (C)",
          data: data.max_temps,
          borderColor: "#d55353",
          backgroundColor: "rgba(213, 83, 83, 0.15)",
          fill: false,
          tension: 0.3,
          pointRadius: 3,
        },
        {
          label: "Min Temp (C)",
          data: data.min_temps,
          borderColor: "#1f8f5f",
          backgroundColor: "rgba(31, 143, 95, 0.15)",
          fill: false,
          tension: 0.3,
          pointRadius: 3,
        },
        {
          label: "Rain Chance (%)",
          data: data.precip_chance,
          type: "bar",
          backgroundColor: "rgba(44, 119, 175, 0.3)",
          borderRadius: 8,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          title: { display: true, text: "Temperature (C)" },
        },
        y1: {
          beginAtZero: true,
          position: "right",
          min: 0,
          max: 100,
          grid: { drawOnChartArea: false },
          title: { display: true, text: "Rain Chance (%)" },
        },
      },
    },
  });
}

function renderHomeMap() {
  if (!window.ecoAwareHomeMap || typeof L === "undefined") return;

  const mapContainer = document.getElementById("delhiHotspotMap");
  if (!mapContainer) return;

  const mapData = window.ecoAwareHomeMap;
  const map = L.map(mapContainer).setView(
    [mapData.center.lat, mapData.center.lng],
    10
  );

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  (mapData.markers || []).forEach((marker) => {
    const circle = L.circleMarker([marker.lat, marker.lng], {
      radius: 9,
      color: marker.color || "#d55353",
      fillColor: marker.color || "#d55353",
      fillOpacity: 0.7,
      weight: 1.5,
    }).addTo(map);

    circle.bindPopup(
      `<strong>${marker.name}</strong><br/>AQI: ${marker.aqi}<br/>Status: ${marker.status}<br/>${marker.latest_date}`
    );
  });
}

window.addEventListener("DOMContentLoaded", () => {
  renderHomeChart();
  renderTemperatureChart();
  renderHomeMap();
});
