function readJsonData(scriptId) {
  const node = document.getElementById(scriptId);
  if (!node) return null;

  try {
    return JSON.parse(node.textContent);
  } catch (error) {
    console.error(`Invalid JSON in ${scriptId}:`, error);
    return null;
  }
}

function renderHomeChart() {
  if (typeof Chart === "undefined") return;

  const canvas = document.getElementById("homeTrendChart");
  if (!canvas) return;

  const data = readJsonData("home-chart-data");
  if (!data || !Array.isArray(data.labels) || !Array.isArray(data.aqi_values)) return;

  new Chart(canvas, {
    type: "line",
    data: {
      labels: [...data.labels, "Tomorrow"],
      datasets: [
        {
          label: "AQI",
          data: [...data.aqi_values, data.prediction],
          borderColor: "#000000",
          backgroundColor: "rgba(10, 163, 234, 0.18)",
          borderWidth: 2,
          tension: 0.35,
          fill: true,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#222b38" },
        },
      },
      scales: {
        x: {
          ticks: { color: "#3b4f62" },
          grid: { color: "#ffffff" },
        },
        y: {
          ticks: { color: "#b8bdc2" },
          grid: { color: "#acb7c1" },
        },
      },
    },
  });
}

function renderHomeMap() {
  if (typeof L === "undefined") return;

  const mapContainer = document.getElementById("delhiHotspotMap");
  if (!mapContainer) return;

  const mapData = readJsonData("home-map-data");
  if (!mapData || !mapData.center) return;

  const map = L.map(mapContainer).setView(
    [mapData.center.lat, mapData.center.lng],
    10
  );

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  (mapData.markers || []).forEach((marker) => {
    const circle = L.circleMarker([marker.lat, marker.lng], {
      radius: 8,
      color: marker.color || "#520e0e",
      fillColor: marker.color || "#d55353",
      fillOpacity: 0.78,
      weight: 1.2,
    }).addTo(map);

    circle.bindPopup(
      `<strong>${marker.name}</strong><br/>AQI: ${marker.aqi}<br/>Status: ${marker.status}<br/>Spread: ${marker.relative_label}<br/>${marker.latest_date}`
    );
  });
}

window.addEventListener("DOMContentLoaded", () => {
  renderHomeChart();
  renderHomeMap();
});
