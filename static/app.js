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
  if (!data || !Array.isArray(data.labels)) return;

  const allLabels = [...data.labels, ...(data.forecast_labels || [])];

  const actualData = [
    ...data.actual,
    ...Array(data.forecast_labels?.length || 0).fill(null),
  ];

  const smoothedData = [
    ...data.smoothed,
    ...Array(data.forecast_labels?.length || 0).fill(null),
  ];

  const forecastData = [
    ...Array(data.labels.length - 1).fill(null),
    data.actual[data.actual.length - 1],
    ...(data.forecast || []),
  ];

  new Chart(canvas, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: [
        {
          label: "Actual AQI",
          data: actualData,
          borderColor: "#2f8f8b",
          backgroundColor: "rgba(47,143,139,0.12)",
          borderWidth: 2,
          tension: 0.35,
          fill: true,
          pointRadius: 2,
          spanGaps: false,
        },
        {
          label: "Smoothed",
          data: smoothedData,
          borderColor: "#4db6ac",
          backgroundColor: "transparent",
          borderWidth: 1.5,
          borderDash: [4, 3],
          tension: 0.4,
          fill: false,
          pointRadius: 0,
          spanGaps: true,
        },
        {
          label: "Forecast",
          data: forecastData,
          borderColor: "#e67e22",
          backgroundColor: "rgba(230,126,34,0.10)",
          borderWidth: 2,
          borderDash: [6, 4],
          tension: 0.3,
          fill: false,
          pointRadius: 3,
          pointBackgroundColor: "#e67e22",
          spanGaps: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#222b38", font: { size: 12 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y ?? "—"}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#3b4f62", maxTicksLimit: 10 },
          grid: { color: "rgba(255,255,255,0.6)" },
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

  const map = L.map(mapContainer, { zoomControl: true }).setView(
    [mapData.center.lat, mapData.center.lng], 11
  );

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  (mapData.markers || []).forEach((marker) => {
    const circle = L.circleMarker([marker.lat, marker.lng], {
      radius: marker.radius || 10,
      color: marker.color,
      fillColor: marker.color,
      fillOpacity: 0.75,
      weight: 2,
    }).addTo(map);

    circle.bindPopup(`
      <div style="font-family:sans-serif;min-width:140px">
        <strong style="font-size:14px">${marker.name}</strong><br/>
        <span style="font-size:22px;font-weight:800;color:${marker.color}">${marker.aqi}</span>
        <span style="font-size:11px;color:#666"> AQI</span><br/>
        <span style="font-size:12px;color:#444">${marker.status}</span><br/>
        <span style="font-size:11px;color:#888">${marker.advice}</span>
      </div>
    `);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  renderHomeChart();
  renderHomeMap();
});