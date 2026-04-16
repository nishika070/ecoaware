function renderHomeChart() {
  if (!window.ecoAwareHomeChart) {
    return;
  }

  const canvas = document.getElementById("homeTrendChart");
  if (!canvas) {
    return;
  }

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

async function loadDelhiTemperatures() {
  const locations = [
    {
      name: "Anand Vihar",
      latitude: 28.65,
      longitude: 77.31,
      elementId: "temp-anand-vihar"
    },
    {
      name: "RK Puram",
      latitude: 28.57,
      longitude: 77.18,
      elementId: "temp-rk-puram"
    },
    {
      name: "Punjabi Bagh",
      latitude: 28.67,
      longitude: 77.12,
      elementId: "temp-punjabi-bagh"
    },
    {
      name: "Mandir Marg",
      latitude: 28.63,
      longitude: 77.21,
      elementId: "temp-mandir-marg"
    }
  ];

  const latitudeList = locations.map((location) => location.latitude).join(",");
  const longitudeList = locations.map((location) => location.longitude).join(",");

  const apiUrl =
    `https://api.open-meteo.com/v1/forecast?latitude=${latitudeList}` +
    `&longitude=${longitudeList}` +
    `&current=temperature_2m`;

  try {
    const response = await fetch(apiUrl);

    if (!response.ok) {
      throw new Error("Failed to fetch temperature data");
    }

    const data = await response.json();
    const results = Array.isArray(data) ? data : [data];

    results.forEach((result, index) => {
      const location = locations[index];
      const temperature = result?.current?.temperature_2m;
      const element = document.getElementById(location.elementId);

      if (!element) {
        return;
      }

      if (temperature !== undefined && temperature !== null) {
        element.textContent = `${temperature}°C`;
      } else {
        element.textContent = "Not available";
      }
    });
  } catch (error) {
    console.error("Temperature API error:", error);

    locations.forEach((location) => {
      const element = document.getElementById(location.elementId);
      if (element) {
        element.textContent = "Error loading";
      }
    });
  }
}

function renderTemperatureChart() {
  if (!window.ecoAwareTemperatureChart) {
    return;
  }

  const canvas = document.getElementById("temperaturePatternChart");
  if (!canvas) {
    return;
  }

  const data = window.ecoAwareTemperatureChart;

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: data.map((item) => item.month),
      datasets: [
        {
          label: "Average AQI",
          data: data.map((item) => item.aqi),
          backgroundColor: "rgba(31, 143, 95, 0.72)",
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  });
}

renderHomeChart();
loadDelhiTemperatures();
renderTemperatureChart();
