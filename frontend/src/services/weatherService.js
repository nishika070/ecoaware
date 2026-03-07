export async function getWeather(lat, lon) {

  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`;

  const response = await fetch(url);

  const data = await response.json();

  return data.current_weather;

}