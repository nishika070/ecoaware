import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders ecoaware navigation", () => {
  render(<App />);
  const homeElement = screen.getByText(/EcoAware Delhi/i);
  expect(homeElement).toBeInTheDocument();
});
