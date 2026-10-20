import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { AdminMediaImage } from "./AdminMediaImage";

it("shows a visible fallback when an admin image fails to load", () => {
  render(<AdminMediaImage src="/missing.png" alt="Découpe test" />);
  fireEvent.error(screen.getByRole("img", { name: "Découpe test" }));
  expect(screen.getByText("Image indisponible")).toBeVisible();
});
