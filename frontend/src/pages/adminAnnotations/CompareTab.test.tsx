import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import type { AdminAnnotationQueue } from "../../types";
import { CompareTab } from "./CompareTab";

const api = vi.hoisted(() => ({
  getComparableModels: vi.fn(),
  startModelComparison: vi.fn(),
  getAdminTrainingJob: vi.fn(),
  getLatestAdminTrainingJob: vi.fn(),
  adminAnnotationMediaUrl: (path: string) => `http://api.test${path}`,
}));
vi.mock("../../services/api", () => api);
beforeEach(() => api.getLatestAdminTrainingJob.mockResolvedValue({ job: null }));

it("restores the last comparison when returning to the tab without rerunning it", async () => {
  api.getComparableModels.mockResolvedValue({ versions: [{ version_id: "candidate-2", status: "candidate" }] });
  api.getLatestAdminTrainingJob.mockResolvedValue({ job: {
    run_id: "comparison-restored", kind: "comparison", model_version_id: "candidate-2",
    status: "succeeded", comparison: {
      sample_count: 0, warnings: [], rows: [],
      protocol: { metrics_scope: "ad_hoc", device: "cpu", base_historical_independence: "unknown" },
      metrics: {
        common: { support: 0, base_correct: 0, candidate_correct: 0, gains: 0, regressions: 0, both_wrong: 0 },
        new_classes: { support: 0, candidate_correct: 0 },
        coverage: { base: 0, candidate: 0 }, top3: { base_correct: 0, candidate_correct: 0 },
      },
    },
  } });
  const first = render(<CompareTab queue={null} />);
  expect(await screen.findByLabelText("Résumé de la comparaison")).toBeVisible();
  first.unmount();
  render(<CompareTab queue={null} />);
  expect(await screen.findByLabelText("Résumé de la comparaison")).toBeVisible();
  expect(screen.getByLabelText("Version candidate")).toHaveValue("candidate-2");
  expect(api.startModelComparison).not.toHaveBeenCalled();
});

it("compares every bbox on both full pages, synchronizes selection, and handles a missing image", async () => {
  api.getComparableModels.mockResolvedValue({ versions: [
    { version_id: "candidate-1", status: "candidate" },
    { version_id: "candidate-2", status: "candidate" },
  ] });
  const base = { class_name: "atl", confidence: 0.9, rejected: false, top_k: [{ class_name: "atl", confidence: 0.9 }] };
  const candidate = { class_name: "calli", confidence: 0.7, rejected: false, top_k: [{ class_name: "calli", confidence: 0.7 }] };
  const response = { job: {
    run_id: "comparison-1", status: "succeeded", dry_run: false, device: "cpu", batch_size: 1,
    comparison: {
      sample_count: 2, warnings: [],
      protocol: { evaluation_scope: "ad_hoc", metrics_scope: "ad_hoc", base_historical_independence: "unknown", device: "cpu" },
      metrics: {
        common: { support: 1, base_correct: 1, candidate_correct: 0, gains: 0, regressions: 1, unchanged: 0, both_wrong: 0 },
        new_classes: { support: 0, candidate_correct: 0 },
        coverage: { base: 1, candidate: 1 }, latency_ms: { base: 1, candidate: 1 },
        by_scope: {}, top3: { base_correct: 1, candidate_correct: 0 }, per_class: {},
      },
      rows: [
        {
          sample_id: "sample-1", analysis_id: "a1", index: 0, image_name: "page.png",
          bbox: [2, 3, 4, 5], source_image_size: [20, 30], source_image_sha256: "same-page",
          scope: "ad_hoc", review_status: "approved", candidate_exposure: "train",
          expected_class: "atl", base_supported: true, base, candidate,
          outcome: "regression", crop_url: "/crop-1.png", source_image_url: "/source.png",
        },
        {
          sample_id: "sample-2", analysis_id: "a1", index: 1, image_name: "page.png",
          bbox: [10, 12, 5, 6], source_image_size: [20, 30], source_image_sha256: "same-page",
          scope: "ad_hoc", review_status: "pending", candidate_exposure: "train",
          expected_class: null, base_supported: true, base, candidate: base,
          outcome: "unchanged", crop_url: "/crop-2.png", source_image_url: "/source-2.png",
        },
      ],
    },
  } };
  api.startModelComparison.mockResolvedValue(response);

  const { container } = render(<CompareTab queue={null} />);
  await userEvent.click(await screen.findByRole("button", { name: "Comparer" }));
  expect(screen.getByText(/Bilan du groupe · Page exploratoire · 2 régions capturées au total · 0 correction · 1 régression/i)).toBeVisible();
  expect(screen.getByText(/Mêmes bbox enregistrées, pas de nouvelle détection/i)).toBeVisible();
  expect(screen.getByText(/Cette page a servi à entraîner le candidat/)).toBeVisible();
  const active = screen.getByRole("region", { name: "Modèle actif" });
  const challenger = screen.getByRole("region", { name: "Modèle candidat" });
  expect(within(active).getByRole("img", { name: /Toutes les bbox/ }).querySelectorAll('[data-overlay-region="true"]')).toHaveLength(2);
  expect(within(challenger).getByRole("img", { name: /Toutes les bbox/ }).querySelectorAll('[data-overlay-region="true"]')).toHaveLength(2);
  expect(screen.getByRole("navigation", { name: "Régions comparées" })).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: /région #1/i }));
  expect(within(active).getByRole("img", { name: /Toutes les bbox/ }).querySelector('[data-box-id="sample-2"]')).toHaveAttribute("data-selected", "true");
  expect(within(challenger).getByRole("img", { name: /Toutes les bbox/ }).querySelector('[data-box-id="sample-2"]')).toHaveAttribute("data-selected", "true");
  expect(screen.getByText(/Région #1/)).toBeVisible();
  await userEvent.selectOptions(screen.getByLabelText("Mettre en évidence"), "disagreement");
  expect(screen.getByText(/Mise en évidence : 1\/2 régions.*autres cadres restent visibles/i)).toBeVisible();
  expect(within(active).getByRole("img", { name: /Toutes les bbox/ }).querySelectorAll('[data-overlay-region="true"]')).toHaveLength(2);
  fireEvent.error(container.querySelector('img[src="http://api.test/source.png"]')!);
  expect(within(active).getByText("Page source indisponible")).toBeVisible();

  const firstReport = response.job.comparison;
  api.startModelComparison.mockResolvedValueOnce({ job: {
    ...response.job,
    comparison: {
      ...firstReport,
      sample_count: 3,
      protocol: { ...firstReport.protocol, metrics_scope: "locked_test" },
      rows: [...firstReport.rows, {
        ...firstReport.rows[0], sample_id: "sample-3", source_image_sha256: "test-page",
        image_name: "test.png", scope: "locked_test",
      }],
    },
  } });
  await userEvent.click(screen.getByRole("button", { name: "Comparer" }));
  await userEvent.selectOptions(screen.getByLabelText("Page à inspecter"), "same-page");
  expect(screen.getByText(/Bilan du groupe · Test réservé/i)).toBeVisible();
  expect(screen.getByText(/les chiffres ci-dessus restent ceux du groupe Test réservé/i)).toBeVisible();
  await userEvent.selectOptions(screen.getByLabelText("Version candidate"), "candidate-2");
  expect(screen.queryByRole("region", { name: "Modèle actif" })).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Résumé de la comparaison")).not.toBeInTheDocument();
});

it("chooses the analyzed page with most stored regions and explains a one-box page", async () => {
  api.startModelComparison.mockClear();
  api.getComparableModels.mockResolvedValue({ versions: [{ version_id: "candidate-1", status: "candidate" }] });
  api.startModelComparison.mockResolvedValue({ job: { run_id: "comparison-2", status: "running" } });
  const queue = { analyses: [
    { analysis_id: "single", image_name: "single.jpg", image_exists: true, elements: [{ index: 0 }] },
    { analysis_id: "rich", image_name: "rich.jpg", image_exists: true, elements: [{ index: 0 }, { index: 1 }] },
  ] } as AdminAnnotationQueue;
  render(<CompareTab queue={queue} />);
  expect(await screen.findByRole("option", { name: "Page analysée · rich.jpg · 2 régions" })).toBeVisible();
  expect(screen.getByLabelText("Source des images")).toHaveValue("rich");
  await userEvent.selectOptions(screen.getByLabelText("Source des images"), "single");
  expect(screen.getByText(/Pour comparer plusieurs zones sur une même image/)).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "Comparer" }));
  expect(api.startModelComparison).toHaveBeenCalledWith("candidate-1", "single");
});

it("does not offer comparison launches to a training reader", async () => {
  api.getComparableModels.mockResolvedValue({ versions: [{ version_id: "candidate-1", status: "candidate" }] });
  render(<CompareTab queue={null} canRunTraining={false} />);
  expect(await screen.findByRole("option", { name: "candidate-1" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Comparer" })).toBeDisabled();
  expect(screen.getByText(/autorisation de lancement requise/i)).toBeInTheDocument();
});

it("follows a new candidate selected from the training deep link", async () => {
  api.getComparableModels.mockResolvedValue({ versions: [
    { version_id: "candidate-1", status: "candidate" },
    { version_id: "candidate-2", status: "candidate" },
  ] });
  const { rerender } = render(<CompareTab queue={null} initialVersionId="candidate-1" />);
  expect(await screen.findByRole("option", { name: "candidate-2" })).toBeInTheDocument();
  rerender(<CompareTab queue={null} initialVersionId="candidate-2" />);
  await waitFor(() => expect(screen.getByLabelText("Version candidate")).toHaveValue("candidate-2"));
});
