import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AdminAnnotationsPage from "./AdminAnnotationsPage";
import { ClassDistributionBars } from "./adminAnnotations/TrainingHelpers";
import type {
  AdminAnnotationQueue,
  AdminTrainingJob,
  AdminTrainingSummary,
} from "../types";

const apiMock = vi.hoisted(() => ({
  adminAnnotationMediaUrl: vi.fn((path: string) => `http://api.test${path}`),
  getClasses: vi.fn(),
  getAdminClasses: vi.fn(),
  confirmAdminClass: vi.fn(),
  getAdminAnnotationHistory: vi.fn(),
  restoreAdminAnnotationElement: vi.fn(),
  getComparableModels: vi.fn(),
  getAdminTrainingJob: vi.fn(),
  startModelComparison: vi.fn(),
  getAdminAnnotationQueue: vi.fn(),
  getAdminTrainingSummary: vi.fn(),
  getLatestAdminTrainingJob: vi.fn(),
  modifyAdminAnnotationElement: vi.fn(),
  setAdminAnnotationReviewStatus: vi.fn(),
  startAdminTrainingJob: vi.fn(),
}));

vi.mock("../services/api", () => apiMock);

it("does not draw positive bars for empty dataset splits", () => {
  const { container } = render(<ClassDistributionBars splitCounts={{ train: 1, val: 0, test: 0, excluded: 0 }} />);
  expect([...container.querySelectorAll("rect")].map(rect => rect.getAttribute("height"))).toEqual(["96", "0", "0", "0"]);
});

function queueWithStatuses(
  status0: "pending" | "approved" | "rejected",
  status1: "pending" | "approved" | "rejected",
): AdminAnnotationQueue {
  const trainable0 = status0 === "approved";
  const trainable1 = status1 === "approved";
  return {
    status: "ok",
    schema_version: 1,
    local_only: true,
    warning:
      "Local/dev-only annotation review endpoint. It is not production-secured.",
    counts: {
      total: 2,
      pending: [status0, status1].filter((status) => status === "pending")
        .length,
      approved: [status0, status1].filter((status) => status === "approved")
        .length,
      rejected: [status0, status1].filter((status) => status === "rejected")
        .length,
      trainable: [trainable0, trainable1].filter(Boolean).length,
    },
    analyses: [
      {
        analysis_id: "analysis-1",
        uploaded_at: "2026-05-26T13:00:00+00:00",
        image_path: "/tmp/annotations/analysis-1/image.png",
        image_url: "/admin/annotations/analysis-1/image",
        image_exists: true,
        elements: [
          {
            key: "analysis-1:0",
            revision: 0,
            analysis_id: "analysis-1",
            index: 0,
            class_name: "atl",
            bbox: [0, 1, 2, 3],
            crop_path: "/tmp/annotations/analysis-1/elements/0.png",
            crop_url: "/admin/annotations/analysis-1/0/crop",
            crop_exists: true,
            review_status: status0,
            trainable: trainable0,
            source_fingerprint: "fingerprint-0",
            stale_decision: false,
            dataset_split: trainable0 ? "train" : "excluded",
            split_reason: trainable0
              ? "trainable_hash_80_10_10"
              : status0 === "rejected"
                ? "rejected_review"
                : "pending_review",
          },
          {
            key: "analysis-1:1",
            revision: 0,
            analysis_id: "analysis-1",
            index: 1,
            class_name: "calli",
            bbox: [4, 5, 6, 7],
            crop_path: "/tmp/annotations/analysis-1/elements/1.png",
            crop_url: "/admin/annotations/analysis-1/1/crop",
            crop_exists: true,
            review_status: status1,
            trainable: trainable1,
            source_fingerprint: "fingerprint-1",
            stale_decision: false,
            dataset_split: trainable1 ? "val" : "excluded",
            split_reason: trainable1
              ? "trainable_hash_80_10_10"
              : status1 === "rejected"
                ? "rejected_review"
                : "pending_review",
          },
        ],
      },
    ],
    diagnostics: [],
  };
}

function trainingJob(
  overrides: Partial<AdminTrainingJob> = {},
): AdminTrainingJob {
  return {
    run_id: "run-1",
    status: "running",
    dry_run: true,
    device: "cpu",
    batch_size: 8,
    started_at: "2026-05-26T14:00:00+00:00",
    exit_code: null,
    log_tail: ["stage 1"],
    ...overrides,
  };
}

function trainingSummary(
  overrides: Partial<AdminTrainingSummary> = {},
): AdminTrainingSummary {
  return {
    status: "ok",
    local_only: true,
    warning: "local only",
    training_jobs_enabled: false,
    launch_allowed_for_request: false,
    launch_disabled_reasons: [
      "disabled_by_default: set ENABLE_ADMIN_TRAINING_JOBS=1 to allow local launches",
    ],
    data: {
      total: 4,
      pending: 1,
      approved: 2,
      rejected: 1,
      trainable: 1,
      classes: ["atl"],
      per_class: { atl: 1 },
      split_counts: { train: 1, val: 0, test: 0, excluded: 3 },
      diagnostics: [],
    },
    parameters: {
      editable: {
        dry_run: true,
        device: ["auto", "cpu", "mps", "cuda"],
        batch_size: { default: 16, min: 1, max: 256 },
      },
      script_env_defaults: { BATCH_SIZE: "16", DEVICE: "auto" },
      config: {
        training: { num_epochs: 100 },
        model: { backbone: "dinov2_vits14" },
      },
    },
    paths: {
      script: "/repo/scripts/retrain.sh",
      runs_dir: "/repo/backend/training_runs",
      model_dir_override_active: false,
    },
    artifacts: {
      approved_export_manifest: {
        path: "/repo/backend/training_data/approved/Elements/_approved_export_manifest.json",
        exists: true,
        sha256: "abc123",
      },
      model_registry: {
        status: "ok",
        promoted_version: "20260527T010203Z-demo",
      },
    },
    training_snapshot: {
      configured: true,
      data_revision: "a".repeat(64),
      valid: true,
      snapshot_id: "snapshot-test",
      snapshot_manifest_sha256: "snapshot-sha",
      row_count: 9266,
      class_count: 286,
      live_annotation_count: 7,
      live_train_count: 7,
      ready_for_training: true,
      promotion_evaluation_ready: false,
      split_counts: { train: 9128, dev: 56, locked_test: 82 },
      paths: {
        manifest: "/repo/backend/training_corpus/snapshots/snapshot-test/snapshot_manifest.json",
      },
      errors: [],
    },
    latest_job: null,
    ...overrides,
  };
}

async function renderLoaded(queue = queueWithStatuses("pending", "rejected")) {
  apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queue);
  render(<AdminAnnotationsPage themeMode="light" onToggleTheme={vi.fn()} />);
  await screen.findByRole("heading", { name: /poste de triage/i });
  return screen.findByRole("listbox", { name: /file de triage/i });
}

describe("AdminAnnotationsPage", () => {
  beforeEach(() => {
    apiMock.adminAnnotationMediaUrl.mockClear();
    apiMock.getClasses.mockReset();
    apiMock.getAdminClasses.mockReset();
    apiMock.confirmAdminClass.mockReset();
    apiMock.getAdminAnnotationHistory.mockReset();
    apiMock.restoreAdminAnnotationElement.mockReset();
    apiMock.getComparableModels.mockReset();
    apiMock.getAdminTrainingJob.mockReset();
    apiMock.startModelComparison.mockReset();
    apiMock.getAdminAnnotationQueue.mockReset();
    apiMock.getAdminTrainingSummary.mockReset();
    apiMock.getLatestAdminTrainingJob.mockReset();
    apiMock.modifyAdminAnnotationElement.mockReset();
    apiMock.setAdminAnnotationReviewStatus.mockReset();
    apiMock.startAdminTrainingJob.mockReset();
    apiMock.getClasses.mockResolvedValue({
      num_classes: 2,
      class_names: ["atl", "calli"],
    });
    apiMock.getAdminClasses.mockResolvedValue({
      revision: "a".repeat(64),
      classes: ["atl", "calli"].map((class_name, class_label) => ({
        class_name, class_label, status: "active",
        counts: { pending: 0, approved: 0, rejected: 0 }, trainable_count: 0,
      })),
    });
    apiMock.getAdminAnnotationHistory.mockResolvedValue({ revision: 0, history: [] });
    apiMock.getComparableModels.mockResolvedValue({ versions: [] });
  });

  it("renders the reference visual triage workstation with rows and decision inspector", async () => {
    await renderLoaded();

    expect(
      screen.getByRole("tablist", { name: /étapes du poste de triage/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /trier/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: /dataset/i })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(screen.getByRole("tab", { name: /entraîner/i })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(screen.queryByText(/codex-014/i)).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /retour à l’analyse/i })).toBeInTheDocument();
    expect(screen.getByText(/options du poste/i)).toBeInTheDocument();

    expect(
      screen.getByRole("listbox", { name: /file de triage/i }),
    ).toBeInTheDocument();
    const selectedReviewRow = screen.getByRole("option", {
      name: /ouvrir l'élément 0 atl du triage/i,
    });
    expect(selectedReviewRow).toHaveAttribute("aria-selected", "true");
    expect(selectedReviewRow).toHaveAccessibleName(/statut À vérifier/i);
    expect(
      screen.getByRole("heading", { name: /élément #0 · atl/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /image complète analysis-1/i }),
    ).toHaveAttribute(
      "src",
      "http://api.test/admin/annotations/analysis-1/image",
    );
    const imageHeader = screen.getByTestId("admin-review-image-header");
    expect(imageHeader).toContainElement(
      screen.getByText(/détails techniques \/ audit/i).closest("details"),
    );
    expect(
      screen.getByRole("button", { name: /^zoomer$/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /ajuster à la vue/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("img", { name: /découpe 0 pour atl/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/effet sur le dataset/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /valider/i })).toBeEnabled();
    expect(
      screen.queryByRole("button", { name: /retrain/i }),
    ).not.toBeInTheDocument();
  });

  it("keeps a legacy element with no bbox visible for correction", async () => {
    const queue = queueWithStatuses("pending", "rejected");
    queue.analyses[0].elements[0].bbox = [];
    await renderLoaded(queue);

    expect(screen.getAllByText("Zone à corriger").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /corriger/i })).toBeInTheDocument();
  });

  it("filters triage rows, recovers empty filters, and manually refreshes the queue", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queueWithStatuses("pending", "rejected"))
      .mockResolvedValueOnce(queueWithStatuses("approved", "rejected"));

    render(<AdminAnnotationsPage />);

    expect(
      await screen.findByText(/2 \/ 2 éléments affichés/i),
    ).toBeInTheDocument();

    await user.selectOptions(
      screen.getByRole("combobox", { name: /^statut$/i }),
      "rejected",
    );
    expect(screen.getByText(/1 \/ 2 éléments affiché/i)).toBeInTheDocument();
    expect(
      screen.getByRole("list", { name: /filtres de triage appliqués/i }),
    ).toHaveTextContent(/statut : rejetés/i);

    await user.selectOptions(
      screen.getByRole("combobox", { name: /^classe$/i }),
      "atl",
    );
    expect(
      screen.getByText(/aucun élément ne correspond aux filtres actifs/i),
    ).toBeInTheDocument();
    await user.click(
      screen.getAllByRole("button", { name: /effacer les filtres/i }).at(-1)!,
    );
    expect(
      screen.getByRole("option", { name: /ouvrir l'élément 0 atl/i }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /actualiser/i }));
    await waitFor(() =>
      expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(2),
    );
    expect(
      await screen.findByLabelText(
        /élément sélectionné 0 atl : statut Validé/i,
      ),
    ).toBeInTheDocument();
  });

  it("shows the Validés filter even when every annotation is approved", async () => {
    const user = userEvent.setup();
    const approvedQueue = queueWithStatuses("approved", "approved");
    apiMock.getAdminAnnotationQueue.mockResolvedValue(approvedQueue);
    render(<AdminAnnotationsPage />);

    await user.click(await screen.findByRole("button", { name: "Validés" }));
    const approvedList = screen.getByRole("listbox", { name: /file de triage/i });
    expect(within(approvedList).getAllByRole("option")).toHaveLength(2);
    expect(screen.getByRole("combobox", { name: /^statut$/i })).toHaveValue("approved");
    await user.selectOptions(screen.getByRole("combobox", { name: /^classe$/i }), "atl");
    expect(within(approvedList).getAllByRole("option")).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: /actualiser/i }));
    await waitFor(() => expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("option", { name: /ouvrir l'élément 0 atl/i }))
      .toHaveAccessibleName(/statut Validé/i);
  });

  it("keeps current queue visible when manual refresh fails", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queueWithStatuses("pending", "rejected"))
      .mockRejectedValueOnce(new Error("offline"));

    render(<AdminAnnotationsPage />);

    expect(
      await screen.findByRole("option", { name: /ouvrir l'élément 0 atl/i }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /actualiser/i }));

    expect(
      await screen.findByText(
        /impossible de charger la file locale de triage/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: /ouvrir l'élément 0 atl/i }),
    ).toBeInTheDocument();
  });

  it("selects triage rows and navigates the inspector within active filters", async () => {
    const user = userEvent.setup();
    await renderLoaded();

    const firstRow = screen.getByRole("option", {
      name: /ouvrir l'élément 0 atl/i,
    });
    const secondRow = screen.getByRole("option", {
      name: /ouvrir l'élément 1 calli/i,
    });
    expect(firstRow).toHaveAttribute("aria-current", "true");

    await user.click(secondRow);

    expect(
      screen.getByRole("heading", { name: /élément #1 · calli/i }),
    ).toBeInTheDocument();
    expect(secondRow).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: /suivant/i })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: /précédent/i }));
    expect(
      screen.getByRole("heading", { name: /élément #0 · atl/i }),
    ).toBeInTheDocument();

    await user.selectOptions(
      screen.getByRole("combobox", { name: /^statut$/i }),
      "rejected",
    );
    expect(
      screen.getByRole("option", { name: /ouvrir l'élément 1 calli/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: /ouvrir l'élément 0 atl/i }),
    ).not.toBeInTheDocument();
  });

  it("separates dataset buckets with filters and jumps back to triage", async () => {
    const user = userEvent.setup();
    const mixed = queueWithStatuses("approved", "approved");
    mixed.analyses[0].elements[1] = {
      ...mixed.analyses[0].elements[1],
      trainable: false,
      stale_decision: true,
      dataset_split: "excluded",
      split_reason: "stale_decision",
    };
    mixed.analyses[0].elements.push(
      {
        ...mixed.analyses[0].elements[0],
        key: "analysis-1:2",
        index: 2,
        class_name: "atl",
        bbox: [2, 2, 4, 4],
        crop_url: "/admin/annotations/analysis-1/2/crop",
        review_status: "rejected",
        trainable: false,
        dataset_split: "excluded",
        split_reason: "rejected_review",
        source_fingerprint: "fingerprint-2",
      },
      {
        ...mixed.analyses[0].elements[0],
        key: "analysis-1:3",
        index: 3,
        class_name: "maya",
        bbox: [3, 3, 4, 4],
        crop_url: "/admin/annotations/analysis-1/3/crop",
        review_status: "pending",
        trainable: false,
        dataset_split: "excluded",
        split_reason: "pending_review",
        source_fingerprint: "fingerprint-3",
      },
    );
    mixed.counts = {
      total: 4,
      pending: 1,
      approved: 2,
      rejected: 1,
      trainable: 1,
    };
    mixed.diagnostics = [
      {
        code: "stale_decision",
        message: "fingerprint mismatch",
        key: "analysis-1:1",
        analysis_id: "analysis-1",
        index: 1,
      },
    ];
    await renderLoaded(mixed);
    await user.click(screen.getByRole("tab", { name: /dataset/i }));

    expect(screen.getByLabelText(/classes dataset/i)).toBeInTheDocument();
    expect(screen.getByText((_, element) => element?.tagName === "SPAN" && element.textContent?.trim() === "1 découpe validée")).toBeInTheDocument();
    expect(screen.getByText((_, element) => element?.tagName === "SPAN" && element.textContent?.trim() === "1 image source")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Faibles" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /effacer/i })).toHaveLength(1);
    const datasetList = screen.getByRole("listbox", { name: /vue dataset/i });
    expect(datasetList).toBeInTheDocument();
    expect(
      within(datasetList).getByRole("option", {
        name: /ouvrir élément dataset 0 atl/i,
      }),
    ).toHaveAttribute("aria-selected", "true");
    expect(
      screen.getAllByRole("img", { name: /découpe dataset 0 pour atl/i })[0],
    ).toBeInTheDocument();
    expect(
      within(datasetList).queryByRole("option", {
        name: /ouvrir élément dataset 1 calli/i,
      }),
    ).not.toBeInTheDocument();
    expect(
      within(screen.getByLabelText(/filtres split dataset/i)).queryByRole(
        "button",
        { name: /exclus/i },
      ),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(/1 \/ 1 élément du dataset affiché/i),
    ).toBeInTheDocument();

    await user.click(
      within(screen.getByLabelText(/classes dataset/i)).getByRole("button", {
        name: /atl/i,
      }),
    );
    expect(
      screen.getByText(/1 \/ 1 élément du dataset affiché/i),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: /ouvrir dans le triage/i }),
    );

    expect(screen.getByRole("tab", { name: /trier/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      screen.getByRole("heading", { name: /élément #0 · atl/i }),
    ).toBeInTheDocument();
  });

  it("filters real classes named all and Sans nom separately from an unnamed row", async () => {
    const user = userEvent.setup();
    const queue = queueWithStatuses("approved", "approved");
    queue.analyses[0].elements[0].class_name = "all";
    queue.analyses[0].elements[1].class_name = "";
    queue.analyses[0].elements.push({ ...queue.analyses[0].elements[1], key: "analysis-1:2", index: 2, class_name: "Sans nom" });
    queue.counts = { total: 3, pending: 0, approved: 3, rejected: 0, trainable: 3 };
    await renderLoaded(queue);
    await user.click(screen.getByRole("tab", { name: /dataset/i }));

    await user.click(screen.getByRole("button", { name: /^all 1$/i }));
    expect(within(screen.getByRole("listbox", { name: /vue dataset/i })).getAllByRole("option")).toHaveLength(1);
    expect(screen.getByText("Classe : all")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Sans nom \(non renseigné\) 1/i }));
    expect(within(screen.getByRole("listbox", { name: /vue dataset/i })).getAllByRole("option")).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: /^Sans nom 1$/i }));
    expect(within(screen.getByRole("listbox", { name: /vue dataset/i })).getAllByRole("option")).toHaveLength(1);

    await user.click(screen.getByRole("tab", { name: /trier/i }));
    await user.selectOptions(screen.getByRole("combobox", { name: /^classe$/i }), "class:all");
    expect(within(screen.getByRole("listbox", { name: /file de triage/i })).getAllByRole("option")).toHaveLength(1);
    expect(screen.getByText("Classe : all")).toBeVisible();
    await user.selectOptions(screen.getByRole("combobox", { name: /^classe$/i }), "class:");
    expect(within(screen.getByRole("listbox", { name: /file de triage/i })).getAllByRole("option")).toHaveLength(1);
    await user.selectOptions(screen.getByRole("combobox", { name: /^classe$/i }), "class:Sans nom");
    expect(within(screen.getByRole("listbox", { name: /file de triage/i })).getAllByRole("option")).toHaveLength(1);
  });

  it("shows the candidate workflow and blocking reason without fake metrics", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary());

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));

    expect(await screen.findByRole("heading", { name: /créer un candidat/i })).toBeInTheDocument();
    expect(screen.getByText(/il ne sera pas activé automatiquement/i)).toBeInTheDocument();
    expect(screen.getByText(/préparation bloquée/i)).toBeInTheDocument();
    expect(screen.getByText(/1 annotations utilisables/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /vérifier la préparation/i })).toBeDisabled();
    expect(document.body).not.toHaveTextContent(/placeholder|LossSketch|ValidationAccuracySketch/i);
  });

  it("renders training without annotation queue access for the ML operator", async () => {
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary());
    render(<AdminAnnotationsPage initialTab="training" canReadQueue={false} />);
    expect(await screen.findByRole("heading", { name: /créer un candidat/i })).toBeInTheDocument();
    expect(apiMock.getAdminAnnotationQueue).not.toHaveBeenCalled();
    expect(apiMock.getAdminClasses).not.toHaveBeenCalled();
    expect(screen.queryByRole("tab", { name: /trier/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /dataset/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /classes/i })).not.toBeInTheDocument();
  });

  it("lets a read-only training user inspect readiness without offering a launch", async () => {
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary({
      training_jobs_enabled: true,
      launch_allowed_for_request: true,
      launch_disabled_reasons: [],
    }));
    render(<AdminAnnotationsPage initialTab="training" canReadQueue={false} canRunTraining={false} />);

    expect(await screen.findByRole("button", { name: /vérifier la préparation/i })).toBeDisabled();
    expect(screen.getByText(/autorisation de lancement requise/i)).toBeInTheDocument();
    expect(apiMock.startAdminTrainingJob).not.toHaveBeenCalled();
  });

  it("does not offer training tabs to a reviewer", async () => {
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses("pending", "rejected"));
    render(<AdminAnnotationsPage canReadTraining={false} />);
    expect(await screen.findByRole("listbox", { name: /file de triage/i })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /entraîner/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /comparer/i })).not.toBeInTheDocument();
  });

  it("shows a retryable panel when the review queue fails, then recovers", async () => {
    apiMock.getAdminAnnotationQueue
      .mockRejectedValueOnce(new Error("storage unavailable"))
      .mockResolvedValueOnce(queueWithStatuses("pending", "rejected"));
    render(<AdminAnnotationsPage />);
    const panel = await screen.findByRole("tabpanel", { name: /panneau trier/i });
    expect(await within(panel).findByText("File de triage indisponible")).toBeInTheDocument();
    await userEvent.click(within(panel).getByRole("button", { name: "Réessayer" }));
    expect(await within(panel).findByRole("listbox", { name: /file de triage/i })).toBeInTheDocument();
  });

  it("keeps the independent Classes panel visible if the review queue fails", async () => {
    apiMock.getAdminAnnotationQueue.mockRejectedValueOnce(new Error("storage unavailable"));
    render(<AdminAnnotationsPage initialTab="classes" />);
    const panel = await screen.findByRole("tabpanel", { name: /panneau classes/i });
    expect(await within(panel).findByRole("heading", { name: "Classes disponibles" })).toBeInTheDocument();
  });

  it("explains the legacy review import without calling it a stale annotation", async () => {
    const queue = queueWithStatuses("pending", "approved");
    queue.review_store = { mode: "legacy_readonly", legacy_decisions: 1 };
    apiMock.getAdminAnnotationQueue.mockResolvedValue(queue);
    render(<AdminAnnotationsPage />);

    expect(await screen.findByText(/anciennes validations à importer/i)).toBeInTheDocument();
    expect(screen.getByText(/1 validation conservée/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /valider/i })).toBeDisabled();
  });

  it("groups classes by action and keeps active-model classes collapsed", async () => {
    apiMock.getAdminAnnotationQueue.mockResolvedValue(queueWithStatuses("pending", "rejected"));
    apiMock.getAdminClasses.mockResolvedValue({
      revision: "a".repeat(64),
      classes: [
        { class_name: "atl", class_label: 2, status: "active", counts: { pending: 0, approved: 1, rejected: 0 }, trainable_count: 1 },
        { class_name: "tzapotl", class_label: null, status: "unconfirmed", counts: { pending: 1, approved: 1, rejected: 0 }, trainable_count: 0 },
        { class_name: "maya", class_label: 9, status: "candidate", counts: { pending: 0, approved: 1, rejected: 0 }, trainable_count: 1 },
      ],
    });
    render(<AdminAnnotationsPage initialTab="classes" />);

    expect(await screen.findByRole("heading", { name: /à confirmer/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /nouvelles confirmées/i })).toBeInTheDocument();
    expect(screen.getByText("tzapotl")).toBeInTheDocument();
    expect(screen.getByText("maya")).toBeInTheDocument();
    expect(screen.getByText(/classes du modèle actif/i).closest("details")).not.toHaveAttribute("open");
    expect(await screen.findByRole("region", { name: "Classes illustrées" })).toHaveTextContent("atl");
    expect(await screen.findByAltText("Exemple 1 de atl")).toHaveAttribute(
      "src", "http://api.test/admin/annotations/analysis-1/0/crop",
    );
  });

  it("illustrates pending class reviews with a pending crop, not an already approved crop", async () => {
    const queue = queueWithStatuses("approved", "pending");
    queue.analyses[0].elements[1].class_name = "atl";
    apiMock.getAdminAnnotationQueue.mockResolvedValue(queue);
    apiMock.getAdminClasses.mockResolvedValue({
      revision: "a".repeat(64),
      classes: [{ class_name: "atl", class_label: 2, status: "active", counts: { pending: 1, approved: 1, rejected: 0 }, trainable_count: 1 }],
    });

    render(<AdminAnnotationsPage initialTab="classes" />);

    expect(await within(await screen.findByRole("region", { name: "Classes illustrées" })).findByRole("img", { name: "Exemple de atl" }))
      .toHaveAttribute("src", "http://api.test/admin/annotations/analysis-1/1/crop");
  });

  it("does not invent split metrics when Training metadata is incomplete", async () => {
    const user = userEvent.setup();
    const incompleteSummary = trainingSummary();
    delete (incompleteSummary.data as Partial<typeof incompleteSummary.data>)
      .split_counts;
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(incompleteSummary);

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));

    expect(await screen.findByRole("heading", { name: /créer un candidat/i })).toBeInTheDocument();
    expect(
      screen.queryByText(/split locked: train 0/i),
    ).not.toBeInTheDocument();
  });

  it("keeps technical Training options behind an advanced disclosure", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    const summary = trainingSummary({
      training_jobs_enabled: true,
      launch_allowed_for_request: true,
      launch_disabled_reasons: [],
    });
    summary.training_snapshot.new_classes = ["teocomitl", "tzapotl"];
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(
      summary,
    );

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));

    expect(await screen.findByRole("button", { name: /vérifier la préparation/i })).toBeEnabled();
    expect(screen.getByText("Nouvelles classes incluses : teocomitl, tzapotl")).toBeInTheDocument();
    expect(screen.getByText(/options avancées/i)).toBeInTheDocument();
    await user.click(screen.getByText(/options avancées/i));
    await user.selectOptions(screen.getByLabelText(/machine/i), "cpu");
    await user.clear(screen.getByLabelText(/taille de lot/i));
    await user.type(screen.getByLabelText(/taille de lot/i), "8");
    expect(screen.getByLabelText(/machine/i)).toHaveValue("cpu");
    expect(screen.getByLabelText(/taille de lot/i)).toHaveValue(8);
    expect(screen.getByRole("button", { name: /vérifier la préparation/i })).toBeEnabled();
  });

  it("shows candidate results with training and held-out counts without claiming generalization", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses("approved", "rejected"));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary({
      training_jobs_enabled: true,
      launch_allowed_for_request: true,
      launch_disabled_reasons: [],
      latest_job: trainingJob({
        kind: "training", status: "succeeded", dry_run: false, model_version_id: "candidate-3",
        result: {
          unique_count: 5, duplicate_count: 0, updated_classes: ["atl"],
          base_correct: 1, active_correct: 1, candidate_correct: 3,
          train_count: 3, holdout: { support: 2, base_correct: 1, candidate_correct: 1 },
          generalization_validated: false,
        } as AdminTrainingJob["result"],
      }),
    }));

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));
    expect(await screen.findByText(/apprentissage.*actuel 1\/3.*candidat 3\/3/i)).toBeVisible();
    expect(screen.getByText(/test réservé.*actuel 1\/2.*candidat 1\/2/i)).toBeVisible();
    expect(screen.getByText(/généralisation non validée/i)).toBeVisible();
  });

  it("keeps candidate results visible after the latest job becomes a comparison", async () => {
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses("approved", "rejected"));
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(trainingSummary({
      latest_job: trainingJob({ kind: "comparison", status: "succeeded", dry_run: false }),
      latest_training_job: trainingJob({
        kind: "training", status: "succeeded", dry_run: false, model_version_id: "candidate-3",
        result: { train_count: 3, active_correct: 1, candidate_correct: 3 } as AdminTrainingJob["result"],
      }),
    }));
    render(<AdminAnnotationsPage />);
    await userEvent.click(await screen.findByRole("tab", { name: /entraîner/i }));
    expect(await screen.findByText(/apprentissage.*actuel 1\/3.*candidat 3\/3/i)).toBeVisible();
  });

  it("explains a backend training refusal instead of showing only a generic launch error", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(queueWithStatuses("approved", "rejected"));
    apiMock.getAdminTrainingSummary.mockResolvedValue(trainingSummary({
      training_jobs_enabled: true,
      launch_allowed_for_request: true,
      launch_disabled_reasons: [],
    }));
    apiMock.startAdminTrainingJob.mockRejectedValueOnce({
      isAxiosError: true,
      response: { status: 403, data: { error: "legacy_reviews_require_import" } },
    });
    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));
    await user.click(await screen.findByRole("button", { name: /vérifier la préparation/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/anciennes validations doivent être importées/i);
  });

  it("validates Training launch inputs before calling the backend", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(
      trainingSummary({
        training_jobs_enabled: true,
        launch_allowed_for_request: true,
        launch_disabled_reasons: [],
      }),
    );

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));

    await screen.findByRole("button", { name: /vérifier la préparation/i });
    await user.click(screen.getByText(/options avancées/i));
    await user.clear(screen.getByLabelText(/taille de lot/i));
    await user.type(screen.getByLabelText(/taille de lot/i), "999");
    await user.click(
      screen.getByRole("button", { name: /vérifier la préparation/i }),
    );

    expect(
      await screen.findByText(
        /taille de lot : entier entre 1 et 256/i,
      ),
    ).toBeInTheDocument();
    expect(apiMock.startAdminTrainingJob).not.toHaveBeenCalled();

  }, 15_000);

  it("explains when local retraining is blocked because no element is trainable", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("pending", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValueOnce(
      trainingSummary({
        training_jobs_enabled: true,
        launch_allowed_for_request: false,
        launch_disabled_reasons: [
          "no_trainable_annotations: approve at least one current annotation before launching retraining",
        ],
        data: {
          ...trainingSummary().data,
          approved: 0,
          trainable: 0,
          classes: [],
          per_class: {},
          split_counts: { train: 0, val: 0, test: 0, excluded: 4 },
        },
      }),
    );

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));

    expect(
      await screen.findByText(
        /validez au moins une annotation utilisable dans Trier/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /vérifier la préparation/i }),
    ).toBeDisabled();
  });

  it("starts a guarded dry-run training job and hides log tail behind support details", async () => {
    const user = userEvent.setup();
    const started = trainingJob({
      run_id: "run-started",
      status: "running",
      command: ["bash", "scripts/retrain.sh", "--dry-run"],
      log_tail: ["mock dry run started"],
    });
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("approved", "rejected"),
    );
    apiMock.getAdminTrainingSummary.mockResolvedValue(
      trainingSummary({
        training_jobs_enabled: true,
        launch_allowed_for_request: true,
        launch_disabled_reasons: [],
      }),
    );
    apiMock.startAdminTrainingJob.mockResolvedValueOnce({
      status: "ok",
      local_only: true,
      job: started,
    });

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("tab", { name: /entraîner/i }));
    await screen.findByRole("button", { name: /vérifier la préparation/i });
    await user.click(screen.getByText(/options avancées/i));
    await user.selectOptions(screen.getByLabelText(/machine/i), "cpu");
    await user.clear(screen.getByLabelText(/taille de lot/i));
    await user.type(screen.getByLabelText(/taille de lot/i), "8");
    await user.type(screen.getByLabelText(/notes/i), "smoke");

    await user.click(
      screen.getByRole("button", { name: /vérifier la préparation/i }),
    );

    await waitFor(() => {
      expect(apiMock.startAdminTrainingJob).toHaveBeenCalledWith({
        dry_run: true,
        device: "cpu",
        batch_size: 8,
        notes: "smoke",
        expected_data_revision: "a".repeat(64),
      });
    });
    expect(await screen.findByText(/Vérification ·/i)).toBeInTheDocument();
    expect(screen.getByText(/mock dry run started/i)).not.toBeVisible();
    expect(
      screen.queryByText(/bash scripts\/retrain\.sh --dry-run/i),
    ).not.toBeInTheDocument();
    await user.click(screen.getByText(/détails techniques et journal/i));
    expect(screen.getByText(/mock dry run started/i)).toBeVisible();
    expect(
      screen.getByRole("button", { name: /en cours/i }),
    ).toBeDisabled();
  }, 15_000);

  it("approves an element, reloads the queue, and reports refresh-failure success separately", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queueWithStatuses("pending", "rejected"))
      .mockResolvedValueOnce(queueWithStatuses("approved", "pending"))
      .mockRejectedValueOnce(new Error("offline"));
    apiMock.setAdminAnnotationReviewStatus.mockResolvedValue({
      status: "ok",
      local_only: true,
      warning: "local only",
      element: queueWithStatuses("approved", "rejected").analyses[0]
        .elements[0],
    });

    render(<AdminAnnotationsPage />);
    await screen.findByRole("button", { name: /valider/i });

    await user.click(screen.getByRole("button", { name: /valider/i }));

    await waitFor(() => {
      expect(apiMock.setAdminAnnotationReviewStatus).toHaveBeenCalledWith(
        "analysis-1",
        0,
        "approved",
        0,
      );
    });
    const confirmation = await screen.findByText(/Validé · atl/i);
    expect(confirmation.closest('[role="status"]')).not.toBeNull();
    expect(
      await screen.findByLabelText(
        /élément sélectionné 0 atl : statut Validé/i,
      ),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Validés" }));
    const approvedList = screen.getByRole("listbox", { name: /file de triage/i });
    expect(within(approvedList).getAllByRole("option")).toHaveLength(1);
    expect(within(approvedList).getByRole("option", { name: /ouvrir l'élément 0 atl/i }))
      .toHaveAccessibleName(/statut Validé/i);

    await user.selectOptions(screen.getByRole("combobox", { name: /^statut$/i }), "pending");
    expect(screen.getByRole("heading", { name: /élément #1 · calli/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /rejeter/i }));
    expect(
      await screen.findByText(/mais la file n'a pas pu être actualisée/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/mais la file n'a pas pu être actualisée/i).closest('[role="alert"]'),
    ).not.toBeNull();
  });

  it("keeps the next pending row selected after validating within the pending filter", async () => {
    const user = userEvent.setup();
    const before = queueWithStatuses("pending", "pending");
    const after = queueWithStatuses("pending", "approved");
    const third = {
      ...before.analyses[0].elements[1],
      key: "analysis-1:2",
      index: 2,
      class_name: "bet",
    };
    before.analyses[0].elements.push(third);
    after.analyses[0].elements.push(third);
    before.counts.total = 3;
    before.counts.pending = 3;
    after.counts.total = 3;
    after.counts.pending = 2;
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(before)
      .mockResolvedValueOnce(after);
    apiMock.setAdminAnnotationReviewStatus.mockResolvedValue({ status: "ok" });

    render(<AdminAnnotationsPage />);
    await screen.findByRole("option", { name: /ouvrir l'élément 1 calli/i });
    await user.click(screen.getByRole("button", { name: "À vérifier" }));
    await user.click(screen.getByRole("option", { name: /ouvrir l'élément 1 calli/i }));
    await user.click(screen.getByRole("button", { name: /valider/i }));

    expect(
      await screen.findByRole("heading", { name: /élément #2 · bet/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: /ouvrir l'élément 2 bet/i }),
    ).toHaveAttribute("aria-selected", "true");
  });

  it("reports a restored revision as successful when only queue refresh fails", async () => {
    const user = userEvent.setup();
    const queue = queueWithStatuses("approved", "rejected");
    queue.analyses[0].elements[0].revision = 1;
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queue)
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(queue);
    apiMock.getAdminAnnotationHistory.mockResolvedValue({
      revision: 1,
      history: [{ revision: 0, action: "created", status: "pending", class_name: "atl", bbox: [0, 1, 2, 3] }],
    });
    apiMock.restoreAdminAnnotationElement.mockResolvedValue({ status: "ok" });

    render(<AdminAnnotationsPage />);
    await screen.findByRole("listbox", { name: /file de triage/i });
    await user.click(screen.getByText("Historique des corrections"));
    await user.click(await screen.findByRole("button", { name: "Restaurer" }));

    const warning = await screen.findByText(/Version 0 restaurée.*file n'a pas pu être actualisée/i);
    expect(warning.closest('[role="alert"]')).not.toBeNull();
    await user.click(screen.getByRole("button", { name: /actualiser/i }));
    await waitFor(() => expect(screen.queryByText(/Version 0 restaurée.*file n'a pas pu être actualisée/i)).not.toBeInTheDocument());
    expect(screen.queryByText(/Impossible de restaurer cette version/i)).not.toBeInTheDocument();
  });

  it("pauses automatic refresh while correction is open and saves through modify endpoint", async () => {
    let autoRefresh: (() => void) | undefined;
    const intervalId = 1 as unknown as ReturnType<typeof window.setInterval>;
    const setIntervalSpy = vi
      .spyOn(window, "setInterval")
      .mockImplementation((handler: TimerHandler) => {
        autoRefresh = handler as () => void;
        return intervalId;
      });
    const clearIntervalSpy = vi
      .spyOn(window, "clearInterval")
      .mockImplementation(() => undefined);
    const user = userEvent.setup();
    const initial = queueWithStatuses("pending", "rejected");
    const updated = queueWithStatuses("pending", "rejected");
    updated.analyses[0].elements[0] = {
      ...updated.analyses[0].elements[0],
      class_name: "new-atl",
      bbox: [1, 2, 5, 6],
      trainable: false,
    };
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(initial)
      .mockResolvedValueOnce(updated)
      .mockResolvedValue(updated);
    apiMock.modifyAdminAnnotationElement.mockResolvedValueOnce({
      status: "ok",
      local_only: true,
      warning: "local only",
      element: updated.analyses[0].elements[0],
    });

    try {
      render(<AdminAnnotationsPage />);
      const naturalWidthSpy = vi
        .spyOn(HTMLImageElement.prototype, "naturalWidth", "get")
        .mockReturnValue(100);
      const naturalHeightSpy = vi
        .spyOn(HTMLImageElement.prototype, "naturalHeight", "get")
        .mockReturnValue(100);
      const rectSpy = vi
        .spyOn(SVGElement.prototype, "getBoundingClientRect")
        .mockReturnValue({
          x: 0,
          y: 0,
          left: 0,
          top: 0,
          right: 100,
          bottom: 100,
          width: 100,
          height: 100,
          toJSON: () => ({}),
        } as DOMRect);
      Object.defineProperty(SVGElement.prototype, "setPointerCapture", {
        configurable: true,
        value: vi.fn(),
      });
      Object.defineProperty(SVGElement.prototype, "releasePointerCapture", {
        configurable: true,
        value: vi.fn(),
      });
      Object.defineProperty(SVGElement.prototype, "hasPointerCapture", {
        configurable: true,
        value: vi.fn(() => true),
      });

      await user.click(
        await screen.findByRole("button", { name: /corriger/i }),
      );
      expect(screen.getByLabelText(/nom de l'élément/i)).toBeInTheDocument();
      expect(
        screen.getByLabelText(/contexte de l'annotation sélectionnée/i),
      ).toHaveTextContent(/file 1 \/ 2/i);
      const refreshButton = screen.getByRole("button", { name: /actualiser/i });
      expect(refreshButton).toBeDisabled();
      await act(async () => {
        autoRefresh?.();
      });
      expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(1);

      await act(async () => {
        fireEvent.load(
          screen.getByRole("img", { name: /image à corriger analysis-1/i }),
        );
      });
      const segmentationStage = screen.getByTestId("admin-segmentation-stage");
      expect(segmentationStage.style.aspectRatio).toBe("");
      expect(segmentationStage.style.maxWidth).toBe("");
      expect(
        screen.getByTestId("admin-correction-image-shell"),
      ).toBeInTheDocument();
      const currentCrop = screen.getByRole("img", {
        name: /découpe actuelle 0 pour atl/i,
      });
      const unchangedPreview = screen.getByRole("img", {
        name: /aperçu corrigé de l'élément 0/i,
      });
      expect(unchangedPreview.tagName).toBe("IMG");
      expect(unchangedPreview).toHaveAttribute(
        "src",
        currentCrop.getAttribute("src"),
      );
      expect(currentCrop.getAttribute("src")).toContain("v=");
      expect(
        screen.getByLabelText(/résumé des modifications/i),
      ).toHaveTextContent(/\[0, 1, 2, 3\] · inchangée/i);
      const visualEditor = await screen.findByLabelText(
        /redessiner la segmentation de l'élément 0/i,
      );
      const dispatchPointer = (
        type: string,
        clientX: number,
        clientY: number,
      ) => {
        const event = new Event(type, { bubbles: true, cancelable: true });
        Object.defineProperties(event, {
          clientX: { value: clientX },
          clientY: { value: clientY },
          pointerId: { value: 1 },
        });
        fireEvent(visualEditor, event);
      };
      dispatchPointer("pointerdown", 10, 20);
      dispatchPointer("pointermove", 60, 80);
      dispatchPointer("pointerup", 60, 80);
      expect(screen.getByLabelText(/zone x pour l'élément 0/i)).toHaveValue(10);
      expect(screen.getByLabelText(/zone y pour l'élément 0/i)).toHaveValue(20);
      expect(screen.getByLabelText(/zone w pour l'élément 0/i)).toHaveValue(50);
      expect(screen.getByLabelText(/zone h pour l'élément 0/i)).toHaveValue(60);
      const changedPreview = screen.getByRole("img", {
        name: /aperçu corrigé de l'élément 0/i,
      });
      expect(changedPreview.tagName).toBe("svg");
      expect(changedPreview).toHaveAttribute("viewBox", "10 20 50 60");
      expect(
        screen.getByTestId("admin-correction-preview-clip"),
      ).toHaveAttribute("x", "10");
      expect(
        screen.getByTestId("admin-correction-preview-clip"),
      ).toHaveAttribute("y", "20");
      expect(
        screen.getByTestId("admin-correction-preview-clip"),
      ).toHaveAttribute("width", "50");
      expect(
        screen.getByTestId("admin-correction-preview-clip"),
      ).toHaveAttribute("height", "60");
      expect(
        screen.getByLabelText(/résumé des modifications/i),
      ).toHaveTextContent(/\[0, 1, 2, 3\] → \[10, 20, 50, 60\]/i);

      await user.clear(screen.getByLabelText(/nom de l'élément/i));
      await user.type(screen.getByLabelText(/nom de l'élément/i), "new-atl");
      await user.click(screen.getByRole("button", { name: /^enregistrer$/i }));

      await waitFor(() => {
        expect(apiMock.modifyAdminAnnotationElement).toHaveBeenCalledWith(
          "analysis-1",
          0,
          {
            class_name: "new-atl",
            bbox: [10, 20, 50, 60],
            approve_after_save: undefined,
            expected_revision: 0,
            note: "",
          },
        );
      });
      naturalWidthSpy.mockRestore();
      naturalHeightSpy.mockRestore();
      rectSpy.mockRestore();
      expect(
        await screen.findByRole("heading", { name: /élément #0 · new-atl/i }),
      ).toBeInTheDocument();
      expect(screen.getByLabelText(/nom de l'élément/i)).toHaveValue("new-atl");
      expect(screen.getByLabelText(/zone x pour l'élément 0/i)).toHaveValue(1);
      expect(screen.getByLabelText(/zone y pour l'élément 0/i)).toHaveValue(2);
      expect(screen.getByLabelText(/zone w pour l'élément 0/i)).toHaveValue(5);
      expect(screen.getByLabelText(/zone h pour l'élément 0/i)).toHaveValue(6);
      expect(
        screen.getByLabelText(/contexte de l'annotation sélectionnée/i),
      ).toHaveTextContent(/file 1 \/ 2/i);
      expect(
        screen.getByRole("option", {
          name: /ouvrir l'élément 0 new-atl/i,
        }),
      ).toHaveAttribute("aria-selected", "true");
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  }, 15_000);

  it("saves an edit and marks it approved when using Enregistrer et valider", async () => {
    const user = userEvent.setup();
    const refreshed = queueWithStatuses("approved", "rejected");
    apiMock.getAdminAnnotationQueue
      .mockResolvedValueOnce(queueWithStatuses("pending", "rejected"))
      .mockResolvedValue(refreshed);
    apiMock.modifyAdminAnnotationElement.mockResolvedValueOnce({
      status: "ok",
      local_only: true,
      warning: "local only",
      element: refreshed.analyses[0].elements[0],
    });

    render(<AdminAnnotationsPage />);
    await screen.findByRole("button", { name: /corriger/i });

    await user.click(screen.getByRole("button", { name: /corriger/i }));
    await user.click(
      screen.getByRole("button", { name: /enregistrer et valider/i }),
    );

    await waitFor(() => {
      expect(apiMock.modifyAdminAnnotationElement).toHaveBeenCalledWith(
        "analysis-1",
        0,
        {
          class_name: "atl",
          bbox: [0, 1, 2, 3],
          approve_after_save: true,
          expected_revision: 0,
          note: "",
        },
      );
    });
    expect(
      await screen.findByText(/élément 0 enregistré et validé/i),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Validés" }));
    expect(screen.getByRole("option", { name: /ouvrir l'élément 0 atl/i }))
      .toHaveAccessibleName(/statut Validé/i);
  });

  it("keeps a correction dirty when its already selected row is clicked again", async () => {
    const user = userEvent.setup();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const logout = vi.fn();
    apiMock.getAdminAnnotationQueue.mockResolvedValue(queueWithStatuses("pending", "rejected"));
    render(<AdminAnnotationsPage authSlot={<button type="button" data-auth-logout onClick={logout}>Déconnexion</button>} />);
    try {
      await user.click(await screen.findByRole("button", { name: /corriger/i }));
      await user.clear(screen.getByLabelText(/nom de l'élément/i));
      await user.type(screen.getByLabelText(/nom de l'élément/i), "new-atl");
      await user.click(screen.getByRole("option", { name: /ouvrir l'élément 0 atl/i }));
      await user.click(screen.getByRole("button", { name: "Déconnexion" }));
      expect(logout).not.toHaveBeenCalled();
      await user.click(screen.getByRole("tab", { name: /dataset/i }));
      expect(confirm).toHaveBeenCalled();
      expect(screen.getByRole("tab", { name: /trier/i })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByLabelText(/nom de l'élément/i)).toHaveValue("new-atl");
    } finally {
      confirm.mockRestore();
    }
  });

  it("shows a visible error when editing fails to save", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("pending", "rejected"),
    );
    apiMock.modifyAdminAnnotationElement.mockRejectedValueOnce(
      new Error("offline"),
    );

    render(<AdminAnnotationsPage />);
    await screen.findByRole("button", { name: /corriger/i });

    await user.click(screen.getByRole("button", { name: /corriger/i }));
    await user.click(screen.getByRole("button", { name: /^enregistrer$/i }));

    expect(
      await screen.findByText(
        /impossible d'enregistrer les changements de l'élément 0/i,
      ),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/nom de l'élément/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /^enregistrer$/i }),
    ).toBeEnabled();
  });
  it("preserves visible status and shows an error when a triage mutation fails", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValueOnce(
      queueWithStatuses("pending", "pending"),
    );
    apiMock.setAdminAnnotationReviewStatus.mockRejectedValueOnce(
      new Error("offline"),
    );

    render(<AdminAnnotationsPage />);
    await screen.findByRole("button", { name: /rejeter/i });

    await user.click(screen.getByRole("button", { name: /rejeter/i }));

    expect(
      await screen.findByText(
        /impossible de marquer l'élément 0 comme Rejeté/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getAllByLabelText(/statut À vérifier/i).length,
    ).toBeGreaterThan(1);
    expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(1);
  });

  it("keeps the conflict explanation visible after refreshing a stale decision", async () => {
    const user = userEvent.setup();
    apiMock.getAdminAnnotationQueue.mockResolvedValue(queueWithStatuses("pending", "pending"));
    apiMock.setAdminAnnotationReviewStatus.mockRejectedValueOnce({
      isAxiosError: true,
      response: { status: 409, data: { error_code: "ANNOTATION_CONFLICT" } },
    });

    render(<AdminAnnotationsPage />);
    await user.click(await screen.findByRole("button", { name: /rejeter/i }));

    await waitFor(() => expect(apiMock.getAdminAnnotationQueue).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("alert")).toHaveTextContent(/cet élément a changé depuis son chargement/i);
  });
});
