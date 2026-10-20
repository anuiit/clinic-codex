import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AdminTrainingJob } from "../../types";
import { TrainingJobPanel } from "./TrainingHelpers";

describe("TrainingJobPanel", () => {
  it("separates training, reserved images, duplicates and label conflicts", () => {
    const job = {
      run_id: "real-corpus", status: "succeeded", dry_run: false,
      device: "cpu", batch_size: 16,
      result: {
        unique_count: 486, duplicate_count: 21, conflict_count: 2,
        train_count: 434, updated_classes: ["atl"],
        base_correct: 206, active_correct: 206, candidate_correct: 226,
        holdout: { support: 52, base_correct: 24, candidate_correct: 24 },
        generalization_validated: false,
      },
    } as AdminTrainingJob;
    render(<TrainingJobPanel job={job} />);
    expect(screen.getByText(/486 images uniques.*21 doublons et 2 annotations contradictoires exclus/)).toBeInTheDocument();
    expect(screen.getByText(/434 images apprises : base 206\/434, candidat 226\/434/)).toBeInTheDocument();
    expect(screen.getByText(/52 images réservées : actif 24\/52, candidat 24\/52/)).toBeInTheDocument();
  });
});
