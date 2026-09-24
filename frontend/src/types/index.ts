export type ClassLabel = string | number;

export interface RuntimeVersionInfo {
  app_name: string;
  app_version: string;
  model_version: string | null;
}

export interface AuthUser {
  id: string;
  email: string;
  role?: string;
  roles: string[];
  permissions: string[];
}

export interface AuthSession {
  auth_enabled: boolean;
  user: AuthUser | null;
  status?: string;
  csrf_token?: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface BootstrapStatus {
  status: 'ok';
  auth_enabled: boolean;
  bootstrap_available: boolean;
}

export interface TopKItem {
  class_name: string;
  class_label?: ClassLabel | null;
  confidence: number;
}

export interface ClassifyResult {
  class_name: string;
  class_label?: ClassLabel | null;
  confidence: number;
  rejected: boolean;
  top_k: TopKItem[];
}

export interface DetectedElement extends ClassifyResult {
  bbox: [number, number, number, number]; // [x, y, w, h]
  // Free-text research note. Annotator memo only — never exported to the
  // training dataset.
  note?: string;
}

export interface SegmentResult {
  num_elements: number;
  image_size: [number, number]; // [w, h]
  elements: DetectedElement[];
}

export interface ClassesResult {
  num_classes: number;
  class_names: string[];
}

export interface SimilarItem {
  rank: number;
  match_type: string;
  class_name: string;
  class_label: ClassLabel | null;
  similarity: number;
  band: 'high' | 'moderate' | 'low';
  asset: string | null;
}

export interface SimilarResult {
  query: { bbox: [number, number, number, number]; mode: string };
  best_match: { class_name: string; similarity: number; rejected: boolean };
  results: SimilarItem[];
}

export interface TrustSignals {
  predicted_class_rank: number;
  predicted_class_similarity: number;
  top1_class: string;
  top1_similarity: number;
  margin_to_second: number;
  above_rejection_threshold: boolean;
  rejection_threshold: number;
  ambiguous: boolean;
  entropy: number;
  top_k: TopKItem[];
}

export interface TrustResult {
  query: { bbox: [number, number, number, number]; predicted_class: string };
  trust: TrustSignals;
}

export type AnnotationStatus = 'draft' | 'validated';

export interface AnalysisRecord {
  id: string;
  imageName: string;
  imageDataUrl: string;
  timestamp: number;
  result: SegmentResult;
  annotations: Record<number, string>;
  annotationStatus?: Record<number, AnnotationStatus>;
}

export interface SaveAnnotationPayload {
  analysis_id: string;
  image_name: string;
  image_data_url: string;
  timestamp: number;
  annotations: Array<{
    index: number;
    bbox: [number, number, number, number];
    class_name: string;
    note?: string;
  }>;
}

export interface SaveAnnotationResponse {
  status: "ok" | "error";
  analysis_id: string;
  saved_count: number;
  classes: string[];
  saved_at?: string;
  error?: string;
}

export type SaveAnnotationErrorCode =
  | 'VALIDATION_ERROR'
  | 'ANNOTATION_CONFLICT'
  | 'PERMISSION_DENIED'
  | 'DISK_FULL'
  | 'STORAGE_ERROR'
  | 'INTERNAL_ERROR'
  | 'NETWORK_ERROR';

export interface SaveAnnotationError {
  ok: false;
  error_code: SaveAnnotationErrorCode;
  message: string;
  hint?: string;
  trace_id?: string;
}

export type SaveAnnotationSuccess = SaveAnnotationResponse & { ok: true };

export type SaveAnnotationResult = SaveAnnotationSuccess | SaveAnnotationError;

export type AdminAnnotationReviewStatus = 'pending' | 'approved' | 'rejected';
export type AdminDatasetSplit = 'train' | 'val' | 'test' | 'excluded';

export interface AdminAnnotationDiagnostic {
  code: string;
  message: string;
  analysis_id?: string;
  index?: number;
  key?: string;
}

export interface AdminAnnotationCounts {
  total: number;
  pending: number;
  approved: number;
  rejected: number;
  trainable: number;
}

export interface AdminAnnotationElement {
  key: string;
  revision: number;
  analysis_id: string;
  index: number;
  class_name: string;
  bbox: [number, number, number, number] | number[];
  note?: string | null;
  crop_path: string;
  crop_url: string;
  crop_exists: boolean;
  review_status: AdminAnnotationReviewStatus;
  trainable: boolean;
  dataset_split: AdminDatasetSplit;
  split_reason: string;
  source_fingerprint: string;
  stale_decision: boolean;
}

export interface AdminAnnotationAnalysis {
  analysis_id: string;
  image_name?: string | null;
  uploaded_at?: string;
  image_path: string;
  image_url: string;
  image_exists: boolean;
  elements: AdminAnnotationElement[];
}

export interface AdminAnnotationQueue {
  status: 'ok';
  schema_version: number;
  local_only: boolean;
  warning: string;
  review_store?: { mode: 'sqlite' | 'legacy_readonly' | 'empty'; legacy_decisions: number };
  counts: AdminAnnotationCounts;
  analyses: AdminAnnotationAnalysis[];
  diagnostics: AdminAnnotationDiagnostic[];
}

export interface AdminAnnotationMutationResponse {
  status: 'ok';
  local_only: boolean;
  warning: string;
  element: AdminAnnotationElement;
  counts?: AdminAnnotationCounts;
}

export interface AdminAnnotationModifyPayload {
  class_name: string;
  bbox: [number, number, number, number];
  expected_revision: number;
  note?: string | null;
  approve_after_save?: boolean;
  status?: AdminAnnotationReviewStatus;
}

export interface AdminAnnotationHistory {
  revision: number;
  history: Array<{
    revision: number;
    action: string;
    status: AdminAnnotationReviewStatus;
    class_name: string;
    bbox: number[];
    note?: string | null;
    reviewed_at?: string;
    reviewed_by?: string | null;
  }>;
}

export interface AdminClassCatalogue {
  revision: string;
  classes: Array<{
    class_name: string;
    class_label: number | null;
    status: 'active' | 'candidate' | 'unconfirmed';
    counts: { pending: number; approved: number; rejected: number };
    trainable_count: number;
  }>;
}

export interface AdminModelComparison {
  candidate_version_id: string;
  sample_count: number;
  warnings: string[];
  protocol: {
    evaluation_scope: string;
    metrics_scope: string;
    base_historical_independence: string;
    device: string;
    latency_scope?: string;
  };
  metrics: {
    common: { support: number; base_correct: number; candidate_correct: number; gains: number; regressions: number; unchanged: number; both_wrong: number };
    new_classes: { support: number; candidate_correct: number };
    coverage: { base: number; candidate: number };
    latency_ms: { base: number; candidate: number };
    by_scope: Record<string, Omit<AdminModelComparison['metrics'], 'by_scope' | 'latency_ms'>>;
    top3: { base_correct: number; candidate_correct: number };
    per_class: Record<string, { support: number; base_correct: number; candidate_correct: number; base_top3: number; candidate_top3: number }>;
  };
  rows: Array<{
    sample_id: string;
    analysis_id: string;
    index: number;
    image_name: string | null;
    bbox: number[];
      source_image_size?: [number, number] | null;
      source_image_sha256?: string;
      scope: 'train' | 'locked_test' | 'ad_hoc';
      review_status?: 'approved' | 'pending' | 'rejected' | null;
      candidate_exposure?: 'train' | 'locked_test' | 'unseen_exact' | 'unknown';
    expected_class: string | null;
    base_supported: boolean;
    base: ClassifyResult;
    candidate: ClassifyResult;
    outcome: 'gain' | 'regression' | 'both_wrong' | 'unchanged' | 'new_class' | 'disagreement';
    crop_url: string;
    source_image_url: string;
  }>;
}

export interface AdminTrainingFileInfo {
  path: string;
  exists: boolean;
  size?: number;
  mtime?: string;
  sha256?: string | null;
}

export interface AdminTrainingJob {
  run_id: string;
  kind?: 'training' | 'comparison';
  stage?: string;
  comparison?: AdminModelComparison;
  model_version_id?: string;
  candidate_version_dir?: string;
  error?: string;
  result?: {
    unique_count: number;
    duplicate_count: number;
    conflict_count?: number;
    updated_classes: string[];
    base_correct: number;
    active_correct: number;
    candidate_correct: number;
    train_count?: number;
    holdout?: { support: number; base_correct: number; candidate_correct: number };
    generalization_validated: false;
  };
  status: 'running' | 'succeeded' | 'failed' | 'disabled' | 'rejected';
  local_only?: boolean;
  dry_run: boolean;
  device: string;
  batch_size: number;
  notes?: string;
  started_at?: string;
  finished_at?: string | null;
  exit_code?: number | null;
  pid?: number | null;
  process_identity?: string | null;
  command?: string[];
  cwd?: string;
  env?: Record<string, string>;
  log_path?: string;
  log_tail?: string[];
  artifacts?: Record<string, unknown>;
  training_snapshot?: AdminTrainingSnapshot;
  training_snapshot_manifest_hash?: string | null;
}

export interface AdminTrainingSnapshot {
  data_revision?: string;
  new_classes?: string[];
  mode?: 'local_prior';
  configured: boolean;
  valid: boolean;
  snapshot_id: string | null;
  snapshot_manifest_sha256: string | null;
  row_count: number | null;
  class_count: number | null;
  live_annotation_count: number | null;
  live_train_count?: number | null;
  live_split_counts?: Record<"train" | "dev" | "locked_test" | "excluded", number> | null;
  live_annotations_sha256?: string | null;
  ready_for_training: boolean;
  promotion_evaluation_ready: boolean;
  split_counts: Record<"train" | "dev" | "locked_test", number> | null;
  paths: Record<string, string | null>;
  errors: string[];
}

export interface AdminTrainingSummary {
  status: 'ok';
  local_only: boolean;
  warning: string;
  training_jobs_enabled: boolean;
  launch_allowed_for_request: boolean;
  launch_disabled_reasons: string[];
  data: {
    total: number;
    pending: number;
    approved: number;
    rejected: number;
    trainable: number;
    classes: string[];
    per_class: Record<string, number>;
    split_counts: Record<AdminDatasetSplit, number>;
    diagnostics: AdminAnnotationDiagnostic[];
  };
  parameters: {
    editable: {
      dry_run: boolean;
      device: string[];
      batch_size: { default: number; min: number; max: number };
    };
    script_env_defaults: Record<string, string>;
    config: Record<string, unknown>;
  };
  paths: Record<string, string | boolean | null>;
  artifacts: Record<string, unknown>;
  training_snapshot: AdminTrainingSnapshot;
  latest_job?: AdminTrainingJob | null;
  latest_training_job?: AdminTrainingJob | null;
}

export interface AdminTrainingJobResponse {
  status: 'ok';
  local_only: boolean;
  job: AdminTrainingJob | null;
}

export interface AdminTrainingStartPayload {
  dry_run: boolean;
  expected_data_revision?: string;
  device: string;
  batch_size: number;
  notes?: string;
}
