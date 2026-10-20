import axios, { type AxiosRequestConfig } from 'axios';
import type {
  SegmentResult,
  ClassifyResult,
  ClassesResult,
  SimilarResult,
  TrustResult,
  SaveAnnotationPayload,
  SaveAnnotationResponse,
  SaveAnnotationResult,
  SaveAnnotationErrorCode,
  AdminAnnotationModifyPayload,
  AdminAnnotationHistory,
  AdminClassCatalogue,
  AdminAnnotationQueue,
  AdminAnnotationMutationResponse,
  AdminAnnotationReviewStatus,
  AdminTrainingJobResponse,
  AdminTrainingStartPayload,
  AdminTrainingSummary,
  AuthSession,
  BootstrapStatus,
  LoginPayload,
  RuntimeVersionInfo,
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:7117';
const CSRF_HEADER = 'X-CSRF-Token';

axios.defaults.withCredentials = true;
let csrfToken: string | null = null;

export interface ApiRequestOptions {
  signal?: AbortSignal;
}

type SaveAnnotationErrorBody = Partial<{
  error_code: SaveAnnotationErrorCode;
  message: string;
  error: string | { code?: SaveAnnotationErrorCode | string; message?: string };
  hint: string | null;
  trace_id: string;
}>;

function apiUrl(path: string): string {
  return `${BASE_URL}${path}`;
}

export function adminAnnotationMediaUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  return apiUrl(path);
}

function requestConfig(options?: ApiRequestOptions): AxiosRequestConfig | undefined {
  return options?.signal ? { signal: options.signal } : undefined;
}

async function postData<T>(path: string, payload: unknown, options?: ApiRequestOptions): Promise<T> {
  const config = requestConfig(options);
  const { data } = config
    ? await axios.post<T>(apiUrl(path), payload, config)
    : await axios.post<T>(apiUrl(path), payload);
  return data;
}

async function getData<T>(path: string, options?: ApiRequestOptions): Promise<T> {
  const config = requestConfig(options);
  const { data } = config
    ? await axios.get<T>(apiUrl(path), config)
    : await axios.get<T>(apiUrl(path));
  return data;
}

function storeSession(session: AuthSession): AuthSession {
  csrfToken = session.csrf_token ?? null;
  if (csrfToken) {
    axios.defaults.headers.common[CSRF_HEADER] = csrfToken;
  } else {
    delete axios.defaults.headers.common[CSRF_HEADER];
  }
  return session;
}

export async function getAuthSession(options?: ApiRequestOptions): Promise<AuthSession> {
  return storeSession(await getData<AuthSession>('/auth/me', options));
}

export async function getBootstrapStatus(options?: ApiRequestOptions): Promise<BootstrapStatus> {
  return getData<BootstrapStatus>('/auth/bootstrap/status', options);
}

export async function getRuntimeVersion(
  options?: ApiRequestOptions,
): Promise<RuntimeVersionInfo> {
  return getData<RuntimeVersionInfo>('/version', options);
}

export async function createFirstAdmin(
  payload: LoginPayload,
  options?: ApiRequestOptions,
): Promise<AuthSession> {
  return storeSession(await postData<AuthSession>('/auth/bootstrap', payload, options));
}

export async function login(payload: LoginPayload, options?: ApiRequestOptions): Promise<AuthSession> {
  return storeSession(await postData<AuthSession>('/auth/login', payload, options));
}

export async function logout(options?: ApiRequestOptions): Promise<void> {
  try {
    await postData('/auth/logout', {}, options);
  } finally {
    csrfToken = null;
    delete axios.defaults.headers.common[CSRF_HEADER];
  }
}

function imageForm(file: File): FormData {
  const form = new FormData();
  form.append('image', file);
  return form;
}

function dataUrlPayload(imageDataUrl: string): string | undefined {
  return imageDataUrl.split(',')[1];
}

export async function segmentGlyph(file: File, options?: ApiRequestOptions): Promise<SegmentResult> {
  return postData<SegmentResult>('/segment', imageForm(file), options);
}

export async function classifyElement(file: File, options?: ApiRequestOptions): Promise<ClassifyResult> {
  return postData<ClassifyResult>('/classify', imageForm(file), options);
}

export async function getClasses(options?: ApiRequestOptions): Promise<ClassesResult> {
  return getData<ClassesResult>('/classes', options);
}

export async function getAnnotationClasses(options?: ApiRequestOptions): Promise<ClassesResult> {
  return getData<ClassesResult>('/annotation-classes', options);
}

export async function getAdminClasses(options?: ApiRequestOptions): Promise<AdminClassCatalogue> {
  return requireAdminClasses(await getData<AdminClassCatalogue>('/admin/classes', options));
}

export async function confirmAdminClass(className: string, revision: string): Promise<AdminClassCatalogue> {
  return requireAdminClasses(await postData<AdminClassCatalogue>('/admin/classes', { class_name: className, expected_revision: revision }));
}

function requireAdminClasses(data: AdminClassCatalogue): AdminClassCatalogue {
  if (!Array.isArray(data?.classes) || typeof data.revision !== 'string' ||
      !data.classes.every((item) => item && typeof item.class_name === 'string' &&
        ['active', 'candidate', 'unconfirmed'].includes(item.status) &&
        (item.class_label === null || Number.isInteger(item.class_label)) &&
        item.counts && (['pending', 'approved', 'rejected'] as const)
          .every((key) => Number.isInteger(item.counts[key])) &&
        Number.isInteger(item.trainable_count))) {
    throw new Error('Invalid admin classes response');
  }
  return data;
}

export async function getSimilar(
  imageDataUrl: string,
  bbox: [number, number, number, number],
  limit = 5,
  options?: ApiRequestOptions,
): Promise<SimilarResult> {
  return postData<SimilarResult>(
    '/similar',
    {
      image_base64: dataUrlPayload(imageDataUrl),
      bbox,
      limit,
      mode: 'prototype',
    },
    options,
  );
}

export async function getTrust(
  imageDataUrl: string,
  bbox: [number, number, number, number],
  predictedClass: string,
  topK = 10,
  options?: ApiRequestOptions,
): Promise<TrustResult> {
  return postData<TrustResult>(
    '/trust',
    {
      image_base64: dataUrlPayload(imageDataUrl),
      bbox,
      predicted_class: predictedClass,
      top_k: topK,
    },
    options,
  );
}

export async function saveAnnotation(
  payload: SaveAnnotationPayload,
  options?: ApiRequestOptions,
): Promise<SaveAnnotationResult> {
  const requestInit: RequestInit = {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(csrfToken ? { [CSRF_HEADER]: csrfToken } : {}),
    },
    body: JSON.stringify(payload),
  };
  if (options?.signal) {
    requestInit.signal = options.signal;
  }

  try {
    const res = await fetch(apiUrl('/save-annotation'), requestInit);

    if (!res.ok) {
      const errorData = await res.json().catch(() => null) as SaveAnnotationErrorBody | null;
      const nestedError = typeof errorData?.error === 'object' ? errorData.error : null;
      const legacyMessage = typeof errorData?.error === 'string' ? errorData.error : nestedError?.message;
      const errorCode = errorData?.error_code
        ?? (nestedError?.code as SaveAnnotationErrorCode | undefined)
        ?? (legacyMessage ? 'VALIDATION_ERROR' : 'NETWORK_ERROR');

      return {
        ok: false,
        error_code: errorCode,
        message: errorData?.message ?? legacyMessage ?? `save-annotation failed: ${res.status}`,
        hint: errorData?.hint ?? undefined,
        trace_id: errorData?.trace_id,
      };
    }

    const data = await res.json() as SaveAnnotationResponse;
    return { ok: true, ...data };
  } catch {
    return {
      ok: false,
      error_code: 'NETWORK_ERROR',
      message: 'Network error while saving annotation',
    };
  }
}

export async function getAdminAnnotationQueue(
  options?: ApiRequestOptions,
): Promise<AdminAnnotationQueue> {
  const data = await getData<AdminAnnotationQueue>('/admin/annotations', options);
  if (!Array.isArray(data?.analyses) || !Array.isArray(data.diagnostics) ||
      !data.counts ||
      !(['total', 'pending', 'approved', 'rejected', 'trainable'] as const)
        .every((key) => Number.isFinite(data.counts[key])) ||
      !data.analyses.every((analysis) => analysis && typeof analysis.analysis_id === 'string' &&
        Array.isArray(analysis.elements) && analysis.elements.every((element) =>
          element && typeof element.key === 'string' && typeof element.class_name === 'string' &&
          Number.isInteger(element.index) && Number.isInteger(element.revision) &&
          Array.isArray(element.bbox) &&
          element.bbox.every(Number.isFinite) &&
          ['pending', 'approved', 'rejected'].includes(element.review_status) &&
          ['train', 'val', 'test', 'excluded'].includes(element.dataset_split) &&
          typeof element.trainable === 'boolean' && typeof element.crop_exists === 'boolean' &&
          typeof element.crop_url === 'string'))) {
    throw new Error('Invalid admin annotation queue response');
  }
  return data;
}

export async function setAdminAnnotationReviewStatus(
  analysisId: string,
  index: number,
  status: AdminAnnotationReviewStatus,
  expectedRevision: number,
  options?: ApiRequestOptions,
): Promise<AdminAnnotationMutationResponse> {
  return postData<AdminAnnotationMutationResponse>(
    `/admin/annotations/${encodeURIComponent(analysisId)}/${index}/review`,
    { status, expected_revision: expectedRevision },
    options,
  );
}

export async function getAdminAnnotationHistory(
  analysisId: string,
  index: number,
): Promise<AdminAnnotationHistory> {
  const data = await getData<AdminAnnotationHistory>(
    `/admin/annotations/${encodeURIComponent(analysisId)}/${index}/history`,
  );
  if (!Number.isInteger(data?.revision) || !Array.isArray(data.history) ||
      !data.history.every((entry) => entry && Number.isInteger(entry.revision) &&
        typeof entry.class_name === 'string' &&
        ['pending', 'approved', 'rejected'].includes(entry.status) &&
        Array.isArray(entry.bbox) && entry.bbox.length === 4 &&
        entry.bbox.every(Number.isFinite))) {
    throw new Error('Invalid admin annotation history response');
  }
  return data;
}

export async function restoreAdminAnnotationElement(
  analysisId: string,
  index: number,
  targetRevision: number,
  expectedRevision: number,
): Promise<AdminAnnotationMutationResponse> {
  return postData<AdminAnnotationMutationResponse>(
    `/admin/annotations/${encodeURIComponent(analysisId)}/${index}/restore`,
    { target_revision: targetRevision, expected_revision: expectedRevision },
  );
}

export async function modifyAdminAnnotationElement(
  analysisId: string,
  index: number,
  payload: AdminAnnotationModifyPayload,
  options?: ApiRequestOptions,
): Promise<AdminAnnotationMutationResponse> {
  return postData<AdminAnnotationMutationResponse>(
    `/admin/annotations/${encodeURIComponent(analysisId)}/${index}/modify`,
    payload,
    options,
  );
}

export async function getAdminTrainingSummary(
  options?: ApiRequestOptions,
): Promise<AdminTrainingSummary> {
  const data = await getData<AdminTrainingSummary>('/admin/training/summary', options);
  if (!data?.training_snapshot || !Array.isArray(data?.data?.classes) ||
      !data.data.classes.every((name) => typeof name === 'string') ||
      !data.data.split_counts || !(['train', 'val', 'test', 'excluded'] as const)
        .every((key) => Number.isFinite(data.data.split_counts[key])) ||
      !Array.isArray(data.launch_disabled_reasons) ||
      !data.parameters?.editable?.batch_size || !Array.isArray(data.parameters.editable.device) ||
      !data.parameters.editable.device.every((device) => typeof device === 'string') ||
      !(['default', 'min', 'max'] as const)
        .every((key) => Number.isInteger(data.parameters.editable.batch_size[key]))) {
    throw new Error('Invalid admin training summary response');
  }
  return data;
}

export async function getLatestAdminTrainingJob(
  options?: ApiRequestOptions,
): Promise<AdminTrainingJobResponse> {
  return getData<AdminTrainingJobResponse>('/admin/training/jobs/latest', options);
}

export async function getAdminTrainingJob(runId: string): Promise<AdminTrainingJobResponse> {
  return getData<AdminTrainingJobResponse>(`/admin/training/jobs/${encodeURIComponent(runId)}`);
}

export async function getComparableModels(): Promise<{ versions: Array<{ version_id: string; status: string; created_at?: string }> }> {
  return getData('/admin/training/models');
}

export async function startModelComparison(versionId: string, analysisId?: string): Promise<AdminTrainingJobResponse> {
  return postData('/admin/training/comparisons', {
    version_id: versionId,
    ...(analysisId ? { analysis_id: analysisId } : {}),
  });
}

export async function startAdminTrainingJob(
  payload: AdminTrainingStartPayload,
  options?: ApiRequestOptions,
): Promise<AdminTrainingJobResponse> {
  return postData<AdminTrainingJobResponse>('/admin/training/jobs', payload, options);
}
