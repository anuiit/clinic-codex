import axios from 'axios';
import type { SegmentResult, ClassifyResult, ClassesResult, SimilarResult, TrustResult, SaveAnnotationPayload, SaveAnnotationResponse, SaveAnnotationResult } from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:7117';

export async function segmentGlyph(file: File): Promise<SegmentResult> {
  const form = new FormData();
  form.append('image', file);
  const { data } = await axios.post<SegmentResult>(`${BASE_URL}/segment`, form);
  return data;
}

export async function classifyElement(file: File): Promise<ClassifyResult> {
  const form = new FormData();
  form.append('image', file);
  const { data } = await axios.post<ClassifyResult>(`${BASE_URL}/classify`, form);
  return data;
}

export async function getClasses(): Promise<ClassesResult> {
  const { data } = await axios.get<ClassesResult>(`${BASE_URL}/classes`);
  return data;
}

export async function getSimilar(imageDataUrl: string, bbox: [number, number, number, number], limit = 5): Promise<SimilarResult> {
  const base64 = imageDataUrl.split(',')[1];
  const { data } = await axios.post<SimilarResult>(`${BASE_URL}/similar`, {
    image_base64: base64,
    bbox,
    limit,
    mode: 'prototype',
  });
  return data;
}

export async function getTrust(imageDataUrl: string, bbox: [number, number, number, number], predictedClass: string, topK = 10): Promise<TrustResult> {
  const base64 = imageDataUrl.split(',')[1];
  const { data } = await axios.post<TrustResult>(`${BASE_URL}/trust`, {
    image_base64: base64,
    bbox,
    predicted_class: predictedClass,
    top_k: topK,
  });
  return data;
}

export async function saveAnnotation(
  payload: SaveAnnotationPayload
): Promise<SaveAnnotationResult> {
  try {
    const res = await fetch(`${BASE_URL}/save-annotation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => null) as Partial<{
        error_code: string;
        message: string;
        hint?: string;
        trace_id?: string;
      }> | null;

      return {
        ok: false,
        error_code: errorData?.error_code ?? 'NETWORK_ERROR',
        message: errorData?.message ?? `save-annotation failed: ${res.status}`,
        hint: errorData?.hint,
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
