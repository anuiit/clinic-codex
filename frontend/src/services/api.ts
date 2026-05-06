import axios from 'axios';
import type { SegmentResult, ClassifyResult, ClassesResult, SimilarResult, TrustResult, SaveAnnotationPayload, SaveAnnotationResponse } from '../types';

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
): Promise<SaveAnnotationResponse> {
  const res = await fetch(`${BASE_URL}/save-annotation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`save-annotation failed: ${res.status} ${errText}`);
  }
  return res.json();
}
