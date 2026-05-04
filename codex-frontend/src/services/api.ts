import axios from 'axios';
import type { SegmentResult, ClassifyResult, ClassesResult } from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000';

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
