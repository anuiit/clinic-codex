export interface TopKItem {
  class_name: string;
  confidence: number;
}

export interface ClassifyResult {
  class_name: string;
  class_label: number;
  confidence: number;
  rejected: boolean;
  top_k: TopKItem[];
}

export interface DetectedElement extends ClassifyResult {
  bbox: [number, number, number, number]; // [x, y, w, h]
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

export interface AnalysisRecord {
  id: string;
  imageName: string;
  imageDataUrl: string;
  timestamp: number;
  result: SegmentResult;
  annotations: Record<number, string>;
}
