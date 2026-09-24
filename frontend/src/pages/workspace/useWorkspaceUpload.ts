import { useCallback, useRef, useState, type DragEvent } from "react";
import { segmentGlyph } from "../../services/api";
import { saveAnalysis } from "../../services/storage";
import type { AnalysisRecord } from "../../types";

type UseWorkspaceUploadOptions = {
  apiErrorLabel: string;
  syncRecords: (preferredId?: string | null) => Promise<unknown>;
};

export function useWorkspaceUpload({
  apiErrorLabel,
  syncRecords,
}: UseWorkspaceUploadOptions) {
  const inputRef = useRef<HTMLInputElement>(null);
  const currentFileRef = useRef<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback((nextFile: File) => {
    currentFileRef.current = nextFile;
    setFile(nextFile);
    setPreview(null);
    setError(null);

    const reader = new FileReader();
    const readError = () => {
      if (currentFileRef.current !== nextFile) return;
      currentFileRef.current = null;
      setFile(null);
      setPreview(null);
      setError("Lecture de l'image impossible.");
      if (inputRef.current) inputRef.current.value = "";
    };
    reader.onload = (event) => {
      if (currentFileRef.current === nextFile) {
        const result = event.target?.result;
        if (typeof result === "string") setPreview(result);
        else readError();
      }
    };
    reader.onerror = readError;
    reader.onabort = readError;
    try {
      reader.readAsDataURL(nextFile);
    } catch {
      readError();
    }
  }, []);

  const clearPendingFile = useCallback(() => {
    currentFileRef.current = null;
    setFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragging(false);
      const nextFile = event.dataTransfer.files[0];
      if (nextFile) {
        handleFile(nextFile);
      }
    },
    [handleFile],
  );

  const saveAnalyzedFile = async (source: File, imageDataUrl: string) => {
    const result = await segmentGlyph(source);
    const record: AnalysisRecord = {
      id: crypto.randomUUID(),
      imageName: source.name,
      imageDataUrl,
      timestamp: Date.now(),
      result,
      annotations: {},
    };
    await saveAnalysis(record);
    await syncRecords(record.id);
  };

  const analyze = async () => {
    if (!file || !preview) return;
    setLoading(true);
    setError(null);
    try {
      await saveAnalyzedFile(file, preview);
      clearPendingFile();
    } catch (issue) {
      setError(issue instanceof Error ? issue.message : apiErrorLabel);
    } finally {
      setLoading(false);
    }
  };

  const analyzeExample = async (url: string, name: string) => {
    if (loading) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error("Image d'exemple indisponible.");
      const blob = await response.blob();
      const exampleFile = new File([blob], name, { type: blob.type || "image/jpeg" });
      const imageDataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(new Error("Lecture de l'image impossible."));
        reader.readAsDataURL(exampleFile);
      });
      await saveAnalyzedFile(exampleFile, imageDataUrl);
    } catch (issue) {
      setError(issue instanceof Error ? issue.message : apiErrorLabel);
    } finally {
      setLoading(false);
    }
  };

  return {
    inputRef,
    dragging,
    file,
    preview,
    loading,
    error,
    setDragging,
    handleFile,
    clearPendingFile,
    onDrop,
    analyze,
    analyzeExample,
  };
}
