import { Link, useParams, useSearchParams } from "react-router";
import { ArrowLeft, Loader2 } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { ImageBBoxStage } from "../components/ImageBBoxStage";
import { UnsavedNavigationPrompt } from "../components/UnsavedNavigationPrompt";
import type { ThemeMode } from "../components/ThemeToggle";
import { appText } from "../i18n/text";
import { AnnotationAnalyzerToolbar } from "./AnnotationAnalyzerToolbar";
import { AnnotationElementList } from "./AnnotationElementList";
import { AnnotationPageChrome } from "./AnnotationPageChrome";
import { AnnotationSelectedInspector } from "./annotation/AnnotationSelectedInspector";
import { AnnotationToast } from "./annotation/AnnotationToast";
import annotationStyles from "./annotation/AnnotationChrome.module.css";
import { formatBboxLabel } from "./annotation/annotationUtils";
import { RESIZE_HANDLE_VISUAL_SIZE } from "./annotation/useBBoxEditing";
import { useAnnotationElementModel } from "./annotation/useAnnotationElementModel";
import { useAnnotationRecord } from "./annotation/useAnnotationRecord";
import { useAnnotationSubmission } from "./annotation/useAnnotationSubmission";
import { useAnnotationViewport } from "./annotation/useAnnotationViewport";

const RESIZE_HANDLE_OFFSET = RESIZE_HANDLE_VISUAL_SIZE / 2;

function resizeHandle(
  key: string,
  x: number,
  y: number,
  className: "cursor-nwse-resize" | "cursor-nesw-resize",
) {
  return (
    <rect
      key={key}
      x={x - RESIZE_HANDLE_OFFSET}
      y={y - RESIZE_HANDLE_OFFSET}
      width={RESIZE_HANDLE_VISUAL_SIZE}
      height={RESIZE_HANDLE_VISUAL_SIZE}
      rx={3}
      strokeWidth={1.5}
      className={`annotation-resize-handle ${className}`}
    />
  );
}

type AnnotationPageProps = {
  themeMode?: ThemeMode;
  onToggleTheme?: () => void;
  authSlot?: ReactNode;
  guardRouteTransitions?: boolean;
};

export default function AnnotationPage({
  themeMode = "dark",
  onToggleTheme = () => undefined,
  authSlot,
  guardRouteTransitions = false,
}: AnnotationPageProps = {}) {
  const allowLeaveRef = useRef(false);
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const t = appText.annotation;
  const [imageReadyRecordId, setImageReadyRecordId] = useState<string | null>(null);
  const [nameInputDirty, setNameInputDirty] = useState(false);
  const [noteInputDirty, setNoteInputDirty] = useState(false);

  const elementParam = searchParams.get("element");
  const initialFocusedIdx =
    elementParam !== null && Number.isInteger(Number(elementParam)) && Number(elementParam) >= 0
      ? Number(elementParam)
      : null;

  const annotation = useAnnotationRecord(id, initialFocusedIdx);
  const model = useAnnotationElementModel({
    elements: annotation.elements,
    setElements: annotation.setElements,
    annotationStatus: annotation.annotationStatus,
    setAnnotationStatus: annotation.setAnnotationStatus,
    classes: annotation.classes,
    setCustomClasses: annotation.setCustomClasses,
    focusedIdx: annotation.focusedIdx,
    cardRefs: annotation.cardRefs,
    listReady: !annotation.loading,
  });
  const {
    containerRef,
    imageRef,
    previewCanvasRef,
    handlePreviewImageLoad,
    drawMode,
    zoom,
    panOffset,
    isPanning,
    dragState,
    bboxHistory,
    tempBbox,
    showLabelNames,
    transformSize,
    setDrawMode,
    setShowLabelNames,
    updateStageSize,
    zoomIn,
    zoomOut,
    resetAnnotationView,
    handleStageWheel,
    handleSvgPointerDown,
    handleSvgPointerMove,
    handleSvgPointerUp,
    undoLastBboxChange,
    removeElement,
  } = useAnnotationViewport({
    record: annotation.record,
    elements: annotation.elements,
    setElements: annotation.setElements,
    annotationStatus: annotation.annotationStatus,
    setAnnotationStatus: annotation.setAnnotationStatus,
    focusedIdx: annotation.focusedIdx,
    setFocusedIdx: annotation.setFocusedIdx,
    setHoveredIdx: annotation.setHoveredIdx,
    setNamingFocusToken: model.setNamingFocusToken,
    loading: annotation.loading,
    initialFocusedIdx,
    imageReady: imageReadyRecordId === id && annotation.record?.id === id,
  });
  const submission = useAnnotationSubmission({
    id,
    record: annotation.record,
    elements: annotation.elements,
    annotationStatus: annotation.annotationStatus,
    labels: {
      submitBlockedUnnamed: t.submitBlockedUnnamed,
      submitBlockedNone: t.submitBlockedNone,
    },
  });
  const dirty = submission.dirty || nameInputDirty || noteInputDirty;

  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  if (!annotation.record && !annotation.loading) {
    return (
      <div className="ui-empty-state mx-auto max-w-2xl px-6 py-12 text-center">
        <h2 className="ui-title-md mb-4 text-2xl">{annotation.storageError ? "Stockage local indisponible" : t.notFound}</h2>
        {annotation.storageError ? <p className="mb-4">L'analyse n'a pas été supprimée. Vérifiez l'accès au stockage du navigateur.</p> : null}
        {annotation.storageError ? <button type="button" className="ui-action-primary mb-4 rounded-full px-3 py-1.5" onClick={annotation.retryLoad}>Réessayer</button> : null}
        <Link
          to="/"
          className="ui-action-ghost inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium"
        >
          <ArrowLeft size={18} /> {t.backToHistory}
        </Link>
      </div>
    );
  }

  if (annotation.loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="animate-spin text-[var(--accent)]" size={32} />
      </div>
    );
  }

  return (
    <div
      className={`${annotationStyles.owner} annotation-app flex h-full min-h-0 w-full flex-col gap-0 overflow-hidden rounded-none p-0`}
      onClickCapture={(event) => {
        if (!dirty || !(event.target instanceof Element)) return;
        if (event.target.closest("[data-auth-logout]")) {
          if (!window.confirm("Les modifications de cette annotation ne sont pas enregistrées. Se déconnecter quand même ?")) {
            event.preventDefault();
            event.stopPropagation();
          }
          return;
        }
        if (!event.target.closest("a[href]")) return;
        if (!window.confirm("Les modifications de cette annotation ne sont pas enregistrées. Quitter quand même ?")) event.preventDefault();
        else allowLeaveRef.current = true;
      }}
    >
      {guardRouteTransitions ? <UnsavedNavigationPrompt dirty={dirty} allowLeaveRef={allowLeaveRef} message="Les modifications de cette annotation ne sont pas enregistrées. Quitter quand même ?" /> : null}
      <AnnotationPageChrome
        labels={t}
        analysisId={id}
        imageName={annotation.record?.imageName}
        readyCount={model.submittedCount}
        totalCount={annotation.elements.length}
        saving={submission.saving}
        sending={submission.sending}
        onSubmitNamed={model.submitNamedElements}
        onSave={submission.handleSave}
        onSendSubmittedForReview={submission.handleSendSubmittedForReview}
        themeMode={themeMode}
        onToggleTheme={onToggleTheme}
        authSlot={authSlot}
      />

      <div className="grid min-h-0 flex-1 gap-0 xl:grid-cols-[minmax(0,1.45fr)_minmax(280px,0.55fr)] 2xl:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.45fr)]">
        <ImageBBoxStage
          tone="annotation"
          mode="edit"
          className="min-h-0"
          imageDataUrl={annotation.record!.imageDataUrl}
          imageName={annotation.record!.imageName}
          imageSize={annotation.record!.result.image_size}
          boxes={annotation.elements.map((element, idx) => ({
            id: idx,
            bbox: element.bbox,
            label: element.class_name,
            confidence: element.confidence,
            rejected: element.rejected,
            status: annotation.annotationStatus[idx] ?? "draft",
          }))}
          selectedId={annotation.focusedIdx}
          showLabelNames={showLabelNames}
          viewport={{ zoom, panOffset, isPanning }}
          transformSize={transformSize}
          imageFit="fill"
          displayBBoxById={
            dragState && dragState.type !== "draw" && tempBbox
              ? { [dragState.idx]: tempBbox }
              : undefined
          }
          boxStateById={Object.fromEntries(
            annotation.elements.map((_, idx) => [
              idx,
              {
                focused: idx === annotation.focusedIdx,
                imageHovered: idx === annotation.hoveredIdx,
                listHovered: idx === annotation.listHoveredIdx,
                submitted: annotation.annotationStatus[idx] === "validated",
              },
            ]),
          )}
          renderLabel={(box) =>
            formatBboxLabel(
              Number(box.id),
              box.label ?? "",
              showLabelNames,
              t.unnamedElement,
            )
          }
          renderBoxExtras={(box, _state, bbox) => {
            const idx = Number(box.id);
            if (idx !== annotation.focusedIdx || drawMode) {
              return null;
            }

            const [x, y, width, height] = bbox;
            return (
              <>
                {resizeHandle("tl", x, y, "cursor-nwse-resize")}
                {resizeHandle("tr", x + width, y, "cursor-nesw-resize")}
                {resizeHandle("bl", x, y + height, "cursor-nesw-resize")}
                {resizeHandle(
                  "br",
                  x + width,
                  y + height,
                  "cursor-nwse-resize",
                )}
              </>
            );
          }}
          overlayChildren={
            drawMode && dragState?.type === "draw" && tempBbox ? (
              <rect
                x={tempBbox[0]}
                y={tempBbox[1]}
                width={tempBbox[2]}
                height={tempBbox[3]}
                strokeWidth={2}
                strokeDasharray="4 4"
                vectorEffect="non-scaling-stroke"
                className="annotation-draw-preview"
              />
            ) : null
          }
          boxTestIdPrefix="annotation-box"
          testIds={{
            toolbar: "annotation-stage-toolbox",
            stage: "annotation-stage-frame",
            transform: "annotation-stage",
            overlay: "annotation-overlay",
          }}
          stageRef={containerRef}
          stageProps={{
            "data-testid": "annotation-stage-frame",
            onWheel: handleStageWheel,
          }}
          transformClassName="annotation-stage"
          toolbar={
            <AnnotationAnalyzerToolbar
              drawMode={drawMode}
              canUndo={bboxHistory.length > 0}
              showLabelNames={showLabelNames}
              labels={t}
              onToggleDrawMode={() => setDrawMode((current) => !current)}
              onUndo={undoLastBboxChange}
              onToggleLabelNames={() =>
                setShowLabelNames((current) => !current)
              }
              onZoomOut={zoomOut}
              onFitToView={resetAnnotationView}
              onZoomIn={zoomIn}
              zoomLabel={`${Math.round(zoom * 100)}%`}
            />
          }
          toolbarPlacement="bottom-center"
          imageProps={{
            ref: imageRef,
            onLoad: () => {
              updateStageSize();
              handlePreviewImageLoad();
              setImageReadyRecordId(id ?? null);
            },
          }}
          svgProps={{
            onPointerDown: handleSvgPointerDown,
            onPointerMove: handleSvgPointerMove,
            onPointerUp: handleSvgPointerUp,
          }}
        />

        <aside
          className="annotation-rail annotation-inspector annotation-inspector-rail flex min-h-0 flex-col overflow-hidden rounded-none p-0"
          aria-label="Inspecteur d’annotation"
        >
          <AnnotationSelectedInspector
            focusedElement={model.focusedElement}
            focusedIdx={annotation.focusedIdx}
            focusedDisplayName={model.focusedDisplayName ?? t.unnamedElement}
            focusedConfidencePercent={model.focusedConfidencePercent}
            focusedIsSubmitted={model.focusedIsSubmitted}
            previewCanvasRef={previewCanvasRef}
            classes={annotation.classes}
            customClasses={annotation.customClasses}
            namingFocusToken={model.namingFocusToken}
            labels={t}
            onCommitElementName={(idx, name) => {
              model.commitElementName(idx, name);
              setNameInputDirty(false);
            }}
            onNameInputChange={(name) => setNameInputDirty(name !== (annotation.elements[annotation.focusedIdx ?? -1]?.class_name ?? ""))}
            onCommitElementNote={(idx, note) => {
              model.commitElementNote(idx, note);
              setNoteInputDirty(false);
            }}
            onNoteInputChange={(note) => setNoteInputDirty(note.trim() !== (annotation.elements[annotation.focusedIdx ?? -1]?.note ?? ""))}
            onSetElementValidation={(idx, submitted) =>
              model.setElementValidation(idx, submitted ? "validated" : "draft")
            }
            onRemoveElement={removeElement}
          />

          <AnnotationElementList
            displayedElements={model.displayedElements}
            annotationStatus={annotation.annotationStatus}
            focusedIdx={annotation.focusedIdx}
            elementsCount={annotation.elements.length}
            submittedCount={model.submittedCount}
            listQuery={model.listQuery}
            statusFilter={model.statusFilter}
            sortMode={model.sortMode}
            labels={t}
            cardRefs={annotation.cardRefs}
            setFocusedIdx={annotation.setFocusedIdx}
            onListQueryChange={model.setListQuery}
            onStatusFilterChange={model.setStatusFilter}
            onSortModeChange={model.setSortMode}
            setListHoveredIdx={annotation.setListHoveredIdx}
          />
        </aside>
      </div>
      {submission.toast && <AnnotationToast toast={submission.toast} />}
    </div>
  );
}
