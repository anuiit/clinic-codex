import { createRef } from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { appText } from '../../i18n/text';
import type { DetectedElement } from '../../types';
import { AnnotationSelectedInspector } from './AnnotationSelectedInspector';

function element(note: string): DetectedElement {
  return {
    bbox: [1, 2, 3, 4],
    class_name: 'atl',
    confidence: 0.8,
    rejected: false,
    top_k: [{ class_name: 'atl', confidence: 0.8 }],
    note,
  };
}

describe('AnnotationSelectedInspector', () => {
  it('resets an uncommitted note when the analysis changes at the same focused index', async () => {
    const onCommitElementNote = vi.fn();
    const commonProps = {
      focusedIdx: 0,
      focusedDisplayName: 'atl',
      focusedConfidencePercent: 80,
      focusedIsSubmitted: false,
      previewCanvasRef: createRef<HTMLCanvasElement>(),
      classes: ['atl'],
      customClasses: [],
      namingFocusToken: 0,
      labels: appText.annotation,
      onCommitElementName: vi.fn(),
      onCommitElementNote,
      onSetElementValidation: vi.fn(),
      onRemoveElement: vi.fn(),
    };

    const { rerender } = render(
      <AnnotationSelectedInspector
        {...commonProps}
        recordId="analysis-a"
        focusedElement={element('note a')}
      />,
    );
    const textarea = screen.getByTestId('annotation-element-note');
    fireEvent.change(textarea, { target: { value: 'uncommitted draft' } });
    expect(textarea).toHaveValue('uncommitted draft');

    rerender(
      <AnnotationSelectedInspector
        {...commonProps}
        recordId="analysis-b"
        focusedElement={element('note b')}
      />,
    );
    await waitFor(() => expect(textarea).toHaveValue('note b'));
    fireEvent.blur(textarea);
    expect(onCommitElementNote).not.toHaveBeenCalled();
  });
});
