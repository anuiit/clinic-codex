import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import type { AnalysisRecord } from '../types';
import { WorkspaceHistoryPanel } from './WorkspaceHistoryPanel';

it('lets a keyboard user delete a history item without selecting it', async () => {
  const record = {
    id: 'analysis-1', imageName: 'page.png', imageDataUrl: 'data:image/png;base64,',
    timestamp: 0, result: { num_elements: 0, image_size: [1, 1], elements: [] }, annotations: {},
  } satisfies AnalysisRecord;
  const onRemoveRecord = vi.fn();
  const onSelectRecord = vi.fn();
  render(<WorkspaceHistoryPanel
    records={[record]} filteredRecords={[record]} currentRecordId={null}
    filter="" historyOpen labels={{
      expandHistory: 'Ouvrir', collapseHistory: 'Fermer', filterPlaceholder: 'Filtrer',
      noAnalyses: 'Vide', noFilterMatch: 'Aucun résultat', elementsSuffix: 'éléments',
      rejectedSuffix: 'rejetés', historyTitle: 'Historique', totalSuffix: 'total',
      deleteLabel: 'Supprimer',
    }}
    onFilterChange={vi.fn()} onToggleHistoryOpen={vi.fn()}
    onSelectRecord={onSelectRecord} onRemoveRecord={onRemoveRecord}
  />);
  screen.getByRole('button', { name: 'Supprimer page.png' }).focus();
  await userEvent.keyboard(' ');
  expect(onRemoveRecord).toHaveBeenCalledWith('analysis-1');
  expect(onSelectRecord).not.toHaveBeenCalled();
});
