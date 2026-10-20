import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AnalysisProvenance } from './AnalysisProvenance';

describe('analysis provenance', () => {
  it('shows the model saved with the analysis and distinguishes human corrections', () => {
    render(<AnalysisProvenance result={{ num_elements: 0, image_size: [8, 6], elements: [], provenance: {
      image_pixel_sha256: 'page', correction_count: 2, corrections_enabled: true, segmenter: null,
      classifier_models: [{ model_version: '1.0', version_id: 'saved-version', artifact_id: 'abcdef1234567890', sha256: {}, backbone: { backbone: 'dinov2_vits14', state_sha256: 'backbone1234567890' } }],
    } }} />);
    expect(screen.getByLabelText('Provenance de cette analyse')).toHaveTextContent('saved-version');
    expect(screen.getByLabelText('Provenance de cette analyse')).toHaveTextContent('dinov2_vits14');
    expect(screen.getByLabelText('Provenance de cette analyse')).toHaveTextContent('backbone1234');
    expect(screen.getByText(/2 correction/)).toHaveTextContent('autres régions restent des propositions du modèle');
  });
});
