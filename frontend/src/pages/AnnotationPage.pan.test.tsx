import React from 'react';
import { render, fireEvent, act } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { vi, describe, it, expect, beforeEach } from 'vitest';

vi.mock('../services/storage', () => ({
  getAnalysisById: vi.fn(),
  updateElements: vi.fn(() => true),
}));

vi.mock('../services/api', () => ({
  getClasses: vi.fn(() => Promise.resolve({ class_names: [] })),
  saveAnnotation: vi.fn(),
}));

import { getAnalysisById } from '../services/storage';
import AnnotationPage from './AnnotationPage';

const STUB_RECORD = {
  id: 'test-id',
  imageDataUrl: 'data:image/png;base64,abc',
  imageName: 'test.png',
  timestamp: '2024-01-01T00:00:00Z',
  result: {
    image_size: [800, 600] as [number, number],
    elements: [],
  },
};

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/annotation/test-id']}>
      <Routes>
        <Route path="/annotation/:id" element={<AnnotationPage />} />
        <Route path="/" element={<div>home</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

const CONTAINER_RECT = {
  left: 0, top: 0, width: 800, height: 600,
  right: 800, bottom: 600, x: 0, y: 0, toJSON: () => {},
} as DOMRect;

beforeEach(() => {
  vi.mocked(getAnalysisById).mockReturnValue(STUB_RECORD as any);
  Element.prototype.getBoundingClientRect = vi.fn(() => CONTAINER_RECT);
});

function getSvgAndWrapper(container: HTMLElement): { svg: SVGSVGElement; wrapper: HTMLElement } {
  const svg = container.querySelector('svg.absolute') as SVGSVGElement;
  svg.setPointerCapture = vi.fn();
  svg.releasePointerCapture = vi.fn();
  svg.hasPointerCapture = vi.fn(() => false);
  return { svg, wrapper: svg.parentElement as HTMLElement };
}

async function clickZoomIn(container: HTMLElement, times: number) {
  const buttons = container.querySelectorAll('button');
  const zoomInBtn = Array.from(buttons).find(b => b.title === '' && b.querySelector('svg') && b.className.includes('p-1'));
  const zoomButtons = Array.from(buttons).filter(b => b.className.includes('p-1.5') || b.className.includes('p-1'));
  const zoomInButton = zoomButtons[0] as HTMLElement;
  for (let i = 0; i < times; i++) {
    await act(async () => { fireEvent.click(zoomInButton); });
  }
}

describe('AnnotationPage pan behavior', () => {
  it('at zoom=1, pointerdown+move on background does NOT change wrapper transform', async () => {
    const { container } = renderPage();
    await act(async () => {});

    const { svg, wrapper } = getSvgAndWrapper(container);
    const transformBefore = wrapper.style.transform;

    await act(async () => {
      fireEvent.pointerDown(svg, { clientX: 100, clientY: 100, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerMove(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerUp(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });

    expect(wrapper.style.transform).toBe(transformBefore);
  });

  it('at zoom=2, pointerdown+move on background updates wrapper transform with translate', async () => {
    const { container } = renderPage();
    await act(async () => {});

    await clickZoomIn(container, 4);

    const { svg, wrapper } = getSvgAndWrapper(container);

    await act(async () => {
      fireEvent.pointerDown(svg, { clientX: 100, clientY: 100, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerMove(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerUp(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });

    expect(wrapper.style.transform).toMatch(/translate\(/);
    expect(wrapper.style.transform).not.toMatch(/translate\(0px,\s*0px\)/);
  });

  it('reset-view button resets pan and zoom', async () => {
    const { container } = renderPage();
    await act(async () => {});

    await clickZoomIn(container, 4);

    const { svg, wrapper } = getSvgAndWrapper(container);

    await act(async () => {
      fireEvent.pointerDown(svg, { clientX: 100, clientY: 100, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerMove(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });
    await act(async () => {
      fireEvent.pointerUp(svg, { clientX: 250, clientY: 250, pointerId: 1 });
    });

    const buttons = container.querySelectorAll('button');
    const resetBtn = Array.from(buttons).find(b => b.title === 'Réinitialiser la vue') as HTMLElement;

    await act(async () => {
      fireEvent.click(resetBtn);
    });

    expect(wrapper.style.transform).toMatch(/translate\(0px,\s*0px\)\s*scale\(1\)/);
  });
});
