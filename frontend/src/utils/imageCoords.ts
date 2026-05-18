/**
 * Screen ↔ image-pixel coordinate mapping.
 *
 * Uses getBoundingClientRect() + viewBox ratio — robust under any CSS transform
 * on ancestor elements (unlike getScreenCTM() which breaks under scale()).
 */

export interface ViewBox {
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

/**
 * Map a client (screen) point to image-pixel coordinates.
 * @param svg   The SVGSVGElement whose bounding rect defines the display area.
 * @param clientX  Pointer clientX.
 * @param clientY  Pointer clientY.
 * @param viewBox  The SVG viewBox dimensions (= image pixel dimensions).
 * @returns Point in image-pixel space. Returns {x:0,y:0} if rect has zero size.
 */
export function clientToImage(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
  viewBox: ViewBox,
): Point {
  const rect = svg.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0 || viewBox.width === 0 || viewBox.height === 0) {
    return { x: 0, y: 0 };
  }
  return {
    x: ((clientX - rect.left) / rect.width) * viewBox.width,
    y: ((clientY - rect.top) / rect.height) * viewBox.height,
  };
}

/**
 * Map an image-pixel point to client (screen) coordinates.
 * Inverse of clientToImage.
 */
export function imageToClient(
  svg: SVGSVGElement,
  imgX: number,
  imgY: number,
  viewBox: ViewBox,
): Point {
  const rect = svg.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0 || viewBox.width === 0 || viewBox.height === 0) {
    return { x: 0, y: 0 };
  }
  return {
    x: rect.left + (imgX / viewBox.width) * rect.width,
    y: rect.top + (imgY / viewBox.height) * rect.height,
  };
}
