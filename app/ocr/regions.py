"""Find where the document sits on a mostly blank scanned page.

Scans of ID cards and passports often fill only a third of a letter page.
Whole-page OCR then reads tiny text (Tesseract finds nothing on the SID card
sample) and vision models waste their image budget on white space.

The document is the largest non-white blob on a coarse render. Small specks
and thin scan-edge smudges form separate, smaller blobs and are ignored.
"""
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageFilter


def _block_min(gray, block):
    """Downscale keeping the DARKEST pixel per block, so thin text strokes survive."""
    arr = np.asarray(gray)
    h, w = arr.shape
    h2, w2 = h // block * block, w // block * block
    blocks = arr[:h2, :w2].reshape(h2 // block, block, w2 // block, block)
    return blocks.min(axis=(1, 3))


def document_box(image, scale=10, thresh=150, close=2, margin=0.02, min_share=0.1):
    """Return (box, outside_ink) or (None, 0).

    box is (x0, y0, x1, y1) as page fractions covering every blob that is at
    least `min_share` of the largest one. outside_ink is the share of dark
    cells that fall outside that box (specks, edge smudges, or real content
    the box would cut off).
    """
    ink = _block_min(image.convert("L"), scale) < thresh
    mask = Image.fromarray((ink * 255).astype("uint8"))
    size = 2 * close + 1
    mask = mask.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))  # close gaps
    grid = np.asarray(mask) > 0
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    blobs = []

    for sy in range(rows):
        for sx in range(cols):
            if not grid[sy, sx] or seen[sy, sx]:
                continue
            stack, count = [(sy, sx)], 0
            seen[sy, sx] = True
            x0 = x1 = sx
            y0 = y1 = sy
            while stack:
                y, x = stack.pop()
                count += 1
                x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            blobs.append((count, x0, y0, x1 + 1, y1 + 1))

    if not blobs:
        return None, 0.0
    biggest = max(b[0] for b in blobs)
    kept = [b for b in blobs if b[0] >= min_share * biggest]
    bx0, by0 = min(b[1] for b in kept), min(b[2] for b in kept)
    bx1, by1 = max(b[3] for b in kept), max(b[4] for b in kept)

    total_ink = max(int(ink.sum()), 1)
    inside_ink = int(ink[by0:by1, bx0:bx1].sum())
    outside = (total_ink - inside_ink) / total_ink
    box = (max(0.0, bx0 / cols - margin), max(0.0, by0 / rows - margin),
           min(1.0, bx1 / cols + margin), min(1.0, by1 / rows + margin))
    return box, outside


def page_clip(page, max_area=0.6, max_outside_ink=0.05):
    """Rect to render for this page, or None when the page should be used whole.

    Never crops if more than `max_outside_ink` of the page's ink would be cut
    off: losing real text is worse than reading a little extra white space.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    box, outside = document_box(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    if not box or outside > max_outside_ink:
        return None
    if (box[2] - box[0]) * (box[3] - box[1]) > max_area:
        return None  # document already fills most of the page
    rect = page.rect
    return fitz.Rect(rect.x0 + box[0] * rect.width, rect.y0 + box[1] * rect.height,
                     rect.x0 + box[2] * rect.width, rect.y0 + box[3] * rect.height)


def render(page, zoom, target_px, clip):
    """Render a page (or its clip). A clip is scaled up to ~target_px wide."""
    if clip is not None:
        zoom = min(4.0, max(zoom, target_px / max(clip.width, 1)))
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)