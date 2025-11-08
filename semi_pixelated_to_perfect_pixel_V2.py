# -*- coding: utf-8 -*-
"""
Semi-Pixelated to Perfect Pixel — Local Script (cleaned, minimal changes)

- Local-only: no Colab/Drive dependencies
- Uses files in the same folder as this script:
    - input:  input_image.png
    - output: fixed_image.png, intersections.txt
- Small fixes:
  * Respect show_plots parameter in batch mode
  * Save output to the correct output path (not the input path)
  * Light refactors + comments; major logic unchanged
"""

import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


# === BATCH MODE (optional) ===
BATCH_MODE = False                 # set True to enable folder processing
INPUT_DIR = "input_images"         # folder with source images
OUTPUT_DIR = "output_images"       # folder to write results
BASE_OUTPUT_NAME = "fixed_image"   # produces fixed_image.png, fixed_image (2).png, ...

# === CONFIGURABLE PARAMETERS ===
SHOW_PLOTS = True  # default plotting behavior for single-image mode

# Minimum number of edge pixels required to consider a row/column as a line candidate
MIN_EDGE_PIXELS = 25

# === CANNY EDGE DETECTION PARAMETERS ===
CANNY_THRESHOLD_LOW = 30     # lower threshold for weak edges
CANNY_THRESHOLD_HIGH = 100   # upper threshold for strong edges

# DBSCAN parameters for clustering nearby detected lines
DBSCAN_EPS = 3               # max distance (in pixels) between clustered line positions
DBSCAN_MIN_SAMPLES = 1       # minimum samples per cluster (1 = even single line allowed)

# Internal runtime toggle so visualize() can respect run-time choice without changing many call sites
_CURRENT_SHOW_PLOTS = SHOW_PLOTS


# ---------- Helpers ----------
def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_image_readable(path: str, flag=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(path, flag)
    if img is None:
        fail(f"Could not read image at '{path}'. Make sure the file exists and is a readable image.")
    return img


def smooth_edges(edges: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    """
    Smooth the edges by eliminating clusters of adjacent edge pixels along rows.
    """
    smoothed = np.copy(edges)

    for i in range(edges.shape[0]):
        cluster = [0] if edges[i, 0] == 255 else []
        for j in range(1, edges.shape[1]):
            if edges[i, j] == 255 and (not cluster or j == cluster[-1] + 1):
                cluster.append(j)
            else:
                if len(cluster) >= min_cluster_size:
                    smoothed[i, cluster] = 0
                cluster = [j] if edges[i, j] == 255 else []
        if len(cluster) >= min_cluster_size:
            smoothed[i, cluster] = 0

    return smoothed


def cluster_line_positions(edge_mask_2d: np.ndarray, axis: str) -> np.ndarray:
    """
    Given a binary edge mask (255 for edge), cluster positions to find
    the dominant line coordinates.

    axis='rows'  -> find horizontal lines (varying along columns, index with y/row)
    axis='cols'  -> find vertical lines   (varying along rows, index with x/col)

    Returns a binary mask of the same shape with averaged line positions set to 255.
    """
    if axis == "rows":
        coords = np.where(edge_mask_2d == 255)[0].reshape(-1, 1)     # y-coords
    elif axis == "cols":
        coords = np.where(edge_mask_2d.T == 255)[0].reshape(-1, 1)   # x-coords via transpose
    else:
        fail("axis must be 'rows' or 'cols'")

    if coords.size == 0:
        fail("No edge pixels found to cluster. Try adjusting Canny thresholds.")

    # Count occurrences and filter out sparse coordinates
    counts = Counter(coords.flatten())
    filtered = np.array([pos for pos in coords if counts[pos[0]] > MIN_EDGE_PIXELS]).reshape(-1, 1)
    if filtered.size == 0:
        fail("Not enough repeated edge positions to cluster. The image might not be grid-like.")

    # Cluster nearby coordinates and average them into single lines
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(filtered)
    labels = clustering.labels_
    averaged_positions = [int(np.mean(filtered[labels == i])) for i in np.unique(labels)]

    out = np.zeros_like(edge_mask_2d, dtype=np.uint8)
    if axis == "rows":
        for y in averaged_positions:
            if 0 <= y < out.shape[0]:
                out[y, :] = 255
    else:
        for x in averaged_positions:
            if 0 <= x < out.shape[1]:
                out[:, x] = 255
    return out


def compute_midpoints_from_grid(grid_mask: np.ndarray) -> np.ndarray:
    """
    From a grid mask with 255 at horizontal and vertical lines, compute midpoints
    for each cell (grid square). Returns an (N, 2) array of (row, col) integer midpoints.
    """
    h, w = grid_mask.shape

    # Ensure a left boundary line exists to detect the first set of cells.
    if h > 1 and w > 1 and grid_mask[1, 0] != 255 and grid_mask[1, 1] != 255:
        grid_mask[:, 0] = 255

    intersections = []

    # Detect horizontal lines and collect intersection points where vertical lines meet them.
    for i in range(1, h):
        if grid_mask[i, 0] == 255 and grid_mask[i, 1] == 255:
            row_points = [[i, 0]]
            for j in range(w):
                if grid_mask[i, j] == 255 and i < h - 1 and grid_mask[i + 1, j] == 255:
                    row_points.append([i, j])
            if row_points:
                row_points.append([i, w - 1])
                intersections.append(row_points)

    # Ensure bottom boundary line is included for closing the last row of cells
    last_row = grid_mask[-1, :]
    last_points = [[h - 1, 0]]
    for j in range(w):
        if last_row[j] == 255 and grid_mask[-2, j] == 255:
            last_points.append([h - 1, j])
    last_points.append([h - 1, w - 1])
    intersections.append(last_points)

    # Compute midpoints between diagonal pairs of intersections
    mids = []
    for r in range(len(intersections) - 1):
        for c in range(len(intersections[r]) - 1):
            if c < len(intersections[r + 1]) - 1:
                mid_r = (intersections[r][c][0] + intersections[r + 1][c + 1][0]) / 2.0
                mid_c = (intersections[r][c][1] + intersections[r + 1][c + 1][1]) / 2.0
                mids.append([mid_r, mid_c])

    midpoints = np.array(mids).round().astype(int)
    if midpoints.size == 0:
        fail("Could not compute any cell midpoints. The detected grid might be degenerate.")
    return midpoints


def visualize(title: str, img: np.ndarray) -> None:
    # Respect the runtime toggle, so batch mode remains quiet.
    if not _CURRENT_SHOW_PLOTS:
        return
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def next_available_output_path(base_dir, base_name="fixed_image", ext=".png", index=None):
    """
    Generate sequentially numbered output filenames like:
      01fixed_image.png, 02fixed_image.png, 03fixed_image.png, ...
    If index is provided, it's used directly.
    Otherwise, the next available number is chosen automatically.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if index is not None:
        # Use the provided index directly
        filename = f"{index:02d}{base_name}{ext}"
        return base_dir / filename

    # Auto-find next number
    existing = sorted(base_dir.glob(f"[0-9][0-9]{base_name}{ext}"))
    next_num = len(existing) + 1
    filename = f"{next_num:02d}{base_name}{ext}"
    return base_dir / filename



def batch_process_folder():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accept common image types
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted(
        [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda x: x.name.lower()
    )


    if not files:
        print(f"No input images found in: {in_dir}")
        return

    for p in files:
        out_png = next_available_output_path(out_dir, BASE_OUTPUT_NAME, ".png")
        txt_name = out_png.with_suffix(".txt")
        run_pipeline(
            input_image_path=str(p),
            output_image_path=str(out_png),
            intersections_txt_path=str(txt_name),
            show_plots=False,   # <- ensure quiet batch mode
        )
        print(f"Processed {p.name} -> {out_png.name}")


def run_pipeline(input_image_path="input_image.png",
                 output_image_path="fixed_image.png",
                 intersections_txt_path="intersections.txt",
                 show_plots=SHOW_PLOTS):
    global _CURRENT_SHOW_PLOTS
    _CURRENT_SHOW_PLOTS = bool(show_plots)  # make visualize() respect this run's choice

    # Read images
    image_gray = check_image_readable(input_image_path, flag=cv2.IMREAD_GRAYSCALE)
    visualize("Grayscale Input", image_gray)

    # Edge detection
    edges = cv2.Canny(image_gray, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    visualize("Canny Edges", edges)

    # Smooth to keep only isolated row- or column-wise edges
    smoothed_rows = smooth_edges(edges)           # remove wide clusters along rows (keeps vertical-ish features)
    smoothed_cols = smooth_edges(edges.T).T       # remove wide clusters along columns (keeps horizontal-ish features)
    visualize("Smoothed Edges (Rows)", smoothed_rows)
    visualize("Smoothed Edges (Cols)", smoothed_cols)

    # Cluster horizontal and vertical lines
    merged_h = cluster_line_positions(smoothed_cols, axis="rows")
    visualize("Merged Horizontal Lines", merged_h)

    merged_v = cluster_line_positions(smoothed_rows, axis="cols")
    visualize("Merged Vertical Lines", merged_v)

    # Combine grid
    grid = np.maximum(merged_h, merged_v).astype(np.uint8)
    visualize("Combined Grid", grid)

    # Overlay grid on original for sanity-check (color input for overlay)
    image_color = check_image_readable(input_image_path, flag=cv2.IMREAD_COLOR)
    overlay = cv2.cvtColor(image_color.copy(), cv2.COLOR_BGR2RGB)
    if overlay.shape[:2] != grid.shape:
        overlay = cv2.resize(overlay, (grid.shape[1], grid.shape[0]))
    overlay[grid == 255] = [255, 0, 0]  # red grid overlay
    visualize("Grid Overlay", overlay)

    # Compute midpoints of grid cells
    midpoints = compute_midpoints_from_grid(grid)

    # Save intersections (midpoints-based info) for debugging
    np.set_printoptions(threshold=sys.maxsize)
    with open(intersections_txt_path, "w", encoding="utf-8") as f:
        f.write(str(midpoints.tolist()))

    # Rebuild true pixel-art image by sampling original colors at cell midpoints
    unique_rows = np.unique(midpoints[:, 0]).shape[0]
    unique_cols = np.unique(midpoints[:, 1]).shape[0]

    new_img = np.zeros((unique_rows, unique_cols, 3), dtype=np.uint8)
    img_for_sampling = check_image_readable(input_image_path, flag=cv2.IMREAD_COLOR)
    if img_for_sampling.shape[:2] != grid.shape:
        img_for_sampling = cv2.resize(img_for_sampling, (grid.shape[1], grid.shape[0]))

    total = unique_rows * unique_cols
    if total > len(midpoints):
        fail("Midpoint count is smaller than expected grid size. The grid detection may have failed.")

    idx = 0
    for i in range(unique_rows):
        for j in range(unique_cols):
            x, y = midpoints[idx]  # (row, col)
            new_img[i, j] = img_for_sampling[int(x), int(y)]
            idx += 1

    # Save reconstructed image to the CORRECT path
    cv2.imwrite(output_image_path, new_img)
    print(f"[OK] Wrote fixed image to: {output_image_path}")
    print(f"[OK] Wrote intersections to: {intersections_txt_path}")

    # reset runtime plotting to default global for safety (optional)
    _CURRENT_SHOW_PLOTS = SHOW_PLOTS


if __name__ == "__main__":
    if BATCH_MODE:
        batch_process_folder()
    else:
        run_pipeline()
