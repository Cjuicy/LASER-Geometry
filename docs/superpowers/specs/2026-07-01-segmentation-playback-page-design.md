# Segmentation Playback Page Design

## Goal

Add a dedicated frame-by-frame playback page to the existing Depth vs Geometry
pipeline report. The page helps inspect temporal segmentation changes without
exporting a video and without changing the existing ten-card report page.

The player compares adjacent frames for both methods at the same time:

| Time | Baseline / Depth | Geometry |
| --- | --- | --- |
| Previous frame | segmentation input depth + segmentation | segmentation input depth + segmentation |
| Current frame | segmentation input depth + segmentation | segmentation input depth + segmentation |

## Output Structure

The report builder writes one additional HTML entry point:

```text
index.html     # existing ten-card report
player.html    # new frame-by-frame player
data.json      # shared manifest
assets/        # shared WebP assets
```

`player.html` reuses the existing `initial` and `merged` assets. It embeds a
playback-oriented snapshot derived from the same canonical manifest used by
`index.html` and `data.json`, so it also works when opened directly without an
HTTP fetch. It does not copy images, encode a video, or require another
inference run.

The existing report header gains a small `Play Sequence` link to `player.html`.
No ten-card row or existing report interaction changes.

## Playback Sequence

Pipeline rows are window-major and overlap frames appear in multiple windows.
Playing rows directly would make the global frame number move backwards at
window boundaries. The report builder therefore creates a canonical playback
sequence with these deterministic rules:

1. Group rows by `global_frame`.
2. Prefer a row whose `is_overlap` value is false.
3. If several rows have the same preference, use the earliest report row.
4. Sort the selected rows by ascending `global_frame`.

Each global frame consequently appears once and playback is monotonic. The
original rows remain unchanged for the detailed report.

## Player Layout

The desktop page uses a two-column, two-row comparison grid. Columns represent
the methods and rows represent the previous and current frames. Each panel
shows the existing horizontal composite of segmentation input depth and the
segmentation visualization.

The fixed control bar contains:

- previous-frame, play/pause, and next-frame controls;
- an Initial Segmentation / Merged Segmentation selector, defaulting to merged;
- playback speeds of 0.5, 1, 2, and 4 FPS, defaulting to 2 FPS;
- a range slider for direct seeking;
- current global frame, sequence position, and segment counts;
- a link back to `index.html`.

On narrow screens the four panels stack vertically. Common controls use icons
with accessible labels and tooltips. The Left and Right arrow keys step one
frame and Space toggles playback.

The first sequence item has no previous frame, so the previous-frame panels
show an explicit empty state. Reaching the final frame pauses playback. Changing
the segmentation stage preserves the current sequence position.

## Runtime Behavior

The page reads its embedded playback manifest and resolves four assets for the
active position:

- previous Depth stage;
- previous Geometry stage;
- current Depth stage;
- current Geometry stage.

Playback advances the current sequence index on a timer derived from the chosen
FPS. The next frame's assets are preloaded after every update to reduce visible
flicker. Starting playback while already at the final frame first seeks to the
beginning.

The player does not calculate a pixel-difference heatmap. Camera motion would
mix viewpoint changes with segmentation changes and could produce a misleading
stability signal. Adjacent images and their segment-count deltas provide the
initial diagnostic surface.

## Errors

Before starting playback, the page validates that the sequence is non-empty and
that every selected row contains matching Depth and Geometry assets for both
supported stages. If an asset or stage is missing, playback pauses and the page
names the missing method, stage, and global frame instead of silently skipping
the frame.

Image load failures replace the affected panel with an error state while
leaving navigation available. Switching stages or seeking clears stale errors
and validates the newly selected position.

## Code Boundaries

The Python report builder owns canonical sequence construction and HTML output.
A small pure helper returns playback row indices from report rows so duplicate
handling is independently testable. Browser JavaScript owns only player state,
asset resolution, timing, keyboard input, preloading, and DOM updates.

The existing `index.html`, report rows, asset names, and v2 trace schema remain
compatible. No inference-engine or segmentation behavior changes are required.

## Verification

Automated tests cover:

- monotonic global-frame ordering;
- duplicate-frame selection preferring non-overlap rows;
- one playback entry per global frame;
- generation of `player.html` and all required controls;
- Initial/Merged stage resolution for both methods;
- empty-sequence and missing-stage error handling.

Browser verification covers play/pause, manual stepping, seeking, stage
switching, speed changes, keyboard controls, final-frame behavior, image
preloading, desktop four-panel layout, and narrow-screen stacking.

## Cloud Update

After the implementation is pushed, the cloud machine only needs to pull the
new code and rerun `eval/build_alignment_pipeline_report.py` against the
existing schema-v2 Depth and Geometry traces. The expensive model inference
runs do not need to be repeated.
