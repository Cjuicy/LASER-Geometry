# Trajectory Playback Design

## Goal

Add automatic frame playback to the standalone full-trajectory Viser viewer so
long sequences can be inspected without repeatedly dragging the frame slider.
Playback must remain optional and must not change inference or saved outputs.

## Scope and Isolation

Only `eval/vis_full_trajectory_compare.py`, its focused tests, and the Viser
command guide change. `demo.py`, `StreamingWindowEngine`, loop closure, and all
alignment/refinement paths remain untouched.

## Controls and Behavior

The existing `Display` folder gains:

- a `Play` button;
- a `Pause` button;
- a `Playback FPS` slider, defaulting to 2 FPS.

Playback advances the existing `Frame` slider one frame at a time. Reaching the
last frame wraps to frame 0. Pressing `Pause` stops automatic advancement but
keeps the currently displayed frame. While playback is active, manually moving
the frame slider does not pause playback; the next tick continues from the
manually selected frame.

Repeated `Play` clicks must not create additional playback workers or cause
frames to advance multiple times per tick. Playback starts only after an
explicit `Play` click and remains paused when the viewer first opens.

## Implementation

A small playback controller owns a single daemon worker and two thread events:
one for the playing state and one for shutdown. The worker reads the current
frame, advances it with modulo frame count, writes the new value through the
existing frame slider, and waits according to the current FPS value. Using the
existing slider preserves the current detail-cloud render callback and manual
frame selection behavior.

The FPS control is read on every tick so speed changes take effect without
restarting playback. The supported range is 0.5 to 10 FPS. Rendering remains
sequential: the next frame is not requested until the current slider update and
its detail-cloud work have been dispatched.

## Error and Lifecycle Behavior

Frame count and FPS must be positive. Invalid controller construction raises a
clear `ValueError`. The worker is a daemon so `Ctrl+C` can still terminate the
standalone viewer normally. Pause is idempotent, and multiple Play clicks are
idempotent.

## Verification

Focused tests cover:

- normal frame advancement and last-frame wraparound;
- pause preserving the current frame;
- manual frame changes becoming the source of the next playback tick;
- repeated Play calls reusing one worker;
- positive frame-count and FPS validation.

The existing full test suite must remain green. Manual Viser verification uses
the current KITTI 09 comparison scene and checks Play, Pause, speed changes,
wraparound, and dragging the Frame slider during playback.
