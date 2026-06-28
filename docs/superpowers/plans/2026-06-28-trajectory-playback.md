# Trajectory Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Play, Pause, adjustable FPS, and looping frame playback to the standalone full-trajectory Viser viewer while preserving manual frame dragging.

**Architecture:** A small `PlaybackController` in the standalone viewer owns one paused daemon thread and advances the existing frame slider through injected getter/setter callbacks. Viser controls call the controller, while the existing slider update callback remains the sole detail-cloud rendering path.

**Tech Stack:** Python 3, `threading`, Viser GUI handles, pytest

---

## File Structure

- Modify `eval/vis_full_trajectory_compare.py`: add the playback controller and connect Play, Pause, and FPS controls to the existing Frame slider.
- Modify `tests/test_full_trajectory_compare.py`: cover controller behavior and button callback registration.
- Modify `docs/VISER_COMMANDS_ZH.md`: document playback controls and long-sequence resource guidance.

### Task 1: Deterministic Playback Controller

**Files:**
- Modify: `tests/test_full_trajectory_compare.py`
- Modify: `eval/vis_full_trajectory_compare.py`

- [x] **Step 1: Write failing controller tests**

Add tests that use mutable dictionaries as real frame/FPS storage:

```python
def test_playback_controller_advances_wraps_and_uses_manual_frame():
    viewer = _load_module()
    state = {"frame": 1, "fps": 2.0}
    controller = viewer.PlaybackController(
        frame_count=3,
        get_frame=lambda: state["frame"],
        set_frame=lambda value: state.__setitem__("frame", value),
        get_fps=lambda: state["fps"],
    )
    try:
        assert controller.advance_once() == 2
        assert state["frame"] == 2
        assert controller.advance_once() == 0
        state["frame"] = 1
        assert controller.advance_once() == 2
    finally:
        controller.stop()


def test_playback_controller_play_pause_are_idempotent():
    viewer = _load_module()
    state = {"frame": 0}
    controller = viewer.PlaybackController(
        frame_count=2,
        get_frame=lambda: state["frame"],
        set_frame=lambda value: state.__setitem__("frame", value),
        get_fps=lambda: 0.5,
    )
    try:
        worker = controller._worker
        controller.play()
        controller.play()
        assert controller.is_playing
        assert controller._worker is worker
        controller.pause()
        controller.pause()
        assert not controller.is_playing
        assert state["frame"] == 0
    finally:
        controller.stop()


def test_playback_controller_worker_advances_while_playing():
    viewer = _load_module()
    advanced = viewer.threading.Event()
    state = {"frame": 0}

    def set_frame(value):
        state["frame"] = value
        advanced.set()

    controller = viewer.PlaybackController(
        frame_count=3,
        get_frame=lambda: state["frame"],
        set_frame=set_frame,
        get_fps=lambda: 10.0,
    )
    try:
        controller.play()
        assert advanced.wait(timeout=1.0)
        assert state["frame"] == 1
    finally:
        controller.stop()


@pytest.mark.parametrize(("frame_count", "fps"), [(0, 2.0), (2, 0.0)])
def test_playback_controller_rejects_non_positive_values(frame_count, fps):
    viewer = _load_module()
    with pytest.raises(ValueError, match="positive"):
        viewer.PlaybackController(
            frame_count=frame_count,
            get_frame=lambda: 0,
            set_frame=lambda _: None,
            get_fps=lambda: fps,
        )
```

- [x] **Step 2: Run the controller tests and verify RED**

Run:

```bash
conda run -n vggt-dem python -m pytest -q \
  tests/test_full_trajectory_compare.py::test_playback_controller_advances_wraps_and_uses_manual_frame \
  tests/test_full_trajectory_compare.py::test_playback_controller_play_pause_are_idempotent \
  tests/test_full_trajectory_compare.py::test_playback_controller_worker_advances_while_playing \
  tests/test_full_trajectory_compare.py::test_playback_controller_rejects_non_positive_values
```

Expected: FAIL because `PlaybackController` does not exist.

- [x] **Step 3: Implement the minimal controller**

Add imports and this controller near the viewer helpers:

```python
import threading
from collections.abc import Callable


class PlaybackController:
    def __init__(
        self,
        *,
        frame_count: int,
        get_frame: Callable[[], int],
        set_frame: Callable[[int], None],
        get_fps: Callable[[], float],
    ) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self._frame_count = frame_count
        self._get_frame = get_frame
        self._set_frame = set_frame
        self._get_fps = get_fps
        self._validate_fps()
        self._playing = threading.Event()
        self._stopped = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name="trajectory-playback",
            daemon=True,
        )
        self._worker.start()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def _validate_fps(self) -> float:
        fps = float(self._get_fps())
        if fps <= 0:
            raise ValueError("playback FPS must be positive")
        return fps

    def advance_once(self) -> int:
        next_frame = (int(self._get_frame()) + 1) % self._frame_count
        self._set_frame(next_frame)
        return next_frame

    def play(self) -> None:
        self._playing.set()

    def pause(self) -> None:
        self._playing.clear()

    def stop(self) -> None:
        self._stopped.set()
        self._playing.set()
        self._worker.join(timeout=1.0)
        self._playing.clear()

    def _run(self) -> None:
        while not self._stopped.is_set():
            if not self._playing.wait(timeout=0.1):
                continue
            if self._stopped.wait(1.0 / self._validate_fps()):
                break
            if self._playing.is_set():
                self.advance_once()
```

- [x] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
conda run -n vggt-dem python -m pytest -q tests/test_full_trajectory_compare.py
```

Expected: all focused tests pass.

### Task 2: Connect Viser Playback Controls

**Files:**
- Modify: `tests/test_full_trajectory_compare.py`
- Modify: `eval/vis_full_trajectory_compare.py`

- [x] **Step 1: Write the failing callback-registration test**

```python
def test_register_playback_callbacks_connects_play_and_pause():
    viewer = _load_module()

    class Button:
        callback = None

        def on_click(self, callback):
            self.callback = callback
            return callback

    class Controller:
        playing = False

        def play(self):
            self.playing = True

        def pause(self):
            self.playing = False

    play_button = Button()
    pause_button = Button()
    controller = Controller()
    viewer.register_playback_callbacks(play_button, pause_button, controller)

    play_button.callback(None)
    assert controller.playing
    pause_button.callback(None)
    assert not controller.playing
```

- [x] **Step 2: Run the callback test and verify RED**

Run:

```bash
conda run -n vggt-dem python -m pytest -q \
  tests/test_full_trajectory_compare.py::test_register_playback_callbacks_connects_play_and_pause
```

Expected: FAIL because `register_playback_callbacks` does not exist.

- [x] **Step 3: Add controls and connect them to the existing slider**

Add the helper:

```python
def register_playback_callbacks(play_button, pause_button, controller) -> None:
    play_button.on_click(lambda _: controller.play())
    pause_button.on_click(lambda _: controller.pause())
```

Inside the existing `Display` folder, immediately after `frame_slider`, add:

```python
playback_fps = server.gui.add_slider(
    "Playback FPS",
    min=0.5,
    max=10.0,
    step=0.5,
    initial_value=2.0,
)
play_button = server.gui.add_button("Play")
pause_button = server.gui.add_button("Pause")
```

After the folder, construct and register one controller:

```python
playback = PlaybackController(
    frame_count=int(baseline["num_frames"]),
    get_frame=lambda: int(frame_slider.value),
    set_frame=lambda value: setattr(frame_slider, "value", value),
    get_fps=lambda: float(playback_fps.value),
)
register_playback_callbacks(play_button, pause_button, playback)
```

Do not add a second rendering path. Programmatic writes to Viser input handles
invoke their existing `on_update` callbacks, so the current frame callback
continues to call `render_detail()` for both manual and automatic changes.

- [x] **Step 4: Run the focused viewer tests**

Run:

```bash
conda run -n vggt-dem python -m pytest -q tests/test_full_trajectory_compare.py
```

Expected: all focused tests pass.

### Task 3: Document and Verify the Viewer

**Files:**
- Modify: `docs/VISER_COMMANDS_ZH.md`

- [x] **Step 1: Document the controls**

Add these controls to the full-trajectory section:

```markdown
- `Play`：从当前帧开始自动播放；末帧后回到第 0 帧。
- `Pause`：暂停并保留当前帧。
- `Playback FPS`：调节播放速度，默认 2 FPS。

播放中仍可拖动 `Frame`；播放不会暂停，而是从手动选择的新帧继续。
```

- [x] **Step 2: Run formatting and focused verification**

Run:

```bash
git diff --check
conda run -n vggt-dem python -m pytest -q tests/test_full_trajectory_compare.py
```

Expected: no whitespace errors and all focused tests pass.

- [x] **Step 3: Run the complete test suite**

Run:

```bash
conda run -n vggt-dem python -m pytest -q
```

Expected: the complete suite passes with zero failures.

- [x] **Step 4: Launch and inspect the current KITTI 09 viewer**

Run the existing command from `docs/VISER_COMMANDS_ZH.md` on an unused port.
Verify in the browser that Play advances the slider and detail clouds, Pause
holds the current frame, manual dragging during playback continues from the new
frame, FPS changes take effect, and the final frame wraps to frame 0.

- [x] **Step 5: Commit and push only tracked feature files**

```bash
git add eval/vis_full_trajectory_compare.py \
  tests/test_full_trajectory_compare.py \
  docs/VISER_COMMANDS_ZH.md \
  docs/superpowers/plans/2026-06-28-trajectory-playback.md
git commit -m "feat: add trajectory playback controls"
git push origin main
```

Confirm that the untracked local `viser/` directory is not staged.
