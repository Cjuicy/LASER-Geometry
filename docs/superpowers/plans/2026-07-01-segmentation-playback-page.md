# Segmentation Playback Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a standalone `player.html` that compares adjacent Baseline/Depth and Geometry segmentation frames with playback, stepping, seeking, stage selection, and speed controls.

**Architecture:** Extend the existing Python report builder with a pure canonical-row selector and a compact playback-manifest builder. Generate `player.html` beside `index.html`, embedding the compact manifest so direct file opening remains supported, while both pages reuse the existing report assets. Keep all runtime playback state in dependency-free browser JavaScript and leave inference and segmentation code unchanged.

**Tech Stack:** Python 3, NumPy/OpenCV report pipeline, static HTML/CSS/JavaScript, pytest, in-app browser verification.

---

## File Map

- Modify `eval/build_alignment_pipeline_report.py`: canonical playback sequence, compact playback manifest, player HTML/CSS/JavaScript, report-page link, file output.
- Modify `tests/test_alignment_pipeline_report.py`: sequence selection, validation, generated controls, and output-file tests.
- Modify `docs/CLOUD_RUN.md`: explain `player.html` and report-only cloud regeneration.

### Task 1: Canonical Playback Sequence

**Files:**
- Modify: `eval/build_alignment_pipeline_report.py`
- Test: `tests/test_alignment_pipeline_report.py`

- [ ] **Step 1: Write failing sequence-selection tests**

Add the import and tests:

```python
from eval.build_alignment_pipeline_report import select_playback_row_indices


def test_playback_sequence_is_unique_monotonic_and_prefers_non_overlap():
    rows = [
        {"global_frame": 2, "is_overlap": True},
        {"global_frame": 0, "is_overlap": False},
        {"global_frame": 1, "is_overlap": True},
        {"global_frame": 1, "is_overlap": False},
        {"global_frame": 2, "is_overlap": False},
    ]

    assert select_playback_row_indices(rows) == [1, 3, 4]


def test_playback_sequence_keeps_earliest_equal_preference():
    rows = [
        {"global_frame": 5, "is_overlap": False},
        {"global_frame": 5, "is_overlap": False},
    ]

    assert select_playback_row_indices(rows) == [0]
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_pipeline_report.py::test_playback_sequence_is_unique_monotonic_and_prefers_non_overlap \
  tests/test_alignment_pipeline_report.py::test_playback_sequence_keeps_earliest_equal_preference -q
```

Expected: collection fails because `select_playback_row_indices` is not defined.

- [ ] **Step 3: Implement the pure selector**

Add near the other report helpers:

```python
def select_playback_row_indices(rows):
    selected = {}
    for row_index, row in enumerate(rows):
        global_frame = int(row["global_frame"])
        current_index = selected.get(global_frame)
        if current_index is None:
            selected[global_frame] = row_index
            continue
        current = rows[current_index]
        if current["is_overlap"] and not row["is_overlap"]:
            selected[global_frame] = row_index
    return [selected[global_frame] for global_frame in sorted(selected)]
```

- [ ] **Step 4: Run the selector tests and verify GREEN**

Run the command from Step 2.

Expected: `2 passed`.

- [ ] **Step 5: Commit the selector**

```bash
git add eval/build_alignment_pipeline_report.py tests/test_alignment_pipeline_report.py
git commit -m "feat: select canonical playback frames"
```

### Task 2: Compact Playback Manifest and Validation

**Files:**
- Modify: `eval/build_alignment_pipeline_report.py`
- Test: `tests/test_alignment_pipeline_report.py`

- [ ] **Step 1: Write failing playback-manifest tests**

Import `build_playback_manifest`:

```python
from eval.build_alignment_pipeline_report import build_playback_manifest
```

Then add:

```python
def test_playback_manifest_contains_two_methods_and_two_stages():
    row = {
        "global_frame": 7,
        "is_overlap": False,
        "stages": [
            {"method": method, "stage": stage, "asset": f"{method}-{stage}.webp", "details": {"segment_count": count}}
            for method, count in (("depth", 3), ("geometry", 8))
            for stage in ("initial", "merged")
        ],
    }

    playback = build_playback_manifest({"metadata": {"sample_interval": 10}, "rows": [row]})

    assert playback["metadata"]["sample_interval"] == 10
    assert playback["frames"][0]["global_frame"] == 7
    assert playback["frames"][0]["stages"]["merged"]["depth"] == {
        "asset": "depth-merged.webp",
        "segment_count": 3,
    }
    assert playback["frames"][0]["stages"]["initial"]["geometry"]["segment_count"] == 8


def test_playback_manifest_names_a_missing_stage():
    row = {
        "global_frame": 9,
        "is_overlap": False,
        "stages": [
            {"method": "depth", "stage": "initial", "asset": "depth.webp", "details": {"segment_count": 2}},
        ],
    }

    with pytest.raises(ValueError, match="geometry.*merged.*global frame 9"):
        build_playback_manifest({"metadata": {}, "rows": [row]})
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_pipeline_report.py -q
```

Expected: import or name failure for `build_playback_manifest`.

- [ ] **Step 3: Implement compact frame extraction**

Add constants and helper:

```python
PLAYBACK_METHODS = ("depth", "geometry")
PLAYBACK_STAGES = ("initial", "merged")


def build_playback_manifest(manifest):
    frames = []
    for row_index in select_playback_row_indices(manifest["rows"]):
        row = manifest["rows"][row_index]
        lookup = {
            (stage["method"], stage["stage"]): stage
            for stage in row["stages"]
        }
        stages = {}
        for stage_name in PLAYBACK_STAGES:
            stages[stage_name] = {}
            for method in PLAYBACK_METHODS:
                stage = lookup.get((method, stage_name))
                if stage is None:
                    raise ValueError(
                        f"Missing {method} {stage_name} stage for global frame {row['global_frame']}"
                    )
                stages[stage_name][method] = {
                    "asset": stage["asset"],
                    "segment_count": int(stage["details"]["segment_count"]),
                }
        frames.append(
            {
                "global_frame": int(row["global_frame"]),
                "row_index": row_index,
                "stages": stages,
            }
        )
    return {"metadata": dict(manifest["metadata"]), "frames": frames}
```

- [ ] **Step 4: Run report tests and verify GREEN**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_pipeline_report.py -q
```

Expected: all report tests pass.

- [ ] **Step 5: Commit compact playback data**

```bash
git add eval/build_alignment_pipeline_report.py tests/test_alignment_pipeline_report.py
git commit -m "feat: build playback manifest"
```

### Task 3: Generate the Standalone Player Page

**Files:**
- Modify: `eval/build_alignment_pipeline_report.py`
- Test: `tests/test_alignment_pipeline_report.py`

- [ ] **Step 1: Write failing output-contract tests**

Extend `test_report_writes_ten_assets_per_row_and_single_entry_html` after `build_report(...)`:

```python
player_html = (out_dir / "player.html").read_text(encoding="utf-8")
assert 'href="player.html"' in html
assert 'href="index.html"' in player_html
assert 'id="playback-grid"' in player_html
assert 'id="previous-button"' in player_html
assert 'id="play-button"' in player_html
assert 'id="next-button"' in player_html
assert 'id="timeline"' in player_html
assert 'id="playback-speed"' in player_html
assert 'data-stage="initial"' in player_html
assert 'data-stage="merged"' in player_html
assert 'id="previous-depth-image"' in player_html
assert 'id="previous-geometry-image"' in player_html
assert 'id="current-depth-image"' in player_html
assert 'id="current-geometry-image"' in player_html
assert '@media (max-width: 760px)' in player_html
```

- [ ] **Step 2: Run the report test and verify RED**

Run:

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_pipeline_report.py::test_report_writes_ten_assets_per_row_and_single_entry_html -q
```

Expected: failure because `player.html` does not exist.

- [ ] **Step 3: Add the player HTML structure and styling**

Create `_build_player_html(playback_manifest)` beside `_build_html`. It must:

```python
def _build_player_html(playback_manifest):
    playback_json = json.dumps(playback_manifest, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LASER Segmentation Playback</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #17212b; background: #eef1f4; font-family: Arial, sans-serif; }}
    .toolbar {{ position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 14px; background: #fff; border-bottom: 1px solid #b9c2ca; }}
    .controls {{ display: flex; align-items: center; gap: 6px; }}
    button, select, input {{ font: inherit; }}
    button {{ min-height: 34px; border: 1px solid #aeb8c1; border-radius: 4px; background: #fff; cursor: pointer; }}
    button[aria-pressed="true"] {{ color: #fff; background: #245a8d; border-color: #245a8d; }}
    #timeline {{ flex: 1 1 260px; min-width: 160px; }}
    .status {{ min-width: 190px; font-variant-numeric: tabular-nums; }}
    main {{ padding: 12px; }}
    .playback-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 10px; }}
    .panel {{ min-width: 0; padding: 8px; background: #fff; border: 1px solid #d5dbe0; border-radius: 6px; }}
    .panel.depth {{ border-top: 4px solid #245a8d; }}
    .panel.geometry {{ border-top: 4px solid #28734c; }}
    .panel h2 {{ margin: 0 0 7px; font-size: 14px; }}
    .panel img {{ display: block; width: 100%; min-height: 160px; object-fit: contain; background: #18212a; }}
    .panel-meta {{ margin-top: 6px; color: #4d5a66; font-size: 12px; }}
    .panel-error {{ display: none; padding: 12px; color: #8b1e1e; background: #fff0f0; }}
    @media (max-width: 760px) {{ .playback-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header class="toolbar">
    <a href="index.html">返回完整报告</a>
    <div class="controls" aria-label="帧控制">
      <button id="previous-button" type="button" title="上一帧" aria-label="上一帧">|&lt;</button>
      <button id="play-button" type="button" title="播放" aria-label="播放">Play</button>
      <button id="next-button" type="button" title="下一帧" aria-label="下一帧">&gt;|</button>
    </div>
    <div class="controls" aria-label="分割阶段">
      <button type="button" data-stage="initial" aria-pressed="false">初始分割</button>
      <button type="button" data-stage="merged" aria-pressed="true">融合后分割</button>
    </div>
    <label>速度 <select id="playback-speed"><option value="0.5">0.5 FPS</option><option value="1">1 FPS</option><option value="2" selected>2 FPS</option><option value="4">4 FPS</option></select></label>
    <input id="timeline" type="range" min="0" value="0" step="1" aria-label="时间轴">
    <span id="playback-status" class="status"></span>
  </header>
  <main>
    <div id="playback-error" class="panel-error" role="alert"></div>
    <div id="playback-grid" class="playback-grid">
      <section class="panel depth"><h2 id="previous-depth-title">BASELINE / DEPTH · 前一帧</h2><img id="previous-depth-image" alt="Baseline previous frame"><div id="previous-depth-meta" class="panel-meta"></div></section>
      <section class="panel geometry"><h2 id="previous-geometry-title">GEOMETRY · 前一帧</h2><img id="previous-geometry-image" alt="Geometry previous frame"><div id="previous-geometry-meta" class="panel-meta"></div></section>
      <section class="panel depth"><h2 id="current-depth-title">BASELINE / DEPTH · 当前帧</h2><img id="current-depth-image" alt="Baseline current frame"><div id="current-depth-meta" class="panel-meta"></div></section>
      <section class="panel geometry"><h2 id="current-geometry-title">GEOMETRY · 当前帧</h2><img id="current-geometry-image" alt="Geometry current frame"><div id="current-geometry-meta" class="panel-meta"></div></section>
    </div>
  </main>
  <script id="playback-data" type="application/json">{playback_json}</script>
</body>
</html>"""
```

Add a `播放序列` link with `href="player.html"` in the existing report header. In `build_report`, build the compact playback manifest and write:

```python
playback_manifest = build_playback_manifest(manifest)
(out_dir / "player.html").write_text(
    _build_player_html(playback_manifest),
    encoding="utf-8",
)
```

- [ ] **Step 4: Run the output test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Commit the player shell**

```bash
git add eval/build_alignment_pipeline_report.py tests/test_alignment_pipeline_report.py
git commit -m "feat: generate segmentation player page"
```

### Task 4: Implement Playback State and Controls

**Files:**
- Modify: `eval/build_alignment_pipeline_report.py`
- Test: `tests/test_alignment_pipeline_report.py`

- [ ] **Step 1: Add failing JavaScript-contract assertions**

Add these assertions to the full report test:

```python
assert "function renderPlayback()" in player_html
assert "function togglePlayback()" in player_html
assert "function preloadNextFrame()" in player_html
assert "setInterval" in player_html
assert "clearInterval" in player_html
assert "event.key === 'ArrowLeft'" in player_html
assert "event.key === 'ArrowRight'" in player_html
assert "event.code === 'Space'" in player_html
assert "frames.length - 1" in player_html
assert "image.onerror" in player_html
```

- [ ] **Step 2: Run the report test and verify RED**

Run the focused test from Task 3 Step 2.

Expected: failure at `function renderPlayback()`.

- [ ] **Step 3: Add the dependency-free player state machine**

Append a script after `playback-data` implementing these exact responsibilities:

```javascript
const playback = JSON.parse(document.getElementById('playback-data').textContent);
const frames = playback.frames;
let frameIndex = 0;
let stageName = 'merged';
let fps = 2;
let timer = null;

function stopPlayback() {
  if (timer !== null) clearInterval(timer);
  timer = null;
  document.getElementById('play-button').textContent = 'Play';
  document.getElementById('play-button').setAttribute('aria-label', '播放');
}

function resolveStage(frame, method) {
  return frame?.stages?.[stageName]?.[method] || null;
}

function renderPanel(prefix, frame, method, label, comparisonFrame = null) {
  const image = document.getElementById(`${prefix}-${method}-image`);
  const title = document.getElementById(`${prefix}-${method}-title`);
  const meta = document.getElementById(`${prefix}-${method}-meta`);
  const methodLabel = method === 'depth' ? 'BASELINE / DEPTH' : 'GEOMETRY';
  if (!frame) {
    image.removeAttribute('src');
    image.hidden = true;
    title.textContent = `${methodLabel} · ${label} · 无`;
    meta.textContent = '这是序列第一帧';
    return;
  }
  const stage = resolveStage(frame, method);
  if (!stage) throw new Error(`缺少 ${method} ${stageName}，G${frame.global_frame}`);
  image.hidden = false;
  image.onerror = () => {
    image.hidden = true;
    meta.textContent = `图片加载失败：${stage.asset}`;
  };
  image.src = stage.asset;
  title.textContent = `${methodLabel} · ${label} · G${String(frame.global_frame).padStart(5, '0')}`;
  const comparisonStage = resolveStage(comparisonFrame, method);
  const delta = comparisonStage ? stage.segment_count - comparisonStage.segment_count : null;
  const deltaText = delta === null ? '' : ` · Δ vs 前一帧: ${delta >= 0 ? '+' : ''}${delta}`;
  meta.textContent = `segments: ${stage.segment_count}${deltaText}`;
}

function preloadNextFrame() {
  const frame = frames[frameIndex + 1];
  if (!frame) return;
  for (const method of ['depth', 'geometry']) {
    const stage = resolveStage(frame, method);
    if (stage) new Image().src = stage.asset;
  }
}

function renderPlayback() {
  const error = document.getElementById('playback-error');
  error.style.display = 'none';
  if (!frames.length) {
    stopPlayback();
    error.textContent = '没有可播放的帧';
    error.style.display = 'block';
    return;
  }
  frameIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
  const current = frames[frameIndex];
  const previous = frameIndex > 0 ? frames[frameIndex - 1] : null;
  try {
    renderPanel('previous', previous, 'depth', '前一帧');
    renderPanel('previous', previous, 'geometry', '前一帧');
    renderPanel('current', current, 'depth', '当前帧', previous);
    renderPanel('current', current, 'geometry', '当前帧', previous);
  } catch (renderError) {
    stopPlayback();
    error.textContent = renderError.message;
    error.style.display = 'block';
  }
  const timeline = document.getElementById('timeline');
  timeline.max = String(frames.length - 1);
  timeline.value = String(frameIndex);
  document.getElementById('playback-status').textContent =
    `G${String(current.global_frame).padStart(5, '0')} · ${frameIndex + 1}/${frames.length}`;
  preloadNextFrame();
}

function stepFrame(delta) {
  stopPlayback();
  frameIndex = Math.max(0, Math.min(frameIndex + delta, frames.length - 1));
  renderPlayback();
}

function togglePlayback() {
  if (timer !== null) {
    stopPlayback();
    return;
  }
  if (!frames.length) return;
  if (frameIndex === frames.length - 1) frameIndex = 0;
  document.getElementById('play-button').textContent = 'Pause';
  document.getElementById('play-button').setAttribute('aria-label', '暂停');
  timer = setInterval(() => {
    if (frameIndex >= frames.length - 1) {
      stopPlayback();
      return;
    }
    frameIndex += 1;
    renderPlayback();
  }, 1000 / fps);
  renderPlayback();
}

document.getElementById('previous-button').addEventListener('click', () => stepFrame(-1));
document.getElementById('play-button').addEventListener('click', togglePlayback);
document.getElementById('next-button').addEventListener('click', () => stepFrame(1));
document.getElementById('timeline').addEventListener('input', (event) => {
  stopPlayback();
  frameIndex = Number(event.target.value);
  renderPlayback();
});
document.getElementById('playback-speed').addEventListener('change', (event) => {
  const wasPlaying = timer !== null;
  stopPlayback();
  fps = Number(event.target.value);
  if (wasPlaying) togglePlayback();
});
document.querySelectorAll('[data-stage]').forEach((button) => {
  button.addEventListener('click', () => {
    stageName = button.dataset.stage;
    document.querySelectorAll('[data-stage]').forEach((candidate) => {
      candidate.setAttribute('aria-pressed', String(candidate === button));
    });
    renderPlayback();
  });
});
document.addEventListener('keydown', (event) => {
  if (event.target.matches('input, select, button, a')) return;
  if (event.key === 'ArrowLeft') stepFrame(-1);
  if (event.key === 'ArrowRight') stepFrame(1);
  if (event.code === 'Space') {
    event.preventDefault();
    togglePlayback();
  }
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopPlayback();
});
renderPlayback();
```

- [ ] **Step 4: Run the complete report test module**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_pipeline_report.py -q
```

Expected: all report tests pass.

- [ ] **Step 5: Commit playback interactions**

```bash
git add eval/build_alignment_pipeline_report.py tests/test_alignment_pipeline_report.py
git commit -m "feat: add frame playback controls"
```

### Task 5: Documentation and End-to-End Verification

**Files:**
- Modify: `docs/CLOUD_RUN.md`
- Verify: `eval/build_alignment_pipeline_report.py`
- Verify: `tests/test_alignment_pipeline_report.py`

- [ ] **Step 1: Document report-only regeneration**

After the report build command in `docs/CLOUD_RUN.md`, add:

```markdown
The builder also writes `player.html`. It compares the previous and current
canonical global frames for Baseline/Depth and Geometry, with Initial/Merged
stage selection and frame controls. Open `/player.html` from the same server.

If the schema-v2 traces already exist, code updates to the report or player only
require rerunning `eval/build_alignment_pipeline_report.py`; model inference
does not need to run again.
```

- [ ] **Step 2: Run full automated verification**

Run:

```bash
conda run -n vggt-dem python -m pytest tests -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 3: Run syntax and diff checks**

Run:

```bash
conda run -n vggt-dem python -m py_compile eval/build_alignment_pipeline_report.py
git diff --check
```

Expected: both commands exit zero with no output.

- [ ] **Step 4: Build a synthetic report and verify in the browser**

Use the report test fixture output or a small schema-v2 report, serve its output
directory, then verify:

```text
1. index.html has a Play Sequence link.
2. player.html starts at merged stage and frame 1.
3. Previous panels are empty at frame 1.
4. Next/Previous and Left/Right update all four panels.
5. Play advances and pauses at the final frame.
6. Timeline seeking updates the adjacent-frame pair.
7. Initial/Merged switching preserves the position.
8. Changing FPS while playing restarts the timer at the new rate.
9. Space toggles playback.
10. Desktop is 2x2 and a viewport below 760 px stacks panels.
11. Browser console has no errors.
```

- [ ] **Step 5: Commit documentation and final verification state**

```bash
git add docs/CLOUD_RUN.md
git commit -m "docs: explain segmentation player workflow"
```

- [ ] **Step 6: Push the completed implementation**

```bash
git push origin main
```
