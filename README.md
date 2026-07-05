# Basketball Vision AI

Basketball Vision AI is a modular, production-quality computer vision framework for basketball analysis. The project is built milestone-by-milestone with a strong emphasis on clean architecture, testing, maintainability, and extensibility.

---

# Project Progress

### ✅ Milestone 0 — Repository Setup

Completed:

- Repository initialization
- Python environment
- Project structure
- Development tooling
- Black
- Ruff
- Pytest
- Git configuration
- Initial documentation

---

### ✅ Milestone 1 — Video Infrastructure

Implemented:

- Video loading
- Video metadata
- Frame iteration
- Video writing
- Immutable video data models
- Context manager support
- Custom exceptions
- Production-quality APIs
- Unit tests
- Integration tests

---

### ✅ Milestone 2 — Detection Framework

Implemented:

- Generic detector architecture
- Abstract `BaseDetector`
- Immutable detection models
- Detector configuration
- Detector factory
- Registry pattern
- Device abstraction
- Strong typing using enums
- Context manager support
- Detection exceptions
- Comprehensive unit tests

---

### ✅ Milestone 3 — YOLO Detector Integration

Implemented:

- Ultralytics YOLO backend
- YOLO-specific configuration
- Detection conversion layer
- YOLO detector implementation
- Factory integration
- Example detection pipeline
- Integration tests
- Mocked inference tests
- End-to-end detection workflow
- 62 passing tests

---

### ✅ Milestone 4 — Multi-Object Tracking

Implemented:

- Simple multi-object tracker
- Track lifecycle management
- Immutable tracking models
- Tracker configuration
- Tracker factory
- Active/lost/removed track states
- Comprehensive unit tests

---

### ✅ Milestone 5 — Team Classification

Implemented:

- Color-based team classification
- Team assignment models
- Classification configuration
- Classification factory
- Classification exceptions
- Deterministic unit tests

---

### ✅ Milestone 6 — Jersey Number Recognition

Implemented:

- Number recognition framework
- Recognition configuration
- Recognition factory
- Immutable recognition models
- Simple number recognizer
- Comprehensive unit tests

---

### ✅ Milestone 7 — Visualization Framework

Implemented:

- Renderer
- Drawing utilities
- Color palette
- Visualization configuration
- Visualization factory
- Frame annotation
- Comprehensive visualization tests

---

### ✅ Milestone 8 — Pipeline Framework

Implemented:

- Unified processing pipeline
- Pipeline configuration
- Pipeline factory
- Pipeline result objects
- Pipeline exceptions
- Integration tests

---

### ✅ Milestone 9 — Command Line Interface

Implemented:

- CLI configuration
- Export utilities
- Command-line entry point
- CLI tests

---

### ✅ Milestone 10 — Real YOLO Integration

Implemented:

- Real YOLO detector
- Real inference support
- Integration tests
- Pipeline validation

---

### ✅ Milestone 11 — End-to-End Rendering Pipeline

Implemented:

- Rendering integration
- Full visualization pipeline
- Demo rendering
- Output generation
- End-to-end testing

---

### ✅ Milestone 12 — Runnable Library Pipeline

Implemented:

- Library-integrated pipeline runner
- Existing module integration
- Mock and real pipeline support
- Demo video generation
- Documentation
- Stable pipeline execution

---

### ✅ Milestone 13 — Basketball Analytics

Implemented:

- Analytics package
- Analytics configuration
- Analytics factory
- Analytics engine
- Player statistics
- Game statistics
- Immutable analytics result models
- CSV export
- JSON export
- Renderer analytics overlay
- Analytics demo
- Analytics documentation
- Comprehensive deterministic unit tests

---

# Current Architecture

```text
Video
    ↓
YOLO Detector
    ↓
DetectionResult
    ↓
SimpleTracker
    ↓
TrackingResult
    ↓
ColorTeamClassifier
    ↓
ClassificationResult
    ↓
SimpleNumberRecognizer
    ↓
RecognitionResult
    ↓
AnalyticsEngine
    ↓
AnalyticsResult
    ↓
Renderer
    ↓
Annotated Video Export
    ↓
CSV / JSON Statistics
```

---

# Analytics Outputs

Running the analytics demo generates:

```text
outputs/demo/analytics_demo.mp4

outputs/analytics/player_statistics.csv

outputs/analytics/game_statistics.json
```

### Player Statistics CSV

Contains:

- Track ID
- Team
- Jersey Number
- Frames Seen
- First Frame
- Last Frame
- Time Visible
- Average Detection Confidence
- Average Bounding Box Area
- Distance Travelled
- Average Speed
- Maximum Speed

### Game Statistics JSON

Contains:

- Total Frames
- Video Length
- Number of Players
- Average Players On Court
- Ball Detections
- Team Counts
- Estimated Ball Possession
- Detection Counts
- Player Statistics

---

# Running the Analytics Demo

```bash
PYTHONPATH=src python examples/analytics_demo.py
```

Generated files:

```text
outputs/demo/analytics_demo.mp4

outputs/analytics/player_statistics.csv

outputs/analytics/game_statistics.json
```

---

# Running the Library Pipeline

```bash
PYTHONPATH=src python examples/run_pipeline.py \
    --input tests/assets/sample.mp4
```

---

# Development

Verify the repository:

```bash
PYTHONPATH=src ruff check .
PYTHONPATH=src pytest
```

Current verified status:

- Ruff ✅
- Pytest ✅ 226 passing tests

---

# Documentation

Available documentation:

- `docs/pipeline_demo.md`
- `docs/real_inference.md`
- `docs/testing.md`
- `docs/visualization.md`
- `docs/analytics.md`

---

# Repository Status

Current milestone:

**Milestone 13 — Basketball Analytics**

Verified outputs:

```text
outputs/demo/render_demo.jpg

outputs/demo/full_pipeline_demo.mp4

outputs/demo/mock_pipeline_check.mp4

outputs/demo/library_pipeline_check.mp4

outputs/demo/analytics_demo.mp4

outputs/analytics/player_statistics.csv

outputs/analytics/game_statistics.json
```
