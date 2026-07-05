# Basketball Analytics

## Overview

The analytics module computes player-level and game-level statistics from the existing Basketball Vision AI pipeline.

It does not perform detection, tracking, classification, or recognition itself. Instead, it consumes the outputs already produced by the pipeline.

## Architecture

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
Renderer Overlay
    ↓
CSV / JSON Export
```

## Player Statistics

The analytics engine tracks:

- Track ID
- Team
- Jersey number
- Frames seen
- First frame
- Last frame
- Time visible
- Average detection confidence
- Average bounding box area
- Distance travelled in pixels
- Average speed
- Maximum speed

## Game Statistics

The analytics engine tracks:

- Total frames
- Video length in seconds
- Number of players
- Average players on court
- Ball detections
- Team counts
- Estimated ball possession
- Detection counts

## CSV Output

Player statistics are exported to:

```text
outputs/analytics/player_statistics.csv
```

Columns:

```text
Track ID
Team
Jersey Number
Frames Seen
Time Visible
Distance
Average Speed
Average Confidence
```

## JSON Output

Game statistics are exported to:

```text
outputs/analytics/game_statistics.json
```

Example keys:

```text
frames
video_seconds
number_of_players
average_players_on_court
ball_detections
team_counts
estimated_ball_possession
detection_counts
players
```

## Renderer Overlay

When analytics data is attached to a frame result, the renderer overlays:

```text
Players: 10
Blue Team: 5
White Team: 5
Frame: 245
FPS: 30
```

## Run the Analytics Demo

```bash
PYTHONPATH=src python examples/analytics_demo.py
```

Expected outputs:

```text
outputs/demo/analytics_demo.mp4
outputs/analytics/player_statistics.csv
outputs/analytics/game_statistics.json
```

## Verify

```bash
PYTHONPATH=src ruff check .
PYTHONPATH=src pytest
```
