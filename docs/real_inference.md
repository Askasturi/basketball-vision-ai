
# Real YOLO Inference

Milestone 12 adds a real inference runner while preserving the existing mock demo.

## Mock mode

Use this when you do not have YOLO weights or a real basketball video ready:

```bash
PYTHONPATH=src python examples/run_pipeline.py \
  --input outputs/demo/full_pipeline_demo.mp4 \
  --output outputs/demo/mock_pipeline_check.mp4 \
  --mock \
  --max-frames 30
