"""Command-line interface for Basketball Vision AI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from basketball_vision.cli.config import CLIConfigError, load_config_file
from basketball_vision.cli.export import export_json
from basketball_vision.pipeline import BasketballVisionPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="basketball-vision",
        description="Basketball Vision AI command-line tools.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process-video",
        help="Run the Basketball Vision pipeline on a video.",
    )
    process_parser.add_argument(
        "input_video",
        type=str,
        help="Path to the input video.",
    )
    process_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where output artifacts should be written.",
    )
    process_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON/YAML config file.",
    )
    process_parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path for JSON result export.",
    )
    process_parser.add_argument(
        "--annotated-video",
        type=str,
        default=None,
        help="Optional path for annotated video output.",
    )
    process_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )

    return parser


def create_pipeline(config_data: dict[str, Any]) -> BasketballVisionPipeline:
    """Create the main pipeline from config data."""
    pipeline_config_data = config_data.get("pipeline", config_data)

    if pipeline_config_data:
        config = PipelineConfig(**pipeline_config_data)
    else:
        config = PipelineConfig()

    return BasketballVisionPipeline(config=config)


def run_process_video(args: argparse.Namespace) -> int:
    """Run the process-video command."""
    input_video = Path(args.input_video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        print(f"Input video does not exist: {input_video}", file=sys.stderr)
        return 1

    try:
        config_data = load_config_file(args.config) if args.config else {}
    except (CLIConfigError, OSError, ValueError) as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 1

    pipeline = create_pipeline(config_data)

    try:
        results = pipeline.process_video(
            video_path=input_video,
            output_dir=output_dir,
            annotated_video_path=args.annotated_video,
            max_frames=args.max_frames,
        )
    except TypeError:
        # Fallback for older/simple pipeline signatures.
        results = pipeline.process_video(input_video)

    json_output = args.json_output
    if json_output is None:
        json_output = output_dir / "results.json"

    export_json(results, json_output)

    print(f"Processed video: {input_video}")
    print(f"Output directory: {output_dir}")
    print(f"JSON results: {json_output}")

    if args.annotated_video:
        print(f"Annotated video: {args.annotated_video}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "process-video":
        return run_process_video(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
