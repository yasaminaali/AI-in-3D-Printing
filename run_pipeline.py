#!/usr/bin/env python3
"""
Entry point script for running the SA dataset generation pipeline.

Usage:
    python run_pipeline.py <machine_id> [--workers N] [--retry-failed] [--config-dir DIR]

Examples:
    python run_pipeline.py yasamin --workers 8
    python run_pipeline.py istiaq
    python run_pipeline.py kazi --retry-failed
"""

import argparse
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run SA dataset generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py yasamin              # Run Yasamin's tasks with default workers
  python run_pipeline.py istiaq --workers 4   # Run Istiaq's tasks with 4 workers
  python run_pipeline.py kazi --retry-failed  # Retry failed tasks for Kazi
        """,
    )

    parser.add_argument(
        "machine_id",
        type=str,
        help="Machine identifier (e.g., yasamin, istiaq, kazi)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: from config or CPU count)",
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing YAML config files (default: config)",
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed tasks",
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="Seconds between progress logs (default: 60)",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status and exit without running",
    )

    args = parser.parse_args()

    # Resolve config directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, args.config_dir)

    if not os.path.isdir(config_dir):
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    # Check if machine config exists
    machine_config_path = os.path.join(config_dir, f"{args.machine_id}.yaml")
    if not os.path.exists(machine_config_path):
        print(f"Error: Machine config not found: {machine_config_path}")
        available = [f.replace(".yaml", "") for f in os.listdir(config_dir) if f.endswith(".yaml") and f != "global_config.yaml"]
        print(f"Available machines: {', '.join(available)}")
        sys.exit(1)

    if args.status:
        # Show status only
        from pipeline.runner import ParallelRunner

        runner = ParallelRunner(
            config_dir=config_dir,
            machine_id=args.machine_id,
            num_workers=args.workers,
        )
        status = runner.get_status()

        print(f"\n=== Status for {args.machine_id} ===")
        print(f"Total tasks: {status['total_tasks']}")
        print(f"Completed: {status['progress']['completed']}")
        print(f"Failed: {status['progress']['failed']}")
        print(f"Remaining: {status['progress']['remaining']}")
        print(f"Progress: {status['progress']['percent']:.1f}%")
        return

    # Run the pipeline
    print(f"\n=== SA Dataset Generation Pipeline ===")
    print(f"Machine: {args.machine_id}")
    print(f"Config dir: {config_dir}")
    print()

    try:
        stats = run_pipeline(
            config_dir=config_dir,
            machine_id=args.machine_id,
            num_workers=args.workers,
            retry_failed=args.retry_failed,
            progress_interval=args.progress_interval,
        )

        # Exit with error code if there were failures
        if stats.get("failed_count", 0) > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
