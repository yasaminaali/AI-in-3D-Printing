#!/usr/bin/env python3
"""
Merge utility for combining SA datasets from multiple machines.

Usage:
    python merge_datasets.py --inputs DIR1 DIR2 DIR3 --output OUTPUT_PATH

Examples:
    python merge_datasets.py --inputs output/yasamin output/istiaq output/kazi --output output/final/Dataset.jsonl
    python merge_datasets.py --inputs output/ --output merged.jsonl --stats
"""

import argparse
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.merge import merge_datasets, read_jsonl, get_dataset_stats, validate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Merge SA datasets from multiple directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_datasets.py --inputs output/yasamin output/istiaq --output output/merged.jsonl
  python merge_datasets.py --inputs output/ --output final.jsonl --no-deduplicate
  python merge_datasets.py --stats output/final/Dataset.jsonl
        """,
    )

    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        help="Input directories containing Dataset.jsonl files",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path for merged dataset",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="Dataset.jsonl",
        help="Name of JSONL files to merge (default: Dataset.jsonl)",
    )

    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Don't remove duplicate records",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Don't validate merged dataset",
    )

    parser.add_argument(
        "--stats",
        type=str,
        nargs="?",
        const="",
        help="Show statistics for a dataset file (or merged output if no path given)",
    )

    parser.add_argument(
        "--validate-only",
        type=str,
        help="Validate a dataset file without merging",
    )

    args = parser.parse_args()

    # Validate-only mode
    if args.validate_only:
        print(f"\n=== Validating {args.validate_only} ===")
        if not os.path.exists(args.validate_only):
            print(f"Error: File not found: {args.validate_only}")
            sys.exit(1)

        records = read_jsonl(args.validate_only)
        print(f"Records: {len(records)}")

        validation = validate_dataset(records)
        print(f"Valid: {validation['valid_records']}")
        print(f"Invalid: {validation['invalid_records']}")
        print(f"Duplicates: {validation['duplicate_run_ids']}")

        if validation["errors"]:
            print("\nErrors:")
            for error in validation["errors"][:20]:
                print(f"  - {error}")
            if len(validation["errors"]) > 20:
                print(f"  ... and {len(validation['errors']) - 20} more")

        sys.exit(0 if validation["is_valid"] else 1)

    # Stats mode
    if args.stats is not None:
        stats_path = args.stats if args.stats else args.output
        if not stats_path:
            print("Error: Provide a file path for --stats or use with --output")
            sys.exit(1)

        if not os.path.exists(stats_path):
            print(f"Error: File not found: {stats_path}")
            sys.exit(1)

        print(f"\n=== Statistics for {stats_path} ===")
        records = read_jsonl(stats_path)
        stats = get_dataset_stats(records)

        print(f"Total records: {stats['total']}")
        print(f"\nBy grid size:")
        for grid, count in sorted(stats["by_grid"].items()):
            print(f"  {grid}: {count}")

        print(f"\nBy zone pattern:")
        for pattern, count in sorted(stats["by_pattern"].items()):
            print(f"  {pattern}: {count}")

        print(f"\nCrossings:")
        print(f"  Initial: min={stats['crossings']['initial']['min']}, "
              f"max={stats['crossings']['initial']['max']}, "
              f"avg={stats['crossings']['initial']['avg']:.1f}")
        print(f"  Final:   min={stats['crossings']['final']['min']}, "
              f"max={stats['crossings']['final']['max']}, "
              f"avg={stats['crossings']['final']['avg']:.1f}")

        print(f"\nSequence length:")
        print(f"  min={stats['sequence_len']['min']}, "
              f"max={stats['sequence_len']['max']}, "
              f"avg={stats['sequence_len']['avg']:.1f}")

        return

    # Merge mode - require inputs and output
    if not args.inputs or not args.output:
        parser.print_help()
        print("\nError: --inputs and --output are required for merging")
        sys.exit(1)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dirs = [
        os.path.join(script_dir, d) if not os.path.isabs(d) else d
        for d in args.inputs
    ]
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    # Check input directories exist
    for d in input_dirs:
        if not os.path.exists(d):
            print(f"Warning: Input directory not found: {d}")

    print(f"\n=== Merging SA Datasets ===")
    print(f"Inputs: {input_dirs}")
    print(f"Output: {output_path}")
    print()

    try:
        report = merge_datasets(
            input_dirs=input_dirs,
            output_path=output_path,
            deduplicate=not args.no_deduplicate,
            validate=not args.no_validate,
            filename=args.filename,
        )

        print(f"\n=== Merge Summary ===")
        print(f"Files merged: {len(report['files_found'])}")
        print(f"Records read: {report['total_records_read']}")
        print(f"Duplicates removed: {report['duplicates_removed']}")
        print(f"Final count: {report['final_record_count']}")

        if report["validation"]:
            if report["validation"]["is_valid"]:
                print("Validation: PASSED")
            else:
                print("Validation: FAILED")
                sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
