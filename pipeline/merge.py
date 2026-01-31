"""
Dataset merge utility for combining JSONL files from multiple machines.
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read all records from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of record dictionaries
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"[Warning] Invalid JSON at {path}:{line_num}: {e}")
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """
    Write records to a JSONL file.

    Args:
        path: Output path
        records: List of record dictionaries
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def find_jsonl_files(directories: List[str], filename: str = "Dataset.jsonl") -> List[str]:
    """
    Find all JSONL files in given directories.

    Args:
        directories: List of directories to search
        filename: Name of JSONL file to find

    Returns:
        List of paths to found JSONL files
    """
    paths = []
    for directory in directories:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            paths.append(path)
        else:
            # Also check subdirectories
            for root, dirs, files in os.walk(directory):
                if filename in files:
                    paths.append(os.path.join(root, filename))
    return paths


def validate_record(record: Dict[str, Any], required_fields: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate a single record.

    Args:
        record: Record dictionary
        required_fields: List of required field names

    Returns:
        (is_valid, error_message)
    """
    if required_fields is None:
        required_fields = [
            "run_id",
            "seed",
            "grid_W",
            "grid_H",
            "zone_pattern",
            "zone_grid",
            "initial_crossings",
            "final_crossings",
            "sequence_len",
            "sequence_ops",
        ]

    for field in required_fields:
        if field not in record:
            return False, f"Missing required field: {field}"

    # Type checks
    if not isinstance(record.get("grid_W"), int) or record["grid_W"] <= 0:
        return False, "grid_W must be positive integer"

    if not isinstance(record.get("grid_H"), int) or record["grid_H"] <= 0:
        return False, "grid_H must be positive integer"

    if not isinstance(record.get("zone_grid"), list):
        return False, "zone_grid must be a list"

    expected_size = record["grid_W"] * record["grid_H"]
    if len(record["zone_grid"]) != expected_size:
        return False, f"zone_grid size mismatch: expected {expected_size}, got {len(record['zone_grid'])}"

    if not isinstance(record.get("sequence_ops"), list):
        return False, "sequence_ops must be a list"

    if len(record["sequence_ops"]) != record.get("sequence_len", -1):
        return False, f"sequence_len mismatch: {record.get('sequence_len')} != {len(record['sequence_ops'])}"

    return True, None


def validate_dataset(
    records: List[Dict[str, Any]],
    check_duplicates: bool = True,
) -> Dict[str, Any]:
    """
    Validate a dataset.

    Args:
        records: List of record dictionaries
        check_duplicates: If True, check for duplicate run_ids

    Returns:
        Validation report dictionary
    """
    report = {
        "total_records": len(records),
        "valid_records": 0,
        "invalid_records": 0,
        "duplicate_run_ids": 0,
        "errors": [],
        "is_valid": True,
    }

    seen_run_ids: Set[str] = set()

    for i, record in enumerate(records):
        is_valid, error = validate_record(record)

        if is_valid:
            report["valid_records"] += 1
        else:
            report["invalid_records"] += 1
            report["errors"].append(f"Record {i}: {error}")
            report["is_valid"] = False

        if check_duplicates:
            run_id = record.get("run_id", "")
            if run_id in seen_run_ids:
                report["duplicate_run_ids"] += 1
                report["errors"].append(f"Record {i}: Duplicate run_id '{run_id}'")
            seen_run_ids.add(run_id)

    if report["duplicate_run_ids"] > 0:
        report["is_valid"] = False

    return report


def merge_datasets(
    input_dirs: List[str],
    output_path: str,
    deduplicate: bool = True,
    validate: bool = True,
    filename: str = "Dataset.jsonl",
) -> Dict[str, Any]:
    """
    Merge JSONL datasets from multiple directories.

    Args:
        input_dirs: List of input directories
        output_path: Path for merged output file
        deduplicate: If True, remove duplicate run_ids (keep first)
        validate: If True, validate merged dataset
        filename: Name of JSONL files to merge

    Returns:
        Merge report dictionary
    """
    report = {
        "input_dirs": input_dirs,
        "output_path": output_path,
        "files_found": [],
        "records_per_file": {},
        "total_records_read": 0,
        "duplicates_removed": 0,
        "final_record_count": 0,
        "validation": None,
    }

    # Find all JSONL files
    jsonl_files = find_jsonl_files(input_dirs, filename)
    report["files_found"] = jsonl_files

    if not jsonl_files:
        print(f"[Merge] No {filename} files found in: {input_dirs}")
        return report

    print(f"[Merge] Found {len(jsonl_files)} files to merge")

    # Read all records
    all_records: List[Dict[str, Any]] = []
    for path in jsonl_files:
        records = read_jsonl(path)
        report["records_per_file"][path] = len(records)
        all_records.extend(records)
        print(f"  - {path}: {len(records)} records")

    report["total_records_read"] = len(all_records)
    print(f"[Merge] Total records read: {len(all_records)}")

    # Deduplicate
    if deduplicate:
        seen_ids: Set[str] = set()
        unique_records: List[Dict[str, Any]] = []

        for record in all_records:
            run_id = record.get("run_id", "")
            if run_id not in seen_ids:
                unique_records.append(record)
                seen_ids.add(run_id)

        duplicates = len(all_records) - len(unique_records)
        report["duplicates_removed"] = duplicates
        all_records = unique_records

        if duplicates > 0:
            print(f"[Merge] Removed {duplicates} duplicate records")

    report["final_record_count"] = len(all_records)

    # Validate
    if validate:
        print("[Merge] Validating merged dataset...")
        validation = validate_dataset(all_records, check_duplicates=False)
        report["validation"] = validation

        if validation["is_valid"]:
            print(f"[Merge] Validation passed: {validation['valid_records']} valid records")
        else:
            print(f"[Merge] Validation failed: {validation['invalid_records']} invalid records")
            for error in validation["errors"][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(validation["errors"]) > 10:
                print(f"  ... and {len(validation['errors']) - 10} more errors")

    # Write output
    print(f"[Merge] Writing {len(all_records)} records to {output_path}")
    write_jsonl(output_path, all_records)

    print("[Merge] Done!")
    return report


def get_dataset_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about a dataset.

    Args:
        records: List of record dictionaries

    Returns:
        Statistics dictionary
    """
    if not records:
        return {"total": 0}

    stats = {
        "total": len(records),
        "by_grid": {},
        "by_pattern": {},
        "crossings": {
            "initial": {"min": float("inf"), "max": 0, "sum": 0},
            "final": {"min": float("inf"), "max": 0, "sum": 0},
        },
        "sequence_len": {"min": float("inf"), "max": 0, "sum": 0},
    }

    for record in records:
        # Grid stats
        grid_key = f"{record.get('grid_W', 0)}x{record.get('grid_H', 0)}"
        stats["by_grid"][grid_key] = stats["by_grid"].get(grid_key, 0) + 1

        # Pattern stats
        pattern = record.get("zone_pattern", "unknown")
        stats["by_pattern"][pattern] = stats["by_pattern"].get(pattern, 0) + 1

        # Crossings stats
        initial = record.get("initial_crossings", 0)
        final = record.get("final_crossings", 0)

        stats["crossings"]["initial"]["min"] = min(stats["crossings"]["initial"]["min"], initial)
        stats["crossings"]["initial"]["max"] = max(stats["crossings"]["initial"]["max"], initial)
        stats["crossings"]["initial"]["sum"] += initial

        stats["crossings"]["final"]["min"] = min(stats["crossings"]["final"]["min"], final)
        stats["crossings"]["final"]["max"] = max(stats["crossings"]["final"]["max"], final)
        stats["crossings"]["final"]["sum"] += final

        # Sequence length stats
        seq_len = record.get("sequence_len", 0)
        stats["sequence_len"]["min"] = min(stats["sequence_len"]["min"], seq_len)
        stats["sequence_len"]["max"] = max(stats["sequence_len"]["max"], seq_len)
        stats["sequence_len"]["sum"] += seq_len

    # Compute averages
    n = len(records)
    stats["crossings"]["initial"]["avg"] = stats["crossings"]["initial"]["sum"] / n
    stats["crossings"]["final"]["avg"] = stats["crossings"]["final"]["sum"] / n
    stats["sequence_len"]["avg"] = stats["sequence_len"]["sum"] / n

    return stats
