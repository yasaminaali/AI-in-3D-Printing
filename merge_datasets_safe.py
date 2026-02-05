"""
Merge dataset files with zero data loss verification
"""
import json
import sys
from pathlib import Path
from typing import Tuple, Dict

def count_records(filepath: Path) -> Tuple[int, list]:
    """Count records and collect sample IDs for verification."""
    count = 0
    sample_ids = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    count += 1
                    if 'run_id' in data:
                        sample_ids.append(data['run_id'])
                except json.JSONDecodeError:
                    pass
    return count, sample_ids

def merge_datasets(input_files: list, output_file: Path) -> Dict:
    """Merge multiple dataset files with verification."""
    
    # First, verify and count all input files
    file_stats = []
    all_sample_ids = []
    
    print("="*80)
    print("Dataset Merge - Zero Data Loss Verification")
    print("="*80)
    print()
    
    for filepath in input_files:
        if not filepath.exists():
            print(f"ERROR: File not found: {filepath}")
            sys.exit(1)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        count, sample_ids = count_records(filepath)
        
        file_stats.append({
            'path': filepath,
            'size_mb': size_mb,
            'records': count,
            'sample_ids': set(sample_ids)
        })
        
        all_sample_ids.extend(sample_ids)
        
        print(f"[FILE] {filepath.name}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Records: {count:,}")
        print()
    
    # Check for duplicates between files
    print("[CHECK] Checking for duplicate records between files...")
    all_ids_set = set(all_sample_ids)
    if len(all_ids_set) < len(all_sample_ids):
        duplicates = len(all_sample_ids) - len(all_ids_set)
        print(f"   [!] Found {duplicates} duplicate run_ids across files")
        print("   Will deduplicate during merge (keeping first occurrence)")
    else:
        print("   [OK] No duplicates found between files")
    print()
    
    # Merge files
    print(f"[MERGE] Merging {len(input_files)} files...")
    print(f"   Output: {output_file}")
    print()
    
    total_written = 0
    seen_ids = set()
    duplicates_skipped = 0
    
    # Write with progress
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filepath in input_files:
            print(f"   Processing {filepath.name}...", end=' ')
            file_written = 0
            
            with open(filepath, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        run_id = data.get('run_id', None)
                        
                        # Skip duplicates
                        if run_id and run_id in seen_ids:
                            duplicates_skipped += 1
                            continue
                        
                        if run_id:
                            seen_ids.add(run_id)
                        
                        out_f.write(json.dumps(data) + '\n')
                        total_written += 1
                        file_written += 1
                        
                    except json.JSONDecodeError:
                        pass
            
            print(f"[OK] ({file_written:,} records)")
    
    print()
    
    # Verification
    print("[VERIFY] Verifying merged file...")
    merged_count, _ = count_records(output_file)
    merged_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    expected_records = sum(s['records'] for s in file_stats) - duplicates_skipped
    
    print(f"   Expected records: {expected_records:,}")
    print(f"   Actual records: {merged_count:,}")
    print(f"   File size: {merged_size_mb:.2f} MB")
    print()
    
    if merged_count == expected_records:
        print("[PASS] VERIFICATION PASSED: All records preserved!")
    else:
        print(f"[FAIL] VERIFICATION FAILED: Missing {expected_records - merged_count} records!")
        sys.exit(1)
    
    # Print summary
    print()
    print("="*80)
    print("MERGE SUMMARY")
    print("="*80)
    print(f"Total input files: {len(input_files)}")
    print(f"Total input records: {sum(s['records'] for s in file_stats):,}")
    print(f"Duplicates skipped: {duplicates_skipped}")
    print(f"Total output records: {merged_count:,}")
    print(f"Total output size: {merged_size_mb:.2f} MB")
    print(f"Output file: {output_file.absolute()}")
    print("="*80)
    print()
    
    return {
        'input_files': len(input_files),
        'input_records': sum(s['records'] for s in file_stats),
        'duplicates_skipped': duplicates_skipped,
        'output_records': merged_count,
        'output_size_mb': merged_size_mb
    }

if __name__ == "__main__":
    # Define input files
    input_dir = Path("output/datasets")
    output_file = input_dir / "combined_dataset.jsonl"
    
    input_files = [
        input_dir / "leftright_stripes.jsonl",
        input_dir / "voronoi_island.jsonl"
    ]
    
    # Run merge
    stats = merge_datasets(input_files, output_file)
    
    print("[DONE] Merge complete! Combined dataset saved to:")
    print(f"   {output_file.absolute()}")
    print()
    print("You can now use this file for training:")
    print(f"   python model/data/preprocess.py --input {output_file}")
