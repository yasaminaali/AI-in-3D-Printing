"""
Dataset Preprocessing for CNN+RNN Training
Paper-compliant stratified split by grid size categories
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class DatasetRecord:
    """Structured dataset record."""
    run_id: str
    grid_W: int
    grid_H: int
    zone_pattern: str
    initial_crossings: int
    final_crossings: int
    sequence_ops: List[Dict]
    improvement: int
    improvement_ratio: float
    data: Dict


class DatasetPreprocessor:
    """Preprocess dataset with stratified split per grid size."""
    
    def __init__(self, config_path: str = "model/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.output_dir = Path(self.data_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_records': 0,
            'filtered_out': 0,
            'grid_size_distribution': defaultdict(int),
            'zone_pattern_distribution': defaultdict(int),
            'split_distribution': defaultdict(lambda: defaultdict(int))
        }
    
    def load_dataset(self, dataset_path: str) -> List[DatasetRecord]:
        """Load and parse dataset records."""
        records = []
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            print(f"Error: Dataset file not found: {dataset_path}")
            return records
        
        with open(dataset_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    initial = data.get('initial_crossings', 0)
                    final = data.get('final_crossings', 0)
                    improvement = initial - final
                    improvement_ratio = improvement / initial if initial > 0 else 0
                    
                    record = DatasetRecord(
                        run_id=data.get('run_id', f'record_{line_num}'),
                        grid_W=data.get('grid_W', 0),
                        grid_H=data.get('grid_H', 0),
                        zone_pattern=data.get('zone_pattern', 'unknown'),
                        initial_crossings=initial,
                        final_crossings=final,
                        sequence_ops=data.get('sequence_ops', []),
                        improvement=improvement,
                        improvement_ratio=improvement_ratio,
                        data=data
                    )
                    
                    records.append(record)
                    
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
        
        self.stats['total_records'] = len(records)
        print(f"Loaded {len(records)} records")
        return records
    
    def filter_records(self, records: List[DatasetRecord]) -> List[DatasetRecord]:
        """Filter out records with no improvement."""
        filter_config = self.data_config['filter']
        min_improvement = filter_config['min_improvement']
        min_ratio = filter_config['min_crossing_reduction_ratio']
        max_seq_len = filter_config['max_sequence_length']
        
        filtered = []
        
        for record in records:
            if record.improvement < min_improvement:
                self.stats['filtered_out'] += 1
                continue
            
            if record.improvement_ratio < min_ratio:
                self.stats['filtered_out'] += 1
                continue
            
            if len(record.sequence_ops) > max_seq_len:
                self.stats['filtered_out'] += 1
                continue
            
            filtered.append(record)
        
        print(f"Filtered: {len(filtered)}/{len(records)} records retained")
        print(f"Discarded: {self.stats['filtered_out']} records")
        
        return filtered
    
    def group_by_grid_size(self, records: List[DatasetRecord]) -> Dict[Tuple[int, int], List[DatasetRecord]]:
        """Group records by grid size."""
        grouped = defaultdict(list)
        
        for record in records:
            grid_size = (record.grid_W, record.grid_H)
            grouped[grid_size].append(record)
            self.stats['grid_size_distribution'][grid_size] += 1
            self.stats['zone_pattern_distribution'][record.zone_pattern] += 1
        
        return dict(grouped)
    
    def stratified_split(self, records: List[DatasetRecord]) -> Tuple[List[DatasetRecord], List[DatasetRecord], List[DatasetRecord]]:
        """Create stratified 80/10/10 split sorted by performance."""
        split_config = self.data_config['split']
        train_ratio = split_config['train_ratio']
        val_ratio = split_config['val_ratio']
        
        # Sort by performance (best first)
        sorted_records = sorted(records, key=lambda r: (r.final_crossings, -r.improvement_ratio))
        
        n = len(sorted_records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_records = sorted_records[:n_train]
        val_records = sorted_records[n_train:n_train + n_val]
        test_records = sorted_records[n_train + n_val:]
        
        return train_records, val_records, test_records
    
    def save_split(self, records: List[DatasetRecord], split_name: str, grid_size: Tuple[int, int]):
        """Save a data split to file."""
        grid_dir = self.output_dir / f"{grid_size[0]}x{grid_size[1]}"
        grid_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = grid_dir / f"{split_name}.jsonl"
        
        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record.data) + '\n')
        
        self.stats['split_distribution'][grid_size][split_name] = len(records)
        return output_file
    
    def process_dataset(self, dataset_path = None):
        """Main preprocessing pipeline."""
        print("\n=== CNN+RNN Dataset Preprocessing ===\n")
        
        if dataset_path is None:
            dataset_path = self.data_config['dataset_path']
        records = self.load_dataset(dataset_path)
        
        if not records:
            print("No records loaded. Exiting.")
            return
        
        records = self.filter_records(records)
        grouped = self.group_by_grid_size(records)
        
        print(f"\nGrid Size Distribution:")
        for grid_size in sorted(grouped.keys()):
            count = len(grouped[grid_size])
            patterns = set(r.zone_pattern for r in grouped[grid_size])
            print(f"  {grid_size[0]}x{grid_size[1]}: {count} records ({', '.join(sorted(patterns))})")
        
        print(f"\nProcessing {len(grouped)} grid size categories...")
        
        all_train_files = []
        all_val_files = []
        all_test_files = []
        
        for grid_size in sorted(grouped.keys()):
            group_records = grouped[grid_size]
            
            print(f"\n{grid_size[0]}x{grid_size[1]}: {len(group_records)} records")
            
            train, val, test = self.stratified_split(group_records)
            
            train_file = self.save_split(train, 'train', grid_size)
            val_file = self.save_split(val, 'val', grid_size)
            test_file = self.save_split(test, 'test', grid_size)
            
            all_train_files.append(train_file)
            all_val_files.append(val_file)
            all_test_files.append(test_file)
            
            print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        
        # Create combined files
        print(f"\nCreating combined splits...")
        combined_train = self.output_dir / "train_all.jsonl"
        combined_val = self.output_dir / "val_all.jsonl"
        combined_test = self.output_dir / "test_all.jsonl"
        
        self._combine_files(all_train_files, combined_train)
        self._combine_files(all_val_files, combined_val)
        self._combine_files(all_test_files, combined_test)
        
        print(f"  Combined files created")
        
        # Save statistics
        self._save_statistics()
        
        print(f"\nPreprocessing complete! Output: {self.output_dir}")
    
    def _combine_files(self, file_list: List[Path], output_file: Path):
        """Combine multiple JSONL files."""
        with open(output_file, 'w') as outfile:
            for file_path in file_list:
                with open(file_path, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
    
    def _save_statistics(self):
        """Save preprocessing statistics."""
        stats_file = self.output_dir / "preprocessing_stats.json"
        
        # Convert tuple keys to strings for JSON serialization
        grid_dist_str = {f"{k[0]}x{k[1]}": v for k, v in self.stats['grid_size_distribution'].items()}
        split_dist_str = {f"{k[0]}x{k[1]}": dict(v) for k, v in self.stats['split_distribution'].items()}
        
        stats_dict = {
            'total_records_loaded': self.stats['total_records'],
            'records_filtered_out': self.stats['filtered_out'],
            'grid_size_distribution': grid_dist_str,
            'zone_pattern_distribution': dict(self.stats['zone_pattern_distribution']),
            'split_distribution': split_dist_str
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess dataset for CNN+RNN training')
    parser.add_argument('--input', type=str, default=None,
                       help='Input dataset file (default: uses config data.dataset_path)')
    parser.add_argument('--config', type=str, default='model/config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    preprocessor = DatasetPreprocessor(config_path=args.config)
    preprocessor.process_dataset(dataset_path=args.input)


if __name__ == "__main__":
    main()
