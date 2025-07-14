#!/usr/bin/env python3
"""
Dataset Analysis Script for GDPR Compliance Detection Benchmark

This script analyzes the original GDPR dataset, Task 1 dataset, and Task 2 dataset
to provide comprehensive statistics about the data distribution, violation patterns,
and dataset characteristics.
"""

import json
import os
from collections import defaultdict, Counter
from datetime import datetime

def load_json_file(filepath):
    """Load JSON file and return data"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {filepath}")
        return []

def analyze_original_dataset(data):
    """Analyze the original GDPR dataset"""
    stats = {
        'total_samples': len(data),
        'unique_apps': set(),
        'unique_repos': set(),
        'unique_commits': set(),
        'article_distribution': Counter(),
        'app_distribution': Counter(),
        'repo_distribution': Counter(),
        'commit_distribution': Counter(),
        'file_extension_distribution': Counter(),
        'code_snippet_lengths': [],
        'annotation_lengths': []
    }
    
    for item in data:
        # Basic counts
        stats['unique_apps'].add(item.get('app_name', 'Unknown'))
        stats['unique_repos'].add(item.get('repo_url', 'Unknown'))
        stats['unique_commits'].add(item.get('Commit_ID', 'Unknown'))
        
        # Distributions
        stats['article_distribution'][item.get('violated_article', 0)] += 1
        stats['app_distribution'][item.get('app_name', 'Unknown')] += 1
        stats['repo_distribution'][item.get('repo_url', 'Unknown')] += 1
        stats['commit_distribution'][item.get('Commit_ID', 'Unknown')] += 1
        
        # File analysis
        code_path = item.get('code_snippet_path', '')
        if ':' in code_path:
            file_path = code_path.split(':')[0]
            if '.' in file_path:
                ext = file_path.split('.')[-1]
                stats['file_extension_distribution'][ext] += 1
        
        # Length analysis
        code_snippet = item.get('code_snippet', '')
        annotation = item.get('annotation_note', '')
        stats['code_snippet_lengths'].append(len(code_snippet))
        stats['annotation_lengths'].append(len(annotation))
    
    return stats

def analyze_task1_dataset(data):
    """Analyze Task 1 dataset (multi-granularity violation detection)"""
    stats = {
        'total_samples': len(data),
        'unique_apps': set(),
        'unique_repos': set(),
        'unique_commits': set(),
        'file_level_violations': 0,
        'module_level_violations': 0,
        'line_level_violations': 0,
        'file_level_articles': Counter(),
        'module_level_articles': Counter(),
        'line_level_articles': Counter(),
        'file_paths': set(),
        'module_names': set(),
        'violation_descriptions': [],
        'line_span_patterns': Counter()
    }
    
    for item in data:
        # Basic counts
        stats['unique_apps'].add(item.get('app_name', 'Unknown'))
        stats['unique_repos'].add(item.get('repo_url', 'Unknown'))
        stats['unique_commits'].add(item.get('Commit_ID', 'Unknown'))
        
        # File level violations
        for violation in item.get('file_level_violations', []):
            stats['file_level_violations'] += 1
            stats['file_paths'].add(violation.get('file_path', ''))
            for article in violation.get('violated_articles', []):
                stats['file_level_articles'][article] += 1
        
        # Module level violations
        for violation in item.get('module_level_violations', []):
            stats['module_level_violations'] += 1
            stats['module_names'].add(violation.get('module_name', ''))
            for article in violation.get('violated_articles', []):
                stats['module_level_articles'][article] += 1
        
        # Line level violations
        for violation in item.get('line_level_violations', []):
            stats['line_level_violations'] += 1
            for article in violation.get('violated_articles', []):
                stats['line_level_articles'][article] += 1
            
            # Line span analysis
            line_span = violation.get('line_spans', '')
            stats['line_span_patterns'][line_span] += 1
            
            # Violation description analysis
            desc = violation.get('violation_description', '')
            stats['violation_descriptions'].append(len(desc))
    
    return stats

def analyze_task2_dataset(data):
    """Analyze Task 2 dataset (snippet-level multi-label classification)"""
    stats = {
        'total_samples': len(data),
        'unique_apps': set(),
        'unique_repos': set(),
        'unique_commits': set(),
        'article_distribution': Counter(),
        'app_distribution': Counter(),
        'repo_distribution': Counter(),
        'commit_distribution': Counter(),
        'file_extension_distribution': Counter(),
        'code_snippet_lengths': [],
        'violations_per_sample': Counter(),
        'file_paths': set(),
        'line_patterns': Counter()
    }
    
    for item in data:
        # Basic counts
        stats['unique_apps'].add(item.get('app_name', 'Unknown'))
        stats['unique_repos'].add(item.get('repo_url', 'Unknown'))
        stats['unique_commits'].add(item.get('Commit_ID', 'Unknown'))
        
        # Distributions
        violated_articles = item.get('violated_articles', [])
        stats['violations_per_sample'][len(violated_articles)] += 1
        
        for article in violated_articles:
            stats['article_distribution'][article] += 1
        
        stats['app_distribution'][item.get('app_name', 'Unknown')] += 1
        stats['repo_distribution'][item.get('repo_url', 'Unknown')] += 1
        stats['commit_distribution'][item.get('Commit_ID', 'Unknown')] += 1
        
        # File analysis
        code_path = item.get('code_snippet_path', '')
        if ':' in code_path:
            file_path = code_path.split(':')[0]
            stats['file_paths'].add(file_path)
            if '.' in file_path:
                ext = file_path.split('.')[-1]
                stats['file_extension_distribution'][ext] += 1
            
            # Line pattern analysis
            line_info = code_path.split(':')[-1].strip()
            stats['line_patterns'][line_info] += 1
        
        # Length analysis
        code_snippet = item.get('code_snippet', '')
        stats['code_snippet_lengths'].append(len(code_snippet))
    
    return stats

def calculate_statistics(data_list, name):
    """Calculate basic statistics for a list of numbers"""
    if not data_list:
        return f"{name}: No data available"
    
    stats = {
        'count': len(data_list),
        'min': min(data_list),
        'max': max(data_list),
        'mean': sum(data_list) / len(data_list),
        'median': sorted(data_list)[len(data_list) // 2]
    }
    
    return f"{name}: count={stats['count']}, min={stats['min']:.1f}, max={stats['max']:.1f}, mean={stats['mean']:.1f}, median={stats['median']:.1f}"

def write_analysis_report(original_stats, task1_stats, task2_stats, output_file):
    """Write comprehensive analysis report to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GDPR COMPLIANCE DETECTION DATASET ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Original Dataset Analysis
        f.write("1. ORIGINAL GDPR DATASET ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total samples: {original_stats['total_samples']}\n")
        f.write(f"Unique applications: {len(original_stats['unique_apps'])}\n")
        f.write(f"Unique repositories: {len(original_stats['unique_repos'])}\n")
        f.write(f"Unique commits: {len(original_stats['unique_commits'])}\n\n")
        
        f.write("Top 10 most frequent applications:\n")
        for app, count in original_stats['app_distribution'].most_common(10):
            f.write(f"  {app}: {count} violations\n")
        f.write("\n")
        
        f.write("GDPR Article distribution:\n")
        for article, count in sorted(original_stats['article_distribution'].items()):
            f.write(f"  Article {article}: {count} violations\n")
        f.write("\n")
        
        f.write("File extension distribution:\n")
        for ext, count in original_stats['file_extension_distribution'].most_common(10):
            f.write(f"  .{ext}: {count} files\n")
        f.write("\n")
        
        f.write("Code snippet length statistics:\n")
        f.write(f"  {calculate_statistics(original_stats['code_snippet_lengths'], 'Length')}\n")
        f.write("\n")
        
        f.write("Annotation length statistics:\n")
        f.write(f"  {calculate_statistics(original_stats['annotation_lengths'], 'Length')}\n")
        f.write("\n")
        
        # Task 1 Dataset Analysis
        f.write("2. TASK 1 DATASET ANALYSIS (Multi-granularity Violation Detection)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples: {task1_stats['total_samples']}\n")
        f.write(f"Unique applications: {len(task1_stats['unique_apps'])}\n")
        f.write(f"Unique repositories: {len(task1_stats['unique_repos'])}\n")
        f.write(f"Unique commits: {len(task1_stats['unique_commits'])}\n\n")
        
        f.write("Violation counts by granularity:\n")
        f.write(f"  File-level violations: {task1_stats['file_level_violations']}\n")
        f.write(f"  Module-level violations: {task1_stats['module_level_violations']}\n")
        f.write(f"  Line-level violations: {task1_stats['line_level_violations']}\n\n")
        
        f.write("File-level GDPR Article distribution:\n")
        for article, count in sorted(task1_stats['file_level_articles'].items()):
            f.write(f"  Article {article}: {count} violations\n")
        f.write("\n")
        
        f.write("Module-level GDPR Article distribution:\n")
        for article, count in sorted(task1_stats['module_level_articles'].items()):
            f.write(f"  Article {article}: {count} violations\n")
        f.write("\n")
        
        f.write("Line-level GDPR Article distribution:\n")
        for article, count in sorted(task1_stats['line_level_articles'].items()):
            f.write(f"  Article {article}: {count} violations\n")
        f.write("\n")
        
        f.write(f"Unique file paths: {len(task1_stats['file_paths'])}\n")
        f.write(f"Unique module names: {len(task1_stats['module_names'])}\n\n")
        
        f.write("Line span patterns (top 10):\n")
        for pattern, count in task1_stats['line_span_patterns'].most_common(10):
            f.write(f"  {pattern}: {count} occurrences\n")
        f.write("\n")
        
        f.write("Violation description length statistics:\n")
        f.write(f"  {calculate_statistics(task1_stats['violation_descriptions'], 'Length')}\n")
        f.write("\n")
        
        # Task 2 Dataset Analysis
        f.write("3. TASK 2 DATASET ANALYSIS (Snippet-level Multi-label Classification)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples: {task2_stats['total_samples']}\n")
        f.write(f"Unique applications: {len(task2_stats['unique_apps'])}\n")
        f.write(f"Unique repositories: {len(task2_stats['unique_repos'])}\n")
        f.write(f"Unique commits: {len(task2_stats['unique_commits'])}\n\n")
        
        f.write("Top 10 most frequent applications:\n")
        for app, count in task2_stats['app_distribution'].most_common(10):
            f.write(f"  {app}: {count} violations\n")
        f.write("\n")
        
        f.write("GDPR Article distribution:\n")
        for article, count in sorted(task2_stats['article_distribution'].items()):
            f.write(f"  Article {article}: {count} violations\n")
        f.write("\n")
        
        f.write("Violations per sample distribution:\n")
        for count, freq in sorted(task2_stats['violations_per_sample'].items()):
            f.write(f"  {count} violation(s): {freq} samples\n")
        f.write("\n")
        
        f.write("File extension distribution:\n")
        for ext, count in task2_stats['file_extension_distribution'].most_common(10):
            f.write(f"  .{ext}: {count} files\n")
        f.write("\n")
        
        f.write("Code snippet length statistics:\n")
        f.write(f"  {calculate_statistics(task2_stats['code_snippet_lengths'], 'Length')}\n")
        f.write("\n")
        
        f.write(f"Unique file paths: {len(task2_stats['file_paths'])}\n\n")
        
        f.write("Line pattern distribution (top 10):\n")
        for pattern, count in task2_stats['line_patterns'].most_common(10):
            f.write(f"  {pattern}: {count} occurrences\n")
        f.write("\n")
        
        # Cross-dataset Comparison
        f.write("4. CROSS-DATASET COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(f"Original dataset size: {original_stats['total_samples']}\n")
        f.write(f"Task 1 dataset size: {task1_stats['total_samples']}\n")
        f.write(f"Task 2 dataset size: {task2_stats['total_samples']}\n\n")
        
        f.write("Common applications across datasets:\n")
        original_apps = original_stats['unique_apps']
        task1_apps = task1_stats['unique_apps']
        task2_apps = task2_stats['unique_apps']
        common_apps = original_apps & task1_apps & task2_apps
        f.write(f"  Applications in all datasets: {len(common_apps)}\n")
        for app in sorted(common_apps):
            f.write(f"    {app}\n")
        f.write("\n")
        
        f.write("Common repositories across datasets:\n")
        original_repos = original_stats['unique_repos']
        task1_repos = task1_stats['unique_repos']
        task2_repos = task2_stats['unique_repos']
        common_repos = original_repos & task1_repos & task2_repos
        f.write(f"  Repositories in all datasets: {len(common_repos)}\n")
        for repo in sorted(common_repos):
            f.write(f"    {repo}\n")
        f.write("\n")
        
        # Article coverage analysis
        f.write("GDPR Article coverage analysis:\n")
        original_articles = set(original_stats['article_distribution'].keys())
        task1_articles = set(task1_stats['file_level_articles'].keys()) | set(task1_stats['module_level_articles'].keys()) | set(task1_stats['line_level_articles'].keys())
        task2_articles = set(task2_stats['article_distribution'].keys())
        
        f.write(f"  Articles in original dataset: {sorted(original_articles)}\n")
        f.write(f"  Articles in Task 1 dataset: {sorted(task1_articles)}\n")
        f.write(f"  Articles in Task 2 dataset: {sorted(task2_articles)}\n")
        f.write(f"  Articles in all datasets: {sorted(original_articles & task1_articles & task2_articles)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("=" * 80 + "\n")

def main():
    """Main function to run the dataset analysis"""
    print("Starting GDPR dataset analysis...")
    
    # Load datasets
    original_data = load_json_file("GDPR_dataset.json")
    task1_data = load_json_file("task1_dataset.json")
    task2_data = load_json_file("task2_dataset.json")
    
    print(f"Loaded {len(original_data)} samples from original dataset")
    print(f"Loaded {len(task1_data)} samples from Task 1 dataset")
    print(f"Loaded {len(task2_data)} samples from Task 2 dataset")
    
    # Analyze datasets
    print("Analyzing original dataset...")
    original_stats = analyze_original_dataset(original_data)
    
    print("Analyzing Task 1 dataset...")
    task1_stats = analyze_task1_dataset(task1_data)
    
    print("Analyzing Task 2 dataset...")
    task2_stats = analyze_task2_dataset(task2_data)
    
    # Write analysis report
    output_file = "dataset_analysis_report.txt"
    print(f"Writing analysis report to {output_file}...")
    write_analysis_report(original_stats, task1_stats, task2_stats, output_file)
    
    print(f"Analysis complete! Report saved to {output_file}")

if __name__ == "__main__":
    main() 