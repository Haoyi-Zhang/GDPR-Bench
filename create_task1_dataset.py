#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict, Counter

def main(input_file='GDPR_dataset.json', output_file='task1_dataset.json'):
    # Read original data - handling multiple JSON objects
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {input_file}")
    
    # Group violations by repository
    repos = {}
    for entry in data:
        key = (entry.get('repo_url',''), entry.get('app_name',''), entry.get('Commit_ID',''))
        if key not in repos:
            repos[key] = []
        repos[key].append(entry)
    
    # Process data for Task 1 format
    task1_data = []
    for (repo_url, app_name, commit_id), violations in repos.items():
        # Group by file for file-level violations
        file_violations = defaultdict(set)
        module_violations = defaultdict(set)
        line_violations = []
        
        for violation in violations:
            file_path = violation.get('code_snippet_path','').split(':')[0].strip()
            article = violation.get('violated_article','')
            
            # Add to file-level violations
            file_violations[file_path].add(article)
            
            # Extract module name (simplified approach - could be improved)
            code_path_parts = file_path.split('/')
            if len(code_path_parts) > 0:
                file_name = code_path_parts[-1]
                class_name = file_name.split('.')[0]
                # Try to find a more specific module name from the code path or annotation
                module_name = class_name
                if 'annotation_note' in violation and '.' in violation['annotation_note']:
                    method_hints = [word for word in violation['annotation_note'].split() 
                                   if "." in word and class_name.lower() in word.lower()]
                    if method_hints:
                        module_name = method_hints[0].strip('.,():;')
                
                module_key = f"{file_path}:{module_name}"
                module_violations[module_key].add(article)
            
            # Extract line spans (handle both 'line' and 'lines')
            code_path = violation.get('code_snippet_path','')
            if "line" in code_path:
                # Handle both 'line 45' and 'lines 67-71'
                if "lines" in code_path:
                    line_info = code_path.split('lines')[-1].strip()
                else:
                    line_info = code_path.split('line')[-1].strip()
                line_spans = line_info.replace('–', '-').strip(':')
                
                line_violations.append({
                    "file_path": file_path,
                    "line_spans": line_spans,
                    "violated_articles": [article],
                    "violation_description": violation.get('annotation_note','')
                })
        
        # Create the repository entry
        repo_entry = {
            "repo_url": repo_url,
            "app_name": app_name,
            "Commit_ID": commit_id,
            "file_level_violations": [
                {"file_path": file_path, "violated_articles": sorted(list(articles))}
                for file_path, articles in file_violations.items()
            ],
            "module_level_violations": [
                {
                    "file_path": module_key.split(':')[0],
                    "module_name": module_key.split(':')[1],
                    "violated_articles": sorted(list(articles))
                }
                for module_key, articles in module_violations.items()
            ],
            "line_level_violations": []
        }
        
        # Merge line violations with the same file path and line spans
        line_map = {}
        for violation in line_violations:
            key = (violation['file_path'], violation['line_spans'])
            if key not in line_map:
                line_map[key] = violation
            else:
                # Merge violated articles
                line_map[key]['violated_articles'] = sorted(list(set(
                    line_map[key]['violated_articles'] + violation['violated_articles']
                )))
        
        repo_entry['line_level_violations'] = list(line_map.values())
        task1_data.append(repo_entry)
    
    # Write to task1_dataset.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(task1_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {output_file} with {len(task1_data)} repositories")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'GDPR_dataset.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'task1_dataset.json'
    main(input_file, output_file) 