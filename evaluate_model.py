#!/usr/bin/env python3
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_task1(pred_file, gold_file, out_txt):
    """Evaluate Task 1: Output Accuracy@1, @2, @3, @4, @5 for each granularity."""
    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)
    levels = ['file_level_violations', 'module_level_violations', 'line_level_violations']
    level_keys = {
        'file_level_violations': lambda x: (x['repo_url'], x['Commit_ID'], x['file_path']),
        'module_level_violations': lambda x: (x['repo_url'], x['Commit_ID'], x['file_path'], x['module_name']),
        'line_level_violations': lambda x: (x['repo_url'], x['Commit_ID'], x['file_path'], x['line_spans'])
    }
    
    with open(out_txt, 'w', encoding='utf-8') as fout:
        for level in levels:
            gold_map = {}
            for item in gold_data:
                for v in item.get(level, []):
                    key = level_keys[level]({**item, **v})
                    gold_map[key] = list(v.get('violated_articles', []))
            
            pred_map = {}
            for item in pred_data:
                for v in item.get(level, []):
                    key = level_keys[level]({**item, **v})
                    pred_map[key] = list(v.get('violated_articles', []))
            
            acc_at = {k: [] for k in [1, 2, 3, 4, 5]}
            for key, gold_articles in gold_map.items():
                pred_articles = pred_map.get(key, [])
                gold_set = set(gold_articles) - set([0])
                pred_list = [a for a in pred_articles if a != 0]
                if not gold_set:
                    continue
                for k in [1, 2, 3, 4, 5]:
                    hits = len(gold_set.intersection(pred_list[:k])) if pred_list else 0
                    acc_at[k].append(hits / len(gold_set))
            
            fout.write(f'==== {level} ====' + '\n')
            for k in [1, 2, 3, 4, 5]:
                mean_acc = np.mean(acc_at[k]) if acc_at[k] else 0
                fout.write(f'Accuracy@{k}: {mean_acc:.4f}\n')
            fout.write('\n')
        
        fout.write('Evaluation finished.\n')
    print(f"Task1 evaluation finished: {out_txt}")

def evaluate_task2(pred_file, gold_file, out_txt, out_png):
    """Evaluate Task 2: Multi-label classification metrics and normalized confusion matrix."""
    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)
    
    gold_map = {}
    for item in gold_data:
        key = (item['repo_url'], item['Commit_ID'], item['code_snippet_path'])
        gold_map[key] = set(item.get('violated_articles', []))
    
    pred_map = {}
    for item in pred_data:
        key = (item['repo_url'], item['Commit_ID'], item['code_snippet_path'])
        pred_map[key] = set(item.get('violated_articles', []))
    
    all_keys = set(gold_map.keys()) | set(pred_map.keys())
    all_arts = set()
    y_true = []
    y_pred = []
    
    for key in all_keys:
        gold = gold_map.get(key, set([0]))
        pred = pred_map.get(key, set([0]))
        gold = set([0]) if gold == set() else gold
        pred = set([0]) if pred == set() else pred
        gold_has = sorted(gold - set([0]))
        pred_has = sorted(pred - set([0]))
        n = max(len(gold_has), len(pred_has))
        gold_pad = gold_has + [0] * (n - len(gold_has)) if n > 0 else [0]
        pred_pad = pred_has + [0] * (n - len(pred_has)) if n > 0 else [0]
        y_true.extend(gold_pad)
        y_pred.extend(pred_pad)
        all_arts |= set(gold_pad) | set(pred_pad)
    
    labels = sorted(list(all_arts))
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels, zero_division=0)
    
    with open(out_txt, 'w', encoding='utf-8') as fout:
        fout.write(f'Accuracy: {acc:.4f}\n')
        fout.write(f'Macro-Precision: {precision:.4f}\n')
        fout.write(f'Macro-Recall: {recall:.4f}\n')
        fout.write(f'Macro-F1: {f1:.4f}\n')
        fout.write('\nPer-class metrics:\n')
        for i, c in enumerate(labels):
            fout.write(f'  Article {c}: Precision={precision_class[i]:.4f}, Recall={recall_class[i]:.4f}, F1={f1_class[i]:.4f}\n')
        fout.write('\nEvaluation finished.\n')
    
    if labels:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Task2 Confusion Matrix (Normalized)')
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Task2 confusion matrix saved: {out_png}")
    
    print(f"Task2 evaluation finished: {out_txt}")

def main():
    print("GDPR Compliance Model Evaluation (Batch)")
    parser = argparse.ArgumentParser(description="Evaluate GDPR model predictions for Task 1 and Task 2.")
    parser.add_argument('--models', type=str, default='all', help='Comma-separated model names to evaluate, or "all" for all models.')
    parser.add_argument('--task', type=str, default='all', choices=['all', '1', '2'], help='Task to evaluate: 1, 2, or all')
    args = parser.parse_args()
    
    task1_pred_dir = "task1_predictions"
    task2_pred_dir = "task2_predictions"
    task1_gold = "task1_dataset.json"
    task2_gold = "task2_dataset.json"
    task1_eval_dir = "task1_eval_results"
    task2_eval_dir = "task2_eval_results"
    
    ensure_dir(task1_eval_dir)
    ensure_dir(task2_eval_dir)

    # Define aggregate result files for each task
    all_task1_txt = os.path.join(task1_eval_dir, "all_models_task1_eval.txt")
    all_task2_txt = os.path.join(task2_eval_dir, "all_models_task2_eval.txt")
    # Clear the aggregate files at the start
    open(all_task1_txt, "w", encoding="utf-8").close()
    open(all_task2_txt, "w", encoding="utf-8").close()
    
    # Get available models from prediction files
    model_set = set()
    if args.task in ['all', '1']:
        for fname in os.listdir(task1_pred_dir):
            if fname.endswith('.json'):
                model_set.add(fname.split('_task1_predictions.json')[0])
    if args.task in ['all', '2']:
        for fname in os.listdir(task2_pred_dir):
            if fname.endswith('.json'):
                model_set.add(fname.split('_task2_predictions.json')[0])
    
    if args.models == 'all':
        selected_models = sorted(model_set)
    else:
        selected_models = [m.strip() for m in args.models.split(',') if m.strip() in model_set]
        if not selected_models:
            print(f"No valid models selected. Available: {sorted(model_set)}")
            return
    
    print(f"Selected models for evaluation: {selected_models}")
    
    # Evaluate Task 1 for all selected models and aggregate results
    if args.task in ['all', '1']:
        for model in selected_models:
            pred_file = os.path.join(task1_pred_dir, f"{model}_task1_predictions.json")
            if not os.path.exists(pred_file):
                print(f"[Task1] Prediction file not found for model: {model}")
                continue
            out_txt = os.path.join(task1_eval_dir, f"{model}_task1_eval.txt")
            evaluate_task1(pred_file, task1_gold, out_txt)
            # Append this model's results to the aggregate file
            with open(out_txt, "r", encoding="utf-8") as fin, open(all_task1_txt, "a", encoding="utf-8") as fout:
                fout.write(f"==== Model: {model} ====" + "\n")
                fout.write(fin.read())
                fout.write("\n\n")
    
    # Evaluate Task 2 for all selected models and aggregate results
    if args.task in ['all', '2']:
        for model in selected_models:
            pred_file = os.path.join(task2_pred_dir, f"{model}_task2_predictions.json")
            if not os.path.exists(pred_file):
                print(f"[Task2] Prediction file not found for model: {model}")
                continue
            out_txt = os.path.join(task2_eval_dir, f"{model}_task2_eval.txt")
            out_png = os.path.join(task2_eval_dir, f"{model}_task2_confusion.png")
            evaluate_task2(pred_file, task2_gold, out_txt, out_png)
            # Append this model's results to the aggregate file
            with open(out_txt, "r", encoding="utf-8") as fin, open(all_task2_txt, "a", encoding="utf-8") as fout:
                fout.write(f"==== Model: {model} ====" + "\n")
                fout.write(fin.read())
                fout.write("\n\n")
    
    print("All evaluations finished.")

if __name__ == "__main__":
    main() 