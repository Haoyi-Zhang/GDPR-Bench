# GDPR Compliance Detection Benchmark

GDPR-Bench is the first benchmark for evaluating large language models (LLMs) on GDPR compliance detection in software. It comprises 545 annotated violation instances from seven open-source repositories, covering all major GDPR articles at file-, module-, and line-level granularity. The benchmark defines two tasks: (1) multi-granularity violation localization, where models identify violated articles at different code scopes using Accuracy@k metrics; and (2) snippet-level multi-label classification, where models assign all applicable GDPR articles to code fragments, evaluated by exact-match accuracy, macro-precision, macro-recall, and macro-F1. We benchmark eight state-of-the-art LLMs—including Deepseek-r1, Qwen2.5-72B, Gemini-2.5-pro (preview and experimental), Claude-3.5-Sonnet, Claude-3.7-Sonnet, GPT-4o, and O1—and find that while LLMs can pinpoint individual violations in narrow contexts, they struggle with holistic reasoning and exhaustive multi-label enumeration. GDPR-Bench and all evaluation artifacts are publicly released to foster further research in AI-driven privacy compliance.

This project provides a comprehensive benchmark for detecting GDPR (General Data Protection Regulation) violations in Android applications using large language models. The project includes dataset generation, multi-task evaluation, and comprehensive analysis tools.

## Project Overview

GDPR-Bench evaluates the ability of large language models to detect GDPR violations in code, focusing on two main tasks:

1. **Task 1: Multi-granularity Violation Localization** - Models identify violated GDPR articles at file, module, and line levels. Performance is measured using Accuracy@k metrics.
2. **Task 2: Snippet-level Multi-label Classification** - Models assign all relevant GDPR articles to individual code fragments, evaluated by exact-match accuracy, macro-precision, macro-recall, and macro-F1.

The benchmark dataset consists of 545 annotated violation instances from seven open-source repositories, covering all major GDPR articles. We evaluate eight state-of-the-art LLMs and report their performance on both tasks, highlighting strengths and current limitations in automated privacy compliance detection.

## Project Structure

```
├── create_task1_dataset.py       # Task 1 dataset creation
├── create_task2_dataset.py       # Task 2 dataset creation
├── predict.py                    # Main prediction script
├── evaluate_model.py             # Evaluation script
├── dataset_analysis.py           # Dataset analysis tool
├── requirements.txt              # Python dependencies
├── GDPR_dataset.json             # Original dataset
├── task1_dataset.json            # Task 1 dataset
├── task2_dataset.json            # Task 2 dataset
├── task1_predictions/            # Task 1 prediction results
├── task2_predictions/            # Task 2 prediction results
├── task1_eval_results/           # Task 1 evaluation results
├── task2_eval_results/           # Task 2 evaluation results
├── logs/                         # Log files
└── repos/                        # Source repositories
```


### Task 1 Evaluation Results

| Model | File-Level Accuracy@1 | @2 | @3 | @4 | @5 |
|-------|----------------------|----|----|----|----|
| claude-3-5-sonnet-20241022 | 0.1257 | 0.2974 | 0.3940 | 0.4760 | 0.5623 |
| claude-3-7-sonnet-20250219 | 0.1439 | 0.3050 | 0.3849 | 0.4684 | 0.5433 |
| deepseek-r1 | 0.1462 | 0.3311 | 0.4031 | 0.4837 | 0.5732 |
| gemini-2.5-pro-exp-03-25 | 0.1289 | 0.2957 | 0.4072 | 0.4612 | 0.5612 |
| gemini-2.5-pro-preview-05-06 | 0.1315 | 0.3014 | 0.3981 | 0.4811 | 0.5903 |
| gpt-4o | 0.0905 | 0.2018 | 0.2569 | 0.3156 | 0.3548 |
| o1 | 0.0253 | 0.0560 | 0.0600 | 0.0693 | 0.0810 |
| qwen2.5-72b-instruct | 0.0712 | 0.1578 | 0.2037 | 0.2410 | 0.2481 |

| Model | Module-Level Accuracy@1 | @2 | @3 | @4 | @5 |
|-------|------------------------|----|----|----|----|
| claude-3-5-sonnet-20241022 | 0.1254 | 0.2920 | 0.3710 | 0.4553 | 0.5351 |
| claude-3-7-sonnet-20250219 | 0.1519 | 0.3233 | 0.3913 | 0.4992 | 0.5875 |
| deepseek-r1 | 0.1323 | 0.3193 | 0.3843 | 0.4533 | 0.5389 |
| gemini-2.5-pro-exp-03-25 | 0.1349 | 0.3213 | 0.4261 | 0.4799 | 0.5905 |
| gemini-2.5-pro-preview-05-06 | 0.1315 | 0.2989 | 0.3947 | 0.4899 | 0.5858 |
| gpt-4o | 0.1007 | 0.2212 | 0.2765 | 0.3261 | 0.3785 |
| o1 | 0.0098 | 0.0339 | 0.0449 | 0.0584 | 0.0758 |
| qwen2.5-72b-instruct | 0.0737 | 0.1763 | 0.2110 | 0.2536 | 0.2696 |

| Model | Line-Level Accuracy@1 | @2 | @3 | @4 | @5 |
|-------|---------------------|----|----|----|----|
| claude-3-5-sonnet-20241022 | 0.1840 | 0.4768 | 0.6428 | 0.7069 | 0.7418 |
| claude-3-7-sonnet-20250219 | 0.3732 | 0.6026 | 0.7155 | 0.7829 | 0.8039 |
| deepseek-r1 | 0.3734 | 0.6817 | 0.7447 | 0.7596 | 0.7602 |
| gemini-2.5-pro-exp-03-25 | 0.1757 | 0.4353 | 0.6496 | 0.7411 | 0.8381 |
| gemini-2.5-pro-preview-05-06 | 0.1767 | 0.4438 | 0.6515 | 0.7555 | 0.8390 |
| gpt-4o | 0.6523 | 0.7250 | 0.7332 | 0.7417 | 0.7417 |
| o1 | 0.3760 | 0.4709 | 0.4800 | 0.4813 | 0.4813 |
| qwen2.5-72b-instruct | 0.6854 | 0.7687 | 0.7901 | 0.8070 | 0.8106 |

## Task 2: Snippet-level Multi-label Classification

### Dataset Statistics

- **Total samples**: 258
- **Unique applications**: 7
- **Unique repositories**: 7
- **Unique commits**: 36

#### Top 10 Most Frequent Applications
- Dash: 61 violations
- Android_Spy_App: 53 violations
- rdroid: 49 violations
- pounce-keys: 43 violations
- Rafel_Rat: 28 violations
- PrivacyBreacher: 17 violations
- L3MON: 7 violations

#### GDPR Article Distribution
- Article 5: 86 violations
- Article 6: 126 violations
- Article 7: 27 violations
- Article 8: 3 violations
- Article 9: 28 violations
- Article 12: 39 violations
- Article 13: 58 violations
- Article 14: 15 violations
- Article 15: 13 violations
- Article 16: 1 violation
- Article 17: 6 violations
- Article 18: 1 violation
- Article 19: 1 violation
- Article 21: 1 violation
- Article 25: 56 violations
- Article 30: 5 violations
- Article 32: 61 violations
- Article 33: 4 violations
- Article 35: 4 violations
- Article 44: 3 violations
- Article 46: 2 violations
- Article 58: 1 violation
- Article 83: 1 violation

#### Violations per sample distribution
- 1 violation: 138 samples
- 2 violations: 39 samples
- 3 violations: 44 samples
- 4 violations: 19 samples
- 5 violations: 11 samples
- 6 violations: 2 samples
- 7 violations: 1 sample
- 8 violations: 1 sample
- 9 violations: 2 samples
- 18 violations: 1 sample

#### File Extension Distribution
- .kt: 104 files
- .php: 59 files
- .java: 50 files
- .xml: 24 files
- .js: 10 files
- .py: 6 files
- .html: 4 files
- .h: 1 file

#### Code Snippet Length Statistics
- Length: count=258, min=12, max=454, mean=139.1, median=106

### Task 2 Evaluation Results

| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|-------|----------|-----------------|--------------|----------|
| claude-3-5-sonnet-20241022 | 0.1048 | 0.0432 | 0.0787 | 0.0523 |
| claude-3-7-sonnet-20250219 | 0.0862 | 0.0352 | 0.0439 | 0.0374 |
| deepseek-r1 | 0.1191 | 0.0454 | 0.0588 | 0.0496 |
| gemini-2.5-pro-exp-03-25 | 0.0889 | 0.0295 | 0.0588 | 0.0380 |
| gemini-2.5-pro-preview-05-06 | 0.0985 | 0.0324 | 0.0649 | 0.0418 |
| gpt-4o | 0.1098 | 0.0636 | 0.0427 | 0.0480 |
| o1 | 0.0210 | 0.0571 | 0.0081 | 0.0137 |
| qwen2.5-72b-instruct | 0.0501 | 0.0426 | 0.0180 | 0.0236 |


### Key Performance Insights

**Task 1 Performance Analysis:**
- **Line-level detection** shows the best performance, with qwen2.5-72b-instruct achieving the highest Accuracy@1 (68.64%)
- **File and module-level detection** are more challenging, with best Accuracy@1 around 14-15%
- **Claude-3-7-sonnet** shows strong performance across all granularities
- **GPT-4o** excels in line-level detection but struggles with file/module levels

**Task 2 Performance Analysis:**
- **Overall accuracy is low** (best model ~12%) due to the complexity of multi-label classification
- **Deepseek-r1** achieves the highest accuracy (12.00%) and best overall performance
- **Article 5 and 6** show better detection rates across most models
- **Many articles have zero precision/recall**, indicating significant detection challenges for rare violations

**Model Comparison Summary:**
- **qwen2.5-72b-instruct**: Best line-level performance in Task 1
- **deepseek-r1**: Best overall performance in Task 2
- **Claude models**: Consistent performance across tasks
- **GPT-4o**: Strong line-level detection but variable performance
- **o1**: Generally lower performance across both tasks

## How to Run the Project

### Prerequisites

1. **Python Environment**: Python 3.8 or higher
2. **Dependencies**: Install required packages using the provided requirements file
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install core dependencies manually:
   ```bash
   pip install requests numpy pandas matplotlib seaborn scikit-learn
   ```
3. **API Keys**: Configure API keys for the language models you want to use

### Step 1: Dataset Generation (Optional)

If you want to regenerate the dataset from scratch:

```bash
python generate_gdpr_dataset.py
```

This will:
- Clone repositories from the `repos/` directory
- Analyze code files for GDPR violations
- Generate the original `GDPR_dataset.json`

### Step 2: Create Task Datasets

```bash
# Create Task 1 dataset
python create_task1_dataset.py

# Create Task 2 dataset
python create_task2_dataset.py
```

### Step 3: Run Predictions

```bash
# Run predictions for all models on all tasks
python predict.py

# Or run specific tasks/models
python predict.py --task task1 --model gpt-4o
python predict.py --task task2 --model claude-3-5-sonnet
```

### Step 4: Evaluate Results

```bash
# Run evaluation for all models
python evaluate_model.py

# Or evaluate specific models
python evaluate_model.py --model gpt-4o --task task1
python evaluate_model.py --model claude-3-5-sonnet --task task2
```

### Step 5: Analyze Dataset (Optional)

```bash
# Generate comprehensive dataset analysis
python dataset_analysis.py
```

This will create `dataset_analysis_report.txt` with detailed statistics.

## Key Findings

### Performance Analysis

1. **Task 1 Performance**:
   - Line-level detection shows the best performance (up to 68.64% Accuracy@1 for some models)
   - File and module-level detection are more challenging (best Accuracy@1 around 14-15%)
   - All models show similar trends, but there is significant variance in line-level performance

2. **Task 2 Performance**:
   - Overall accuracy is low (best model ~12%) due to the complexity of multi-label classification
   - Some articles (5, 6) show better detection rates
   - Many articles have zero precision/recall, indicating detection challenges

3. **Model Comparison**:
   - Deepseek-r1 and qwen2.5-72b-instruct show the highest line-level accuracy in Task 1
   - deepseek-r1 achieves the highest overall accuracy in Task 2
   - GPT-4o, Claude, Gemini, and other models show competitive but generally lower performance

### Dataset Insights

1. **Coverage**: The dataset covers 23 GDPR articles with good distribution
2. **Applications**: 7 Android applications with varying violation patterns
3. **File Types**: Kotlin files dominate, followed by PHP and Java
4. **Multi-label Nature**: Task 2 shows a significant portion of samples have multiple violations

## Contributing

To contribute to this benchmark:

1. Fork the repository
2. Add new models or evaluation metrics
3. Improve the dataset or add new applications
4. Submit a pull request with detailed documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository. 