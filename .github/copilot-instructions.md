# AI Assistant Instructions for ML-Sec Codebase

This repository contains code examples from "Machine Learning and Security" by Clarence Chio and David Freeman (O'Reilly 2018). It demonstrates practical ML applications in cybersecurity across 8 chapters.

## Project Structure

- **Chapter-based organization**: Each `chapter{1-8}/` directory contains related notebooks, utilities, and datasets
- **Jupyter-first approach**: Primary development happens in `.ipynb` notebooks with supporting Python utilities
- **Self-contained examples**: Each chapter can be worked on independently with minimal cross-dependencies
- **Dataset dependencies**: Many examples require external datasets (TREC spam corpus, NSL-KDD, etc.)

## Key Patterns

### Data Loading Conventions
- Datasets typically stored in `chapter{N}/datasets/` subdirectories
- Common pattern: `DATA_DIR` and `LABELS_FILE` variables for paths
- Cross-chapter references use relative paths (e.g., `../chapter5/datasets/nsl-kdd`)

### Utility Modules
- `email_read_util.py`: Standard email parsing with NLTK preprocessing (tokenization, stemming, stopword removal)
- Anomaly detection modules (`ids_heuristics_a.py`, `mad.py`): Time-window based pattern detection
- Chapter-specific `__init__.py` files for module imports

### ML Pipeline Patterns
```python
# Standard imports across notebooks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Common matplotlib setup
%matplotlib inline
```

### Security-Specific Approaches
- **Feature engineering**: Heavy use of TF-IDF for text analysis (emails, web queries)
- **Anomaly detection**: Time-series windowing, threshold-based detection
- **Adversarial examples**: Binary classifier evasion in `chapter8/`
- **Explainability**: LIME integration for model interpretability

## Development Workflow

### Environment Setup
```bash
pip install -r requirements.txt
# Note: Uses older package versions (TensorFlow 1.4.0, scikit-learn 0.19.1)
```

### Dataset Acquisition
- TREC 2007 Spam Corpus: Download from UWaterloo (requires agreement)
- NSL-KDD: Network intrusion detection dataset in `chapter5/datasets/`
- Some datasets included, others require external download

### Notebook Execution
- Notebooks designed for educational walkthrough, not production pipelines
- Cell-by-cell execution with markdown explanations
- Common pattern: data loading → preprocessing → model training → evaluation
- Figures saved to `chapter{N}/figures/` directories

## Code Conventions

### File Naming
- Notebooks: `{topic}-{technique}.ipynb` (e.g., `spam-fighting-naivebayes.ipynb`)
- Utilities: Descriptive names with underscores (`email_read_util.py`)
- Datasets: Domain-specific subdirectories

### Error Handling
- Notebooks use `warnings.filterwarnings('ignore')` for cleaner output
- File operations often use `errors='ignore'` for robustness
- Minimal exception handling - educational focus over production readiness

### Security Analysis Patterns
- **Malware analysis**: Static analysis examples in `chapter4/code-exec-eg/`
- **Network analysis**: PCAP files and feature extraction in `chapter5/`
- **Web security**: WAF training with good/bad query classification in `chapter8/waf/`

## When Working on This Codebase

1. **Check dataset availability** before running notebooks - many require external downloads
2. **Use relative paths** when referencing cross-chapter resources
3. **Follow the book structure** - code is designed to accompany specific chapters
4. **Consider package versions** - this uses older ML stack (pre-TensorFlow 2.0)
5. **Focus on educational value** over production optimization

## Key Integration Points

- Cross-chapter dataset sharing (especially NSL-KDD in chapters 2 & 5)
- Utility modules imported across multiple notebooks within chapters
- Figure generation for book illustrations (saved to `figures/` dirs)
- Model persistence patterns (pickle files, trained models)