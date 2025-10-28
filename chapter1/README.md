# Chapter 1: Spam Fighting

This chapter demonstrates **three different techniques for email spam detection** using the TREC 2007 Public Spam Corpus.

## Dataset Setup

1. Read the "Agreement for use" at https://plg.uwaterloo.ca/~gvcormac/treccorpus07/
2. Download the 255 MB corpus (trec07p.tgz)
3. Extract to `chapter1/datasets/trec07p/`

All three approaches use the same dataset but employ different classification strategies.

## Core Utility: `email_read_util.py`

Provides email preprocessing functions:

- **`extract_email_text(path)`**: Parses email files, extracts subject + body text
- **`load(path)`**: Full NLP preprocessing pipeline:
  - Tokenizes email text using NLTK
  - Removes punctuation
  - Filters out stopwords (common words like "the", "a", "is")
  - Applies Porter stemming (reduces words to root form: "running" → "run")

## Three Approaches

### 1. Naive Bayes Classifier (`spam-fighting-naivebayes.ipynb`)

**Traditional machine learning approach** using scikit-learn:

1. Loads 75,000+ emails with spam/ham labels
2. 70/30 train/test split
3. Feature extraction using `CountVectorizer` (bag-of-words)
4. Trains `MultinomialNB` classifier
5. Evaluates with precision/recall/F1 scores

**Key insight**: The "proper" ML approach with feature vectors and probabilistic classification.

### 2. Blacklist-Based Detection (`spam-fighting-blacklist.ipynb`)

**Simple heuristic approach** based on word filtering:

1. **Build blacklist**: Identifies words appearing in spam but NOT in ham emails
   - Creates sets: `spam_words` and `ham_words`
   - Blacklist = `spam_words - ham_words`
   - Saves to `blacklist.pkl` for reuse
2. **Detection logic**: If email contains ANY blacklist word → spam
3. **Evaluation**: Manually calculates confusion matrix

**Key insight**: Simple rule-based approach. Fast but less sophisticated than ML.

### 3. Locality-Sensitive Hashing (`spam-fighting-lsh.ipynb`)

**Similarity-based detection** using MinHash LSH:

1. Builds MinHash signatures for all training spam emails
2. Uses `MinHashLSH` with Jaccard threshold of 0.5
   - 128 permutation functions for hash signatures
3. Query test email against LSH index
   - If similar to any known spam → spam
   - Otherwise → ham
4. Evaluates with confusion matrix

**Key insight**: Uses approximate nearest neighbor search to detect spam similar to previously seen spam. Good for detecting spam campaigns with slight variations.

## Common Patterns

- **Same dataset**: TREC 2007 corpus
- **70/30 train/test split**: Consistent evaluation
- **Label encoding**: `1 = ham`, `0 = spam`
- **Preprocessing**: All use stemmed tokens from `email_read_util.load()`

## Pedagogical Purpose

Demonstrates the **evolution from simple to sophisticated spam detection**:

1. **Blacklist** = simplest (rule-based)
2. **LSH** = moderate (similarity-based)
3. **Naive Bayes** = most sophisticated (statistical ML)

This teaches that security problems can be approached with varying levels of complexity, and sometimes simpler methods are sufficient depending on requirements.
