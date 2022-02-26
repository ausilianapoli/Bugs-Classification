# BUGS CLASSIFICATION

## Requirements

Install it as `pip install -r requirements.txt`

## Usage

Run `BugsClassification.py` with two parameters:  
- `features_extractor` -> `tfidf` or `doc2vec`
- `classifier` -> `logisticRegression` or `kmeans` or `knn` or `mlp`  
For example `python BugsClassification.py --features_extractor tfidf --classifier logisticRegression`
