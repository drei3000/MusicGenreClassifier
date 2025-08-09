# Genre Classifier

Lightweight music genre classifier using MFCC features and a RandomForest baseline.

## Structure

```
src/                # library code and configs
scripts/            # runnable scripts (dataset/train/evaluate)
tests/              # unit tests
data/               # dataset (raw/processed)
models/             # saved models
notebooks/          # analysis notebooks
```

## Quick start

1) Install deps
```
pip install -r requirements.txt
```

2) Create dataset (group-aware split saved inside `music_dataset.npz`)
```
python scripts/create_dataset.py
```

3) Train model
```
python scripts/train.py
```

4) Evaluate (file-level accuracy + confusion matrix)
```
python scripts/evaluate.py
```

## Notes
- Splits are group-aware by file to prevent leakage.
- `src/config.py` stores default paths and parameters.

