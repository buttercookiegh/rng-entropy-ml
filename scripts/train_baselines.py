"""Optional baseline experiments. Run after preparation using uv.

Outputs metric tables and is NOT a formal randomness/security evaluation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def classification_metrics(y, pred, average):
    return {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, average=average, zero_division=0),
        'recall': recall_score(y, pred, average=average, zero_division=0),
        'f1': f1_score(y, pred, average=average, zero_division=0),
    }


def run_source_classification():
    frame = pd.read_csv(DATA / 'sequence_features.csv')
    train, test = frame[frame.split == 'train'], frame[frame.split == 'test']
    # These are engineering features only; entropy is held out as a separate diagnostic.
    predictors = [c for c in frame if c not in {'sequence_id', 'source_label', 'split', 'shannon_entropy'}]
    rows = []
    for label, model in [
        ('Logistic Regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=400))),
        ('PCA(5) + Logistic Regression', make_pipeline(StandardScaler(), PCA(n_components=5), LogisticRegression(max_iter=400))),
    ]:
        model.fit(train[predictors], train.source_label)
        pred = model.predict(test[predictors])
        rows.append({'task':'source_classification', 'model':label, **classification_metrics(test.source_label, pred, 'macro')})
    return rows


def run_next_bit_prediction():
    tr = pd.read_csv(DATA / 'next_bit_train.csv')
    te = pd.read_csv(DATA / 'next_bit_test.csv')
    cols = [c for c in tr if c.startswith('bit_t_minus_')]
    rows = []
    for source in sorted(tr.source_label.unique()):
        train, test = tr[tr.source_label == source], te[te.source_label == source]
        y_train, y_test = train.next_bit.to_numpy(), test.next_bit.to_numpy()
        prob_base = np.clip(np.mean(y_train), 1e-8, 1 - 1e-8)
        model = LogisticRegression(max_iter=300)
        model.fit(train[cols], y_train)
        predicted_prob = model.predict_proba(test[cols])[:, list(model.classes_).index(1)]
        for name, probabilities in [
            ('Source-specific constant probability', np.full(len(test), prob_base)),
            ('Logistic Regression (16 previous bits)', predicted_prob),
        ]:
            yhat = (probabilities >= 0.5).astype(int)
            rows.append({
                'task':'next_bit', 'source_label':int(source), 'model':name,
                **classification_metrics(y_test, yhat, 'binary'),
                'predictive_cross_entropy_bits': log_loss(y_test, probabilities, labels=[0, 1]) / math.log(2),
            })
    return rows


if __name__ == '__main__':
    source_results = pd.DataFrame(run_source_classification())
    nextbit_results = pd.DataFrame(run_next_bit_prediction())
    source_results.to_csv(OUT / 'source_model_metrics.csv', index=False)
    nextbit_results.to_csv(OUT / 'next_bit_model_metrics.csv', index=False)
    print('\nSOURCE CLASSIFICATION (NOT NEXT-BIT ENTROPY)')
    print(source_results.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nNEXT-BIT PREDICTABILITY (cross-entropy is not certified source entropy)')
    print(nextbit_results.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nSaved metrics in {OUT}')
