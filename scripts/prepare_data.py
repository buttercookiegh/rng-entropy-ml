"""Prepare explicit features/labels from the original 100-bit RNG sequences.

No third-party dependencies; run with ``uv run python scripts/prepare_data.py``.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'AI_2qubits_training_data.txt'
OUT = ROOT / 'data' / 'processed'
SEED = 42
WINDOW = 16
PREDICTORS = [
    'frequency_ones', 'number_of_runs', 'longest_run', 'longest_zero_run',
    'longest_one_run', 'transition_rate', 'lag1_autocorrelation',
    'lag2_autocorrelation', 'pair_00', 'pair_01', 'pair_10', 'pair_11',
    'block_frequency_std',
]
DIAGNOSTICS = ['shannon_entropy']


def read_sequences(path: Path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            tokens = line.split()
            if len(tokens) != 2:
                raise ValueError(f'Line {line_number}: expected bitstring and label')
            bitstring, source_label = tokens
            if len(bitstring) != 100 or set(bitstring) - set('01'):
                raise ValueError(f'Line {line_number}: expected exactly 100 binary digits')
            if source_label not in ('1', '2', '3'):
                raise ValueError(f'Line {line_number}: unknown source label')
            rows.append((f'S{len(rows)+1:04d}', bitstring, int(source_label)))
    if not rows:
        raise ValueError('No data rows found')
    return rows


def pearson_lag(bits: list[int], lag: int) -> float:
    x, y = bits[:-lag], bits[lag:]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    top = sum((a - mx) * (b - my) for a, b in zip(x, y))
    xx = sum((a - mx) ** 2 for a in x)
    yy = sum((b - my) ** 2 for b in y)
    return top / math.sqrt(xx * yy) if xx > 0 and yy > 0 else 0.0


def extract_features(bitstring: str):
    bits = [int(b) for b in bitstring]
    p = sum(bits) / len(bits)
    runs: list[tuple[int, int]] = []
    current, length = bits[0], 1
    for b in bits[1:]:
        if b == current:
            length += 1
        else:
            runs.append((current, length))
            current, length = b, 1
    runs.append((current, length))
    pairs = Counter(bitstring[i:i+2] for i in range(len(bitstring)-1))
    blocks = [statistics.fmean(bits[i:i+10]) for i in range(0, 100, 10)]
    shannon = -sum(q * math.log2(q) for q in (p, 1-p) if q > 0)
    return {
        'frequency_ones': p,
        'number_of_runs': len(runs),
        'longest_run': max(length for _, length in runs),
        'longest_zero_run': max((length for value, length in runs if value == 0), default=0),
        'longest_one_run': max((length for value, length in runs if value == 1), default=0),
        'transition_rate': (len(runs)-1)/99,
        'lag1_autocorrelation': pearson_lag(bits, 1),
        'lag2_autocorrelation': pearson_lag(bits, 2),
        'pair_00': pairs['00']/99,
        'pair_01': pairs['01']/99,
        'pair_10': pairs['10']/99,
        'pair_11': pairs['11']/99,
        'block_frequency_std': statistics.pstdev(blocks),
        'shannon_entropy': shannon,
    }


def make_split(rows, seed: int = SEED):
    rng = random.Random(seed)
    ids_by_source: dict[int, list[str]] = {}
    for identifier, _, source in rows:
        ids_by_source.setdefault(source, []).append(identifier)
    assignment = {}
    for source in sorted(ids_by_source):
        ids = sorted(ids_by_source[source])
        rng.shuffle(ids)
        cutoff = int(len(ids) * 0.8)
        assignment.update({identifier: 'train' if index < cutoff else 'test'
                           for index, identifier in enumerate(ids)})
    return assignment


def make_data(window: int = WINDOW):
    if not 1 <= window < 100:
        raise ValueError('Window must be in [1, 99]')
    rows = read_sequences(RAW)
    assignment = make_split(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    head_seq = ['sequence_id', 'source_label', 'split', *PREDICTORS, *DIAGNOSTICS]
    head_bit = ['sequence_id', 'source_label', 'window_start',
                *[f'bit_t_minus_{x:02d}' for x in range(window, 0, -1)], 'next_bit']
    with (OUT / 'sequence_features.csv').open('w', newline='', encoding='utf-8') as fs, \
         (OUT / 'next_bit_train.csv').open('w', newline='', encoding='utf-8') as ftr, \
         (OUT / 'next_bit_test.csv').open('w', newline='', encoding='utf-8') as fte:
        ws, wt, we = csv.writer(fs), csv.writer(ftr), csv.writer(fte)
        ws.writerow(head_seq)
        wt.writerow(head_bit)
        we.writerow(head_bit)
        counts = Counter()
        for sequence_id, bitstring, source in rows:
            split = assignment[sequence_id]
            feats = extract_features(bitstring)
            ws.writerow([sequence_id, source, split, *[f'{feats[f]:.8f}' if isinstance(feats[f], float) else feats[f] for f in [*PREDICTORS, *DIAGNOSTICS]]])
            writer = wt if split == 'train' else we
            for pos in range(100 - window):
                input_bits = [int(b) for b in bitstring[pos:pos+window]]
                target = int(bitstring[pos+window])
                writer.writerow([sequence_id, source, pos, *input_bits, target])
                counts[split] += 1
    print(f'Validated {len(rows)} source sequences: {Counter(label for _, _, label in rows)}')
    print(f'Sequence features: {len(rows)} rows; {len(PREDICTORS)} engineered predictors + {len(DIAGNOSTICS)} diagnostic')
    print(f'Next-bit rows (window={window}): {dict(counts)}')
    print(f'Outputs: {OUT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=WINDOW, help='Number of preceding bits used to predict the next bit (default 16)')
    args = parser.parse_args()
    make_data(window=args.window)
