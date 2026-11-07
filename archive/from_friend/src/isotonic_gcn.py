"""Train an OOF-isotonic calibrated GCN blend using official train.csv only.

The released candidate is 73.57% of the saved CPU reference model and 26.43%
of a GCN prediction calibrated by isotonic regression. The scaler, GCNs, and
calibrator are all fitted from train.csv; test.csv is read only for inference.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.validation import ROOT, training_data, folds, sha, write_json, assert_protected_hashes
from src.tail_research import TailRegressor, WORK as TAIL_WORK
from src.tail_calibration import CrossFitCalibrator
from src.inference import predict_bundle
from competition_data import read_competition_csv, require_safe_output

CPU_WEIGHT = 0.7357
GCN_WEIGHT = 0.2643
SEEDS = (42, 43)
FOLDS = 5
EPOCHS = 80


def fit_oof(df, y, out, reuse=True):
    """Generate complete canonical-group OOF GCN predictions for calibration."""
    import vacv_cpu_improved as cpu
    protocol_path = TAIL_WORK/'protocol.json'
    if reuse and protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        expected = (protocol.get('training_sha256') == sha(ROOT/'train.csv')
                    and protocol.get('epochs') == EPOCHS
                    and protocol.get('grouped_folds') == FOLDS
                    and protocol.get('seeds') == list(SEEDS))
        if not expected:
            raise ValueError('Cached OOF GCNs do not match train hash/seeds/folds/epochs; use --retrain-oof')
    raw = cpu.chem_features(df.SMILES.astype(str).tolist())
    oof = np.zeros(len(y), dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    for seed in SEEDS:
        split = folds(y, raw['canon'], seed, FOLDS)
        seed_pred = np.full(len(y), np.nan)
        for fold, (tr, va) in enumerate(split, 1):
            cache = TAIL_WORK / f'{seed}_control_log' / f'fold_{fold}.joblib'
            if reuse and cache.exists():
                model = joblib.load(cache)
                if model.training_rows != len(tr) or model.mode != 'control_log':
                    raise ValueError(f'Incompatible cached OOF model: {cache}')
            else:
                model = TailRegressor('control_log', epochs=EPOCHS).fit(
                    df.iloc[tr].reset_index(drop=True), y[tr], seed * 100 + fold - 1)
                cache.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, cache, compress=3)
            seed_pred[va] = model.predict(df.iloc[va].reset_index(drop=True))
            if seed == SEEDS[0]:
                fold_ids[va] = fold
            print(f'OOF GCN seed={seed}, fold={fold}, rows={len(va)}', flush=True)
        if not np.isfinite(seed_pred).all():
            raise ValueError(f'Incomplete OOF predictions for seed {seed}')
        oof += seed_pred / len(SEEDS)
    if not np.isfinite(oof).all() or not np.all(fold_ids > 0):
        raise ValueError('OOF prediction coverage is incomplete')
    pd.DataFrame({'row': np.arange(len(y)), 'fold': fold_ids, 'true': y,
                  'gcn_oof_seed_average': oof}).to_csv(out / 'gcn_oof.csv', index=False)
    return oof


def train_candidate(output, base_model=ROOT/'versions/reference/model.joblib', reuse_oof=True):
    assert_protected_hashes()
    output = require_safe_output(output).resolve()
    if output.exists():
        raise FileExistsError(f'Candidate output already exists: {output}')
    output.mkdir(parents=True)
    df = training_data()
    y = df['SI VACV'].to_numpy(float)
    original_y = y.copy()
    oof = fit_oof(df, y, output, reuse=reuse_oof)
    calibrator = CrossFitCalibrator('isotonic').fit(oof, y)

    # Fit two deterministic final GCNs on all official training rows.
    models = [TailRegressor('control_log', epochs=EPOCHS).fit(
        df, y, final_seed) for final_seed in (4242, 4343)]
    base_bundle = joblib.load(base_model)
    bundle = {'model_type': 'isotonic_gcn_blend', 'base_bundle': base_bundle,
              'gcn_models': models, 'calibrator': calibrator,
              'cpu_weight': CPU_WEIGHT, 'gcn_weight': GCN_WEIGHT,
              'training_rows': len(df), 'training_sha256': sha(ROOT/'train.csv'),
              'base_model_sha256': sha(base_model), 'oof_seeds': list(SEEDS),
              'oof_folds': FOLDS, 'epochs': EPOCHS,
              'calibration': 'isotonic fitted to mean of canonical-group 5-fold OOF GCN predictions'}
    model_path = output/'prediction_bundle.joblib'
    joblib.dump(bundle, model_path, compress=3)
    np.testing.assert_array_equal(y, original_y)

    # The official test split is accessed only after OOF calibration and final fits.
    test = read_competition_csv(ROOT/'test.csv', 'test')
    sample = pd.read_csv(ROOT/'sample_submit.csv')
    if len(sample) != len(test) or not np.array_equal(sample['index'], test['index']):
        raise ValueError('Test rows do not match sample submission indices')
    pred = predict_bundle(bundle, test)
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError('Candidate produced invalid predictions')
    submission = output/'submission.csv'
    pd.DataFrame({'index': sample['index'], 'SI VACV': pred}).to_csv(submission, index=False)
    write_json(output/'manifest.json', {
        'status': 'pending_user_kaggle_check', 'public_mse': None,
        'cpu_weight': CPU_WEIGHT, 'gcn_weight': GCN_WEIGHT,
        'training_sha256': sha(ROOT/'train.csv'), 'test_sha256': sha(ROOT/'test.csv'),
        'submission_sha256': sha(submission), 'prediction_model_sha256': sha(model_path),
        'prediction_model_path': str(model_path), 'oof_seeds': list(SEEDS),
        'oof_folds': FOLDS, 'epochs': EPOCHS, 'test_used_for_model_selection': False,
        'note': 'OOF isotonic GCN blend; external score not yet known.'})
    # Reopen checkpoint and verify reproducibility before presenting the file.
    restored = joblib.load(model_path)
    reproduced = predict_bundle(restored, test)
    np.testing.assert_allclose(pred, reproduced, rtol=1e-9, atol=1e-8)
    assert_protected_hashes()
    print(json.dumps({'submission': str(submission), 'rows': len(pred),
                      'max_prediction': float(pred.max()), 'cpu_weight': CPU_WEIGHT,
                      'gcn_weight': GCN_WEIGHT, 'submission_sha256': sha(submission),
                      'model_reproduced': True}, indent=2))
    return submission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results/isotonic_gcn_7357_2643')
    p.add_argument('--base-model', default='versions/reference/model.joblib')
    p.add_argument('--retrain-oof', action='store_true', help='Ignore cached OOF GCN checkpoints')
    a = p.parse_args()
    train_candidate(ROOT/a.out if not Path(a.out).is_absolute() else a.out,
                    ROOT/a.base_model if not Path(a.base_model).is_absolute() else a.base_model,
                    reuse_oof=not a.retrain_oof)


if __name__ == '__main__':
    main()
