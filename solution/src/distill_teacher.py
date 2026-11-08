"""Distill the saved top1 teacher on official train SMILES; never fit on test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import GroupKFold
from competition_data import read_competition_csv, require_safe_output
from src.validation import training_data, sha
from src.inference import predict_bundle
from src.gpu_training import fingerprints, fit, predict, BITS


def errors(y, p):
    d = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    return {'MSE': float(np.mean(d ** 2)), 'MAE': float(np.mean(np.abs(d))), 'max_absolute_error': float(np.max(np.abs(d)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--teacher', default='versions/top1/blend_model.joblib')
    p.add_argument('--train', default='train.csv')
    p.add_argument('--test', default='test.csv')
    p.add_argument('--reference-submission', default='versions/top1/submission.csv')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.epochs < 1 or a.folds < 2: p.error('epochs must be positive and folds at least 2')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    out = require_safe_output(a.out)
    out.mkdir(parents=True, exist_ok=False)
    train = training_data(a.train)
    test = read_competition_csv(a.test, 'test')
    teacher = joblib.load(a.teacher)
    teacher_train = predict_bundle(teacher, train)
    teacher_test = predict_bundle(teacher, test)
    reference = pd.read_csv(a.reference_submission)
    if list(reference.columns) != ['index', 'SI VACV'] or not np.array_equal(reference['index'], test['index']):
        raise ValueError('Reference submission and test indices do not match')
    reference_values = reference['SI VACV'].to_numpy(float)
    restoration = errors(reference_values, teacher_test)
    if not np.allclose(reference_values, teacher_test, rtol=1e-10, atol=1e-8):
        raise ValueError(f'Teacher does not reproduce reference submission: {restoration}')
    x, groups = fingerprints(train.SMILES)
    test_x, _ = fingerprints(test.SMILES)
    labels = train['SI VACV'].to_numpy(float)
    oof = np.zeros(len(train))
    fold = np.zeros(len(train), dtype=int)
    for i, (tr, va) in enumerate(GroupKFold(n_splits=a.folds).split(x, teacher_train, groups)):
        model = fit(x[tr], teacher_train[tr], a.epochs, a.seed+i, device)
        oof[va] = predict(model, x[va], device)
        fold[va] = i
        print(f'Fold {i+1}/{a.folds} completed', flush=True)
    model = fit(x, teacher_train, a.epochs, a.seed, device)
    student_train = predict(model, x, device)
    student_test = predict(model, test_x, device)
    pd.DataFrame({'row': np.arange(len(train)), 'SMILES': train.SMILES, 'fold': fold, 'actual_SI': labels, 'teacher': teacher_train, 'student_oof': oof, 'student_fit': student_train}).to_csv(out/'train_predictions.csv', index=False)
    pd.DataFrame({'index': test['index'], 'SI VACV': teacher_test}).to_csv(out/'reference_restored.csv', index=False)
    pd.DataFrame({'index': test['index'], 'SI VACV': student_test}).to_csv(out/'student_submission.csv', index=False)
    torch.save({'state_dict': model.cpu().state_dict(), 'bits': BITS, 'train_sha256': sha(a.train), 'teacher_sha256': sha(a.teacher)}, out/'student.pt')
    report = {'device': device, 'epochs': a.epochs, 'folds': a.folds, 'seed': a.seed, 'train_sha256': sha(a.train), 'teacher_sha256': sha(a.teacher), 'reference_sha256': sha(a.reference_submission), 'restoration_vs_saved_reference': restoration, 'student_oof_vs_teacher_train': errors(teacher_train, oof), 'student_fit_vs_teacher_train': errors(teacher_train, student_train), 'student_test_vs_teacher_test': errors(teacher_test, student_test), 'student_oof_vs_train_labels_NOT_independent': errors(labels, oof), 'caveat': 'Teacher was trained on all official train labels. Student OOF measures imitation of this fixed teacher, not unbiased generalization to true SI. Test teacher outputs are used only for post-training comparison.'}
    (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__': main()
