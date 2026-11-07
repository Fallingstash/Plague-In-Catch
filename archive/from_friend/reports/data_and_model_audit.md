# Data and model audit



Official train SHA256: `d9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8`.

320 rows; 312 canonical groups; 83 zeros; max SI=36066.667.

Targets are never recomputed. Only official SMILES and SI enter the new pipeline.

Old 1%-sweep with a test max-prediction constraint was transductive model selection and is excluded from future selection.

No Git repository exists; code snapshots and SHA256 manifests are used instead.

The original second-layer submission (898313.50898) is the actual second-best reported score. Its overwritten checkpoint and original CSV have now been recovered in versions/candidates/historical_two_layer_70_30. The older protected top2 folder is preserved; versions/ranking.json records the actual ranking.

The old default inference model was a stale 5% blend; CLI default now targets top1.

CPU selects epochs in a group-disjoint inner holdout. GCN uses 80 fixed epochs, no early stopping. Graph adjacency ignores bond type; mean pooling may dilute rare substructures.

Legacy CPU CV used 250 trees while final CPU used 350. Legacy runs reproduce the old numbers; new reference350 and nested tree350 runs correct that mismatch without changing historical artifacts.

GCN inverse transform clamps log predictions to [0,14] for numerical behavior; it is not a validation-derived SI cutoff.

Existing ensemble OOF weights were tuned on the same OOF and are exploratory. Prospective nested selection is reported separately.



Seed 42: CPU MSE 4190501.45; GCN 4006894.82; fixed 70/30 4102621.04. Extreme row error share: 94.85%. GCN wins on 169/320 rows. Error correlation 0.9811.

Seed 43: CPU MSE 4190971.01; GCN 3747659.08; fixed 70/30 4048200.04. Extreme row error share: 95.41%. GCN wins on 179/320 rows. Error correlation 0.9957.
