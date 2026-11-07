# Measured model comparison

Canonical 5-fold grouped CV, seeds 42/43. MSE is on original SI. These are standalone models except reference, which is the fixed CPU70/GCN30 blend.

| model | mean_oof_mse | mean_mae | split_seeds |
|---|---|---|---|
| hybrid_gcn | 3,874,141.05 | 223.57 | 2 |
| bond_gcn | 3,876,962.71 | 225.14 | 2 |
| gcn_single | 3,877,276.95 | 225.17 | 2 |
| gcn_mean3 | 3,921,517.49 | 221.88 | 2 |
| dmpnn | 4,057,152.89 | 223.75 | 2 |
| reference350 | 4,074,346.07 | 220.81 | 2 |
| reference | 4,075,410.54 | 221.15 | 2 |
| cpu_component350 | 4,189,514.74 | 221.02 | 2 |
| cpu_component | 4,190,736.23 | 221.73 | 2 |
| krr_mix_0.03 | 4,215,746.72 | 209.71 | 2 |
| residual_mlp | 4,227,483.21 | 209.06 | 2 |
| gp_log | 4,253,906.87 | 208.71 | 2 |
| krr_mix_0.3 | 4,257,481.25 | 208.56 | 2 |
| extra_trees_log | 4,258,756.39 | 209.41 | 2 |
| catboost_d6_log | 4,276,115.25 | 208.67 | 2 |
| krr_morgan2_2048_log | 4,277,035.13 | 209.15 | 2 |
| krr_morgan2_1024_log | 4,277,106.54 | 209.24 | 2 |
| catboost_d4_log | 4,277,929.88 | 209.05 | 2 |
| krr_morgan3_2048_log | 4,281,085.24 | 209.30 | 2 |
| krr_torsion_log | 4,282,229.50 | 209.14 | 2 |
| catboost_desc_log | 4,283,207.05 | 210.67 | 2 |
| krr_atom_pair_log | 4,285,093.69 | 209.77 | 2 |
| krr_rdkit2048_log | 4,285,787.07 | 210.50 | 2 |
| krr_mix_3.0 | 4,290,242.71 | 210.29 | 2 |
| krr_mix_30.0 | 4,299,032.17 | 212.72 | 2 |
| catboost_d6_raw | 4,358,971.49 | 260.99 | 2 |
| krr_morgan3_2048_raw | 4,402,890.59 | 383.97 | 2 |
| krr_morgan2_2048_raw | 4,420,328.98 | 388.82 | 2 |
| krr_morgan2_1024_raw | 4,430,827.81 | 400.11 | 2 |
| krr_torsion_raw | 4,444,075.41 | 388.47 | 2 |
| krr_atom_pair_raw | 4,473,058.56 | 386.36 | 2 |
| optuna_catboost_tuned | 4,509,391.03 | 305.84 | 2 |
| catboost_d4_raw | 4,577,269.42 | 277.09 | 2 |
| krr_rdkit2048_raw | 4,627,231.09 | 383.33 | 2 |
| gp_raw | 4,671,675.07 | 420.98 | 2 |
| catboost_desc_raw | 5,447,940.37 | 376.17 | 2 |

## Fixed 70/30 architecture replacements

| model | reference_recipe | cv | seed | MSE | reference_mse | relative_mse_change |
|---|---|---|---|---|---|---|
| CPU70+gcn_mean330 | reference | canonical | 42 | 4,104,909.30 | 4,102,621.04 | 0.00 |
| CPU70+bond_gcn30 | reference | canonical | 42 | 4,102,616.26 | 4,102,621.04 | -0.00 |
| CPU70+hybrid_gcn30 | reference | canonical | 42 | 4,103,264.65 | 4,102,621.04 | 0.00 |
| CPU70+residual_mlp30 | reference | canonical | 42 | 4,197,367.06 | 4,102,621.04 | 0.02 |
| CPU70+dmpnn30 | reference | canonical | 42 | 4,132,293.81 | 4,102,621.04 | 0.01 |
| CPU70+gcn_mean330 | reference350 | canonical | 42 | 4,104,315.69 | 4,101,813.50 | 0.00 |
| CPU70+bond_gcn30 | reference350 | canonical | 42 | 4,101,809.50 | 4,101,813.50 | -0.00 |
| CPU70+hybrid_gcn30 | reference350 | canonical | 42 | 4,102,531.81 | 4,101,813.50 | 0.00 |
| CPU70+residual_mlp30 | reference350 | canonical | 42 | 4,197,239.82 | 4,101,813.50 | 0.02 |
| CPU70+dmpnn30 | reference350 | canonical | 42 | 4,131,769.23 | 4,101,813.50 | 0.01 |
| CPU70+gcn_mean330 | reference | canonical | 43 | 4,090,446.33 | 4,048,200.04 | 0.01 |
| CPU70+bond_gcn30 | reference | canonical | 43 | 4,048,087.67 | 4,048,200.04 | -0.00 |
| CPU70+hybrid_gcn30 | reference | canonical | 43 | 4,052,326.50 | 4,048,200.04 | 0.00 |
| CPU70+residual_mlp30 | reference | canonical | 43 | 4,197,037.34 | 4,048,200.04 | 0.04 |
| CPU70+dmpnn30 | reference | canonical | 43 | 4,148,128.20 | 4,048,200.04 | 0.02 |
| CPU70+gcn_mean330 | reference350 | canonical | 43 | 4,089,120.13 | 4,046,878.64 | 0.01 |
| CPU70+bond_gcn30 | reference350 | canonical | 43 | 4,046,766.40 | 4,046,878.64 | -0.00 |
| CPU70+hybrid_gcn30 | reference350 | canonical | 43 | 4,051,018.75 | 4,046,878.64 | 0.00 |
| CPU70+residual_mlp30 | reference350 | canonical | 43 | 4,195,759.03 | 4,046,878.64 | 0.04 |
| CPU70+dmpnn30 | reference350 | canonical | 43 | 4,146,814.22 | 4,046,878.64 | 0.02 |
| CPU70+gcn_mean330 | reference | scaffold | 42 | 4,170,031.66 | 4,170,552.65 | -0.00 |
| CPU70+hybrid_gcn30 | reference | scaffold | 42 | 4,171,375.81 | 4,170,552.65 | 0.00 |
| CPU70+dmpnn30 | reference | scaffold | 42 | 4,131,807.85 | 4,170,552.65 | -0.01 |
| CPU70+gcn_mean330 | reference350 | scaffold | 42 | 4,171,120.52 | 4,171,642.84 | -0.00 |
| CPU70+hybrid_gcn30 | reference350 | scaffold | 42 | 4,172,462.15 | 4,171,642.84 | 0.00 |
| CPU70+dmpnn30 | reference350 | scaffold | 42 | 4,132,901.90 | 4,171,642.84 | -0.01 |

## Nested selection: independent outer-fold scoring

| cv | seed | tree_estimators | reference_mse | nested_mse | relative_change |
|---|---|---|---|---|---|
| canonical | 42 | 250 | 4,102,621.04 | 4,143,129.45 | 0.01 |
| canonical | 42 | 350 | 4,101,813.50 | 4,142,018.06 | 0.01 |
| canonical | 43 | 250 | 4,048,200.04 | 4,090,572.65 | 0.01 |
| canonical | 43 | 350 | 4,046,878.64 | 4,089,912.70 | 0.01 |

## Scaffold CV

| model | seed | oof_mse | oof_mae |
|---|---|---|---|
| dmpnn | 42 | 3,912,413.05 | 226.61 |
| gcn_mean3 | 42 | 4,033,580.32 | 221.78 |
| gcn_single | 42 | 4,039,004.26 | 226.70 |
| hybrid_gcn | 42 | 4,040,674.18 | 226.54 |
| reference | 42 | 4,170,552.65 | 231.07 |
| reference350 | 42 | 4,171,642.84 | 231.27 |
| krr_mix_0.03 | 42 | 4,236,651.14 | 207.53 |
| cpu_component | 42 | 4,236,737.39 | 235.59 |
| cpu_component350 | 42 | 4,238,768.05 | 235.87 |
| gp_log | 42 | 4,266,551.44 | 208.58 |
| catboost_d4_log | 42 | 4,275,561.84 | 209.73 |
| krr_morgan2_2048_raw | 42 | 4,447,796.64 | 396.94 |
| catboost_d6_raw | 42 | 4,590,787.57 | 278.69 |
| catboost_d4_raw | 42 | 5,217,859.04 | 305.23 |

Screening winners are not independent estimates after selecting from this table. See paired error CSVs and bootstrap uncertainty. Test labels were never used.
