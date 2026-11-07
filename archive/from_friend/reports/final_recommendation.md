# Final recommendation

Keep the confirmed top1, CPU70% + one-layer GCN30%, Public MSE 886520.98398. No new upload is justified by this research batch.

Recommended submission: versions/top1/submission.csv. Prediction checkpoint: versions/top1/blend_model.joblib.

The nested search worsened mean outer MSE compared with the fixed reference on both seeds. Some fixed neural replacements show small average changes, but no sufficiently robust, independent improvement supports replacing the champion.

| cv | seed | tree_estimators | reference_mse | nested_mse | relative_change |
|---|---|---|---|---|---|
| canonical | 42 | 250 | 4,102,621.04 | 4,143,129.45 | 0.01 |
| canonical | 42 | 350 | 4,101,813.50 | 4,142,018.06 | 0.01 |
| canonical | 43 | 250 | 4,048,200.04 | 4,090,572.65 | 0.01 |
| canonical | 43 | 350 | 4,046,878.64 | 4,089,912.70 | 0.01 |

The recovered historical two-layer model has reported Public MSE 898313.50898. It is retained in versions/candidates/historical_two_layer_70_30 and is the actual second-best reported result. The older immutable top2 folder has score 936601.51958; versions/ranking.json records this distinction.

Legacy 35% GCN remains unconfirmed and is archived with a complete, validated checkpoint. It is not recommended as a new improved model by this batch.

No public score has been inferred from local MSE. No test-derived max-prediction constraint is used for the new decision.

Next useful work should require a predeclared validation improvement across additional molecular groups and decompose the extreme-row contribution. Do not spend uploads on the old monotonic percentage sweep.

Commands: python main.py predict --smiles "CCO"; python main.py similar --smiles "CCO" --top 10; python main.py batch --input molecules.csv --output predictions.csv; python main.py validate-submission --file versions/top1/submission.csv; python main.py evaluate --cv canonical; python main.py evaluate --cv scaffold --seeds 42; python main.py optimize --trials 8.
