# Astra Best solution

Start with the [main README](../../README.md) for the plain-language explanation, results and commands to run the trained model.

- [Retrain the architecture step by step](TRAINING.md)
- [Detailed results and known weaknesses](RESULTS.txt)
- [Technical description from the original delivery](README.txt)
- [Original reproduction notes](REPRODUCE.txt)
- [Validation predictions](results/original_validation_predictions.csv)
- [Holdout comparisons](results/holdout/comparison.csv)
- [Unscored public-test predictions](results/test_predictions.csv)

The root `model.joblib` and `policy.json` are the frozen model and settings. Retraining creates new files under `runs/`; it does not replace those two delivered files.
