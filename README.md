# Predicting-F1-Pitstops

This is a submission to the [Kaggle Playground S6E5 competition](https://www.kaggle.com/competitions/playground-series-s6e5) on predicting whether a Formula 1 driver will pit in the next lap.

Both Jupyter notebooks uses the XGBoost model – an optimised gradient boosting algorithm that combines multiple weak models into a stronger, high-performance model.

Based on [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/xgboost/), XGBoost basically builds decision trees sequentially:
1. Start with a base learner: First decision tree is trained on the train data, aimed to predict the average of the target variable.
2. Calculate errors (between the predicted and actual values)
3. Train the next tree: Correct the errors in the first tree.
4. Repeat the process until a stopping criterion is met.
5. Combine all predictions: The final prediction is the sum of all predictions from al the trees.

## Pipeline 

1. **EDA**: class imbalance, tyre-life signal, compound effect, race-progress patterns, lap-time distributions, and feature correlations.
2. **Feature engineering**: 20 derived features: polynomial transforms, stint/tyre/progress interactions, and domain signals (`Degradation_per_Lap`, `Pit_Urgency`).
3. **Preprocessing**: target encoding for `Driver`/`Compound`/`Race`; median imputation. Fit on train only to prevent leakage.
4. **Train/test split**: stratified 80/20; `scale_pos_weight ≈ 4` for class imbalance.
5. **Hyperparameter tuning**: Bayesian optimisation over 7 XGBoost parameters, maximising validation AUC. Done via Optuna (50 trials, TPE sampler).
6. **Model training**: XGBoost and LightGBM with early stopping on validation AUC; LightGBM marginally outperformed (0.9661 vs 0.9632).
7. **Submission**: CSV of test-set probabilities for the competition.

## Results

- Validation ROC-AUC of **0.94732**
- `TyreLife` is the strongest indicator – pit probability rises near-monotonically with tyre age.
- Pit stops cluster in the 40-70% race progress window, with near zero probability in the first and last 10% if the race.
- HARD compound has a counterintuitively high pit rate due to stint-length confounding *i.e. drivers run longer on HARD, so by the time they stop, tyre age is already high*
- Top predictive features: `TyreLife`, `LapNumber`, `Stint`, `RaceProgress`, and engineered interactions (`Stint × TyreLife`, `Pit_Urgency`)

*The choice of Bayesian optimisation is due to its efficient learning from previous trials. The 7 parameters being tuned are:*
```
'max_depth'        # how deep each tree grows (complexity)
'learning_rate'    # step size per boosting round (smaller = more careful)
'subsample'        # fraction of rows sampled per tree (prevents overfitting)
'colsample_bytree' # fraction of features sampled per tree (prevents overfitting)
'reg_lambda'       # L2 regularisation strength (penalises large weights)
'reg_alpha'        # L1 regularisation strength (encourages sparsity)
'min_child_weight' # minimum data required in a leaf (controls tree growth)
```
*Each trial picks one combination of these 7 values, trains XGBoost to convergence with early stopping, and returns the validation AUC. After 50 trials, Optuna reports the best combination found.*
