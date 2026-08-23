# Credit Risk Scorecard

An interpretable credit risk scorecard built the way lenders deploy them under regulatory constraint — weight-of-evidence binning, information value feature selection, logistic regression, and points scaling.

**Result:** Seven-variable scorecard achieving **AUC 0.857 / Gini 0.713 / KS 0.557** on a held-out test set, with no overfitting. Default rates range from 54.2% in the lowest score band to 0.46% in the highest — a 118x spread.

---

## Why a scorecard rather than a classifier

A lender who declines an applicant must tell them why. That legal requirement rules out models whose decisions can't be decomposed into named factors, which is why scorecards remain standard in credit risk despite better-performing alternatives existing.

This project builds one end to end and reports what the interpretability costs: an explainable seven-variable logistic regression reaches 0.86 AUC with test performance marginally exceeding train.

## Data

[Kaggle, "Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) (2011). 150,000 borrowers, 10 predictors, 6.68% serious delinquency rate. Not redistributed here — download separately.

## Method

1. Bin each variable, enforcing monotonic risk and adequate bin volume
2. Compute Weight of Evidence per bin: `ln(good% / bad%)`
3. Select features by Information Value; drop non-monotonic and sub-threshold variables
4. Verify multicollinearity via VIF after transformation
5. Fit logistic regression on WOE-transformed features (70/30 stratified split)
6. Validate on held-out test data: AUC, Gini, KS
7. Scale log-odds to points (base 600, 50:1 odds, PDO 20)

## Data quality findings

Three anomalies were found. Each was investigated before treatment, and the correct treatment differed in each case — which is the point.

**Sentinel codes.** Values of 96 and 98 appear in the delinquency counts. These aren't counts; they're placeholders from the source system. The 269 affected records **default at 54.65% vs. 6.68% overall.** Dropping them as errors would have discarded the strongest single signal in the dataset. Retained and binned separately.

**Extreme utilization.** Values reach 50,708 — impossible as a ratio. But borrowers in the 1-2 range default at 40% (legitimate over-limit stress) while those above 10 default at 7.05%, essentially the base rate. Opposite conclusion from the sentinel codes: these carry no signal and are corrupted denominators.

**DebtRatio as a disguised proxy.** Values reach 329,664. Of the 28,877 records above 10, **26,771 (93%) are the same records missing MonthlyIncome** — the ratio is uncomputable without a denominator. Their default rate matches the missing-income population, not any debt-burden pattern. Binned as "not meaningful" rather than capped.

**Missingness is informative.** Borrowers with no reported income default at 5.61% vs. 6.95% — lower, not higher. Missing retained as its own bin rather than imputed.

## Feature selection

| Variable | IV | Decision |
|---|---|---|
| Revolving utilization | 1.133 | Retained |
| 90+ days late | 0.878 | Retained |
| 30-59 days late | 0.758 | Retained |
| 60-89 days late | 0.600 | Retained |
| Age | 0.256 | Retained |
| Open credit lines | 0.080 | Dropped — non-monotonic |
| Monthly income | 0.076 | Retained (p=0.003) |
| Real estate loans | 0.062 | Dropped — non-monotonic |
| Debt ratio | 0.061 | Retained |
| Dependents | 0.036 | Dropped — below threshold |

Utilization and 90-day delinquency both exceed the IV 0.50 leakage-check threshold. Both verified benign — they're the two most established predictors in consumer credit and are available at scoring time.

VIF after WOE transformation: 1.08-1.24 across all seven features, well below the 5.0 threshold.

## Results

| Metric | Train | Test |
|---|---|---|
| AUC | 0.8557 | 0.8565 |
| Gini | 0.7114 | 0.7131 |
| KS | — | 0.5573 |

Test exceeding train indicates no overfitting — coarse binning removes noise a flexible model would memorize.

**Score band performance** (20,000-borrower sample):

| Score | Borrowers | Default rate |
|---|---|---|
| 500 or below | 627 | 54.23% |
| 500-550 | 2,079 | 21.69% |
| 550-600 | 9,679 | 4.65% |
| 600-620 | 6,526 | 0.83% |
| 620+ | 1,089 | 0.46% |

Monotonic across all bands. A cutoff at 550 would decline 13.5% of applicants and avoid roughly half of all defaults.

Point ranges show practical influence: utilization spans 63 points, income spans 4. Income is statistically significant and practically negligible.

## Limitations

- **Vintage.** 2011 data. Methodology is period-independent; coefficients are not.
- **Compressed top-end distribution.** Median 592, max 623. Separates good from bad well, discriminates poorly among good borrowers.
- **No out-of-time validation.** Split is random, not temporal. Model risk review requires later-period validation this dataset can't support.
- **No reject inference.** Only booked accounts are present; production scorecards must account for declined applicants with no performance outcome.
- **No fairness testing.** No demographic fields beyond age. Any deployed model requires disparate impact review.
- **Arbitrary scale.** Base 600 / 50:1 / PDO 20 is conventional but not comparable to FICO.

## Next steps

- Reject inference simulation
- Population stability monitoring (PSI)
- Disparate impact analysis on a dataset with demographic fields
- Gradient boosting comparison to quantify the interpretability trade-off

## Reproducing

Requires pandas, numpy, scikit-learn, statsmodels.

Place `cs-training.csv` in `data/`, then run `scorecard.ipynb` top to bottom.

---

Luke Freyman
