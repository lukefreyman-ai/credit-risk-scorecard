# Interview guide - credit-risk-scorecard

## What it does, in three sentences

It builds a classic bank-style credit scorecard on the Kaggle "Give Me Some Credit" data (150,000 borrowers, 6.7 %
went 90+ days delinquent within two years). Each characteristic (utilisation, age, past delinquencies, debt ratio,
income) is binned, each bin is replaced by its weight of evidence, and a logistic regression on those WOE values is
turned into points (600 points = 50:1 odds, 20 points doubles the odds). The result is a score where a 500-550 band has a
21.6 % default rate and the top band under 1 %.

## Why each key decision was made (plain language)

- **WOE binning instead of feeding raw numbers to a model.** Bins make the relationship monotone and easy to
  explain ("utilisation above 90 % costs you 60 points"), handle missing income as its own bin, and are what
  regulators and adverse-action notices expect.
- **Information value to choose characteristics.** IV measures how much a characteristic separates goods from
  bads; everything used is above 0.02, the usual floor, and the past-delinquency and utilisation characteristics are
  the strongest.
- **VIF check.** All variance inflation factors are below 1.25, so the seven characteristics are not saying the same
  thing twice and the coefficients are stable.
- **Logistic regression, not a tree model.** Coefficients become points; points become reasons for a decline. That
  explainability is the whole point of a scorecard.
- **Stratified 70/30 split.** Keeps the 6.7 % bad rate identical in train and test so the AUC comparison is fair.

## What I'd do differently

- Package the notebook as code with tests (done in `model-risk-monitoring`, which reuses this exact recipe).
- Use out-of-time validation instead of a random split; scorecards are judged on the next cohort, not the same one.
- Add reject inference: the data only contains approved borrowers, so the model has never seen a declined applicant.
- Check monotonicity of bad rate across bins programmatically and merge bins that break it.

## Five questions an interviewer would ask, with honest answers

**1. What does an AUC of 0.857 actually mean for the business?**
If you pick one random bad borrower and one random good one, the model ranks the bad one as riskier 85.7 % of the
time. In score terms, the bottom band (under 500) has a 54 % default rate and the top band under 1 %, so the score
lets the bank set a cut-off with a known expected loss.

**2. Train AUC 0.856 and test AUC 0.857 - is that suspicious?**
It means no overfitting, which is expected: a logistic regression on seven WOE features has very few parameters.
The same closeness would be a red flag for a 500-tree model.

**3. Why is the KS statistic 0.557 reported alongside AUC?**
KS is the biggest gap between the cumulative distributions of goods and bads across the score; 0.557 is a strong
scorecard (0.3 is a common minimum). KS is the number risk teams quote because it corresponds to the single best
cut-off; AUC summarises all cut-offs.

**4. How do points get calculated?**
Each characteristic's points for a bin = -(coefficient x WOE + intercept / 7) x factor + offset / 7, with factor =
20 / ln 2 = 28.85 and offset = 600 - 28.85 x ln 50 = 487.1. Summing the seven contributions gives the score. It's a
linear rescaling of the log-odds so that 600 corresponds to 50:1 odds and every 20 points doubles them.

**5. What is the biggest weakness of this model?**
It was built and evaluated on the same snapshot with a random split, and the population contains only accepted
applicants. Both make the real-world performance uncertain - which is what the monitoring repo addresses.

## Numbers to have in your head

150,000 rows · 6.68 % bad · 7 WOE characteristics · train / test AUC 0.856 / 0.857 (Gini 0.713) · KS 0.557 · VIF max 1.24 ·
600 points @ 50:1 odds, PDO 20 · score bands: <500 -> 54 % default, 500-550 -> 22 %, 600+ -> under 2 %.
