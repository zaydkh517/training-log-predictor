# Training Log Progression Predictor

Upload your [Strong](https://www.strong.app/) or [Hevy](https://www.hevyapp.com/) workout export and see where your lifts are headed: strength trajectory per exercise, plateau detection, a 3-month outlook, and a session planner that recommends weights for your next workout.

**Live app: [training-log-predictor-red.vercel.app](https://training-log-predictor-red.vercel.app)**. No export handy? Hit **Try sample data**.

## What it does

- Merges Strong and Hevy CSV exports (lb or kg) into one training history (1,200+ sets of my own data during development), skipping cardio, warm-ups and bodyweight moves
- Estimates one-rep max (e1RM) per exercise per session from the strongest set using the Epley formula, smoothed with a 28-day rolling average. Sets over 12 reps only count when a session has nothing else, since Epley overestimates there
- Tracks machine switches: a session whose e1RM is under 60% or over ~167% of the recent level can't be the same machine (a different cable stack, another gym's leg extension), so it's tracked as its own series. Going back to an earlier machine picks up its history again. Trends, plateaus and outlooks use the current machine; training age counts all of them
- Classifies each lift as progressing, plateaued, declining or inactive over a fixed 28-day window. The plateau line scales with training age instead of being a flat percentage, so slow-but-real progress for an experienced lifter isn't flagged
- Forecasts every lift 3 months out with a model trained on 1.2M meet-to-meet changes from OpenPowerlifting, giving a median and an 80% prediction interval personally calibrated to each upload (see below)
- Plans multi-set sessions around a target rep range with a rep model that is personalized to each upload (see below). Predictions are capped at what you could do fresh at that weight, so a plan can never ramp past your estimated max

## The rep model

Predicting reps set by set has a physics baseline: the previous set implies a same-day e1RM, and the Epley formula carries it to the next set's weight. The model doesn't relearn that. A ridge regression learns only the gap between the baseline and what lifters actually do, from unitless features (set number, % of e1RM, weight change vs. the last set, last set's reps, and how the last set compared to the rolling e1RM). Because nothing is in absolute pounds, a curl and a squat share one model.

**Personalized per upload.** `/analyze` refits the correction on the uploader's own sets, ridge-shrunk toward the shipped default, so a short history stays close to the default and a long one earns its own fit. The model is a handful of coefficients serialized as JSON (no pickles), sent back with each planning request.

**Evaluation** (mean absolute error in reps, 2 lifters, ~1,300 sets; same held-out sets for every row):

| Predictor | Session CV | Time split | New lifter |
|---|---|---|---|
| Epley from rolling e1RM | 2.20 | 2.14 | 2.95 |
| Same reps as last set | 1.45 | 1.56 | 1.81 |
| Last set carried to new weight | 1.45 | 1.60 | 1.84 |
| ... minus median fatigue drop | 1.15 | 1.20 | 1.59 |
| Previous model (ridge on absolute weights) | 1.04 | 1.10 | 1.29 |
| **Baseline + learned correction** | **1.02** | **1.07** | **1.22** |

*Session CV* holds out whole workouts. *Time split* trains on each lifter's earliest 70% and tests on the rest. *New lifter* trains on everyone else and tests on the held-out person, which is the check that matters for a public app. The learned correction beats every baseline on all three, but its margin over the previous model is small (0.02 to 0.07 reps), so most of the gain over simple rules comes from the formula baseline and unitless features, not model capacity.

Everything is trained and scored within ±20% of the prior set's weight, the only range the planner searches (99% of logged sets). The other 1% are machine switches mid-exercise (245 lb then 85 lb for about the same reps on a different machine), where pounds aren't comparable. Rolling e1RM is also computed per machine, which made every predictor's inputs cleaner. With those included, the formula baseline predicted ~80-rep sets and the learned model fell below "same reps as last set". On the synthetic sample lifter, personalizing the default cut time-split error from 0.56 to 0.40 reps. Regenerate the table with `python train_rep_model.py`; extra lifters go in `backend/data/lifters/`, one export file per person (gitignored).

## The 3-month forecast

Gym logs from a few people can't teach a model how strength develops, but [OpenPowerlifting](https://www.openpowerlifting.org) can: millions of public-domain meet results with each lifter tracked across years. Each training example pairs a raw lifter's best squat, bench or deadlift at one meet with their best at the next meet 60-200 days later (1.2M pairs, 189k lifters).

The model is three gradient-boosted quantile regressors (10th/50th/90th percentile of log change). Its inputs are only things a workout log also has: horizon, training age (days since the lift was first logged), and the recent rate of change. It deliberately leaves out lift type, bodyweight and absolute weight, so the same model forecasts any exercise. Giving it lift type, absolute weight, bodyweight, age and sex as well only improved meet-data error from 4.28% to 4.12%, which wasn't worth asking users for.

| MAE, % of e1RM over the horizon | OpenPowerlifting: held-out lifters, 2023+ | My gym log: squat/bench/deadlift | My gym log: 18 other lifts |
|---|---|---|---|
| No change | 4.87 | 7.53 | 8.26 |
| Previous heuristic (damped trend + growth caps) | 5.05 | 6.74 | 7.96 |
| **Learned model (median)** | **4.28** | **6.41** | **7.30** |

The model beats the heuristic it replaced on meet data, and on lifts it never saw (accessories and machines in a gym log). One thing it learned surprised me: after a falling stretch it predicts a rebound. That's expected for noisy single-day meet maxes, but it held on my smoothed gym e1RM too. Lifts that had been falling gained 10.9% over the next ~3 months on average, and the model's error on them was 8.0% against 11.8% for "no change".

**Calibration is the honest weak spot.** On meet data the 80% interval covers 81% of outcomes. On my gym log it covered only 52%: meet lifters are more advanced, and accessory lifts move faster in percentage terms. So `/analyze` backtests the model on the uploader's own history, using only forecasts whose outcome is already logged, and widens the interval (never re-centers it) toward the width that would have covered 80% of their past. With *k* backtests it applies k/(k+20) of that correction. Simulated live on my log, where each forecast only sees outcomes known by its date, this lifted coverage to about 68% with no loss in accuracy. Re-centering on past errors made accuracy worse, because progress slows over time. Retrain and regenerate the table with `python train_forecast_model.py` after downloading the OpenPowerlifting CSV into `backend/data/opl/`.

## Stack

| Piece | Tech |
|---|---|
| Backend | Python, FastAPI, pandas, scikit-learn — deployed on fly.io |
| Frontend | React, TypeScript, Vite, Recharts — deployed on Vercel |
| CI/CD | GitHub Actions: 53-test pytest suite runs on every push and gates the backend deploy |

## Design notes (honest limitations)

- Meet maxes aren't gym e1RMs, and training age is measured from the first logged session (training before that is invisible). The personal interval widening exists because of this gap, and it still under-covers on my log (68% for a nominal 80%).
- Model evaluation splits by whole session, never by individual set, since sets from the same workout share a fatigue state and would leak across train/test.
- The default rep model is trained on two lifters. The new-lifter column is an average over just two held-out people, so treat it as a sanity check, not a benchmark.
- Strong exports without a weight-unit column are assumed to be in lb.
- The sample dataset is synthetic (generated, not real training data).

## Running locally

Backend (Python 3.14):

```
cd backend
pip install -r requirements-dev.txt
uvicorn app:app --reload --port 8000
```

Frontend:

```
cd frontend
npm install
npm run dev
```

Tests:

```
cd backend
pytest
```
