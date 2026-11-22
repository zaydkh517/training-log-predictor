# Training Log Progression Predictor

Upload your [Strong](https://www.strong.app/) or [Hevy](https://www.hevyapp.com/) workout export and see where your lifts are headed: strength trajectory per exercise, plateau detection, a 3-month outlook, and a session planner that recommends weights for your next workout.

**Live app: [training-log-predictor-red.vercel.app](https://training-log-predictor-red.vercel.app)**. No export handy? Hit **Try sample data**.

## What it does

- Merges Strong and Hevy CSV exports (lb or kg) into one training history (1,200+ sets of my own data during development), skipping cardio, warm-ups and bodyweight moves
- Estimates one-rep max (e1RM) per exercise per session from the strongest set using the Epley formula, smoothed with a 28-day rolling average. Sets over 12 reps only count when a session has nothing else, since Epley overestimates there
- Tracks machine switches: a session whose e1RM is under 60% or over ~167% of the recent level can't be the same machine (a different cable stack, another gym's leg extension), so it's tracked as its own series. Going back to an earlier machine picks up its history again. Trends, plateaus and outlooks use the current machine; training age counts all of them
- Classifies each lift as progressing, plateaued, declining or inactive over a fixed 28-day window. The plateau line scales with training age instead of being a flat percentage, so slow-but-real progress for an experienced lifter isn't flagged
- Projects a 3-month outlook using a damped trend that deliberately refuses to extrapolate past the data it was fit on, capped by experience-scaled growth limits from strength-training research
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

## Stack

| Piece | Tech |
|---|---|
| Backend | Python, FastAPI, pandas, scikit-learn — deployed on fly.io |
| Frontend | React, TypeScript, Vite, Recharts — deployed on Vercel |
| CI/CD | GitHub Actions: 52-test pytest suite runs on every push and gates the backend deploy |

## Design notes (honest limitations)

- The outlook range is a heuristic spread based on session-to-session variability, not a statistical confidence interval — the API response says so explicitly.
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
