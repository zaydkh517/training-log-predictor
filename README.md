# Training Log Progression Predictor

Upload your [Strong](https://www.strong.app/) or [Hevy](https://www.hevyapp.com/) workout export and see where your lifts are headed: strength trajectory per exercise, plateau detection, a 3-month outlook, and a session planner that recommends weights for your next workout.

**Live app: [training-log-predictor-red.vercel.app](https://training-log-predictor-red.vercel.app)**. No export handy? Hit **Try sample data**.

## What it does

- Merges Strong and Hevy CSV exports (lb or kg) into one training history (1,200+ sets of my own data during development), skipping cardio, warm-ups and bodyweight moves
- Estimates one-rep max (e1RM) per exercise per session from the strongest set using the Epley formula, smoothed with a 28-day rolling average. Sets over 12 reps only count when a session has nothing else, since Epley overestimates there
- Classifies each lift as progressing, plateaued, declining or inactive over a fixed 28-day window. The plateau line scales with training age instead of being a flat percentage, so slow-but-real progress for an experienced lifter isn't flagged
- Projects a 3-month outlook using a damped trend that deliberately refuses to extrapolate past the data it was fit on, capped by experience-scaled growth limits from strength-training research
- Predicts reps for a planned set with a Ridge regression model trained on set-level history, and plans multi-set sessions around a target rep range. Predictions are capped at what you could do fresh at that weight, so a plan can never ramp past your estimated max

## Stack

| Piece | Tech |
|---|---|
| Backend | Python, FastAPI, pandas, scikit-learn — deployed on fly.io |
| Frontend | React, TypeScript, Vite, Recharts — deployed on Vercel |
| CI/CD | GitHub Actions: 37-test pytest suite runs on every push and gates the backend deploy |

## Design notes (honest limitations)

- The outlook range is a heuristic spread based on session-to-session variability, not a statistical confidence interval — the API response says so explicitly.
- Model evaluation splits by whole session, never by individual set, since sets from the same workout share a fatigue state and would leak across train/test.
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
