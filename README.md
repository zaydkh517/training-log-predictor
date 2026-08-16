# Training Log Progression Predictor

Upload your [Strong](https://www.strong.app/) or [Hevy](https://www.hevyapp.com/) workout export and see where your lifts are headed: strength trajectory per exercise, plateau detection, a 3-month outlook, and a session planner that recommends weights for your next workout.

**Live app: [training-log-predictor-red.vercel.app](https://training-log-predictor-red.vercel.app)**

## What it does

- Merges Strong and Hevy CSV exports into one training history (1,200+ sets of my own data during development)
- Estimates one-rep max (e1RM) per exercise per session using the Epley formula, smoothed with a 28-day rolling average
- Flags plateaus when your rolling e1RM stalls over recent sessions
- Projects a 3-month outlook using a damped trend that deliberately refuses to extrapolate past the data it was fit on, capped by experience-scaled growth limits from strength-training research
- Predicts reps for a planned set with a Ridge regression model trained on set-level history, and plans multi-set sessions around a target rep range

## Stack

| Piece | Tech |
|---|---|
| Backend | Python, FastAPI, pandas, scikit-learn — deployed on fly.io |
| Frontend | React, TypeScript, Vite, Recharts — deployed on Vercel |
| CI/CD | GitHub Actions: 20-test pytest suite runs on every push and gates the backend deploy |

## Design notes (honest limitations)

- The rep model's fatigue-related coefficients point the "wrong" direction — more prior-set effort predicts *more* reps, not fewer. This is a day-level readiness confound in observational training data (good days inflate every set), not something more data or a fancier model fixes. The session planner handles it by bounding its weight search to within 20% of the prior set's weight, so the model is never asked to extrapolate outside the data it has seen.
- The outlook range is a heuristic spread based on session-to-session variability, not a statistical confidence interval — the API response says so explicitly.
- Model evaluation splits by whole session, never by individual set, since sets from the same workout share a fatigue state and would leak across train/test.

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
