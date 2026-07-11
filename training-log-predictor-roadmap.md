# Training Log Progression Predictor — Build Roadmap

Locked scope: 3–4 weeks. Core model + RAG + cookie-based persistence. No exceptions.

## Scope (final)

**Building:** a trained regression model on your Strong training data, deployed as a Flask + React app, with a RAG-powered explanation layer and cookie-based multi-user persistence (no login).

**Not building:** full accounts/login/sessions, Redis, shareable links, anything UUID-facing to the user.

**Total estimate:** ~3–4 weeks at a few hours/day, given SWE experience but ML being newly learned.

## Day 0: Setup

- Python 3.10+, virtual environment
- `pip install pandas numpy scikit-learn xgboost flask pytest matplotlib`
- Export Strong data: Strong app → Settings → Export Data → CSV
- New GitHub repo, commit skeleton
- Create `setup.sh` in the project root, automating venv creation + dependency install (see example below), `chmod +x` it, and commit it to the repo — real dev-experience signal, not busywork


```

## Phase 1: Data Exploration (Days 1–2)

**Do:**
- Load the CSV into pandas
- Inspect columns (exercise, date, weight, reps, sets, RPE)
- Check for lift-name duplicates ("Squat" vs "Barbell Squat")
- Plot volume over time for 2–3 main lifts to see gaps/split changes

**Concepts:** `pandas.groupby()`, `pd.to_datetime()`, what missing data actually means in a training log.

**Pitfall:** decide explicitly whether to filter out warm-up sets — they'll distort intensity calculations if left in.

## Phase 2: Feature Engineering (Days 3–5) — the most important phase

Build, per lift per session:

1. e1RM via Epley formula: `e1RM = weight × (1 + reps / 30)`
2. Rolling trailing average (4-week window) of volume and e1RM — a window, not full history, because split changes make "average since I started" meaningless
3. Intensity trend — working weight as % of current e1RM
4. Weeks-since-program-change — manually tagged

**Pitfall — data leakage:** rolling features must only use data before the prediction date, never after. Verify this manually on one row before trusting it.

## Phase 3: Population Baseline Model (Days 6–7)

Required (needed for Phase 10's multi-user blending, not optional this time): train a Ridge regression on the public OpenPowerlifting dataset to learn general age/bodyweight/experience progression curves. Your personal model becomes a deviation from this baseline.

## Phase 4: Model Training + Evaluation (Days 7–9)

**Do:**
- Train Ridge and XGBoost, compare
- Time-based train/test split only — never random, or the model cheats by seeing the future
- Metrics: MAE, RMSE
- Sanity-check against a lift you know well

**Concepts:** bias-variance tradeoff; with ~50–150 data points per lift, watch for XGBoost overfitting (compare train vs. test error gap).

## Phase 5: Flask API (Days 10–11)

- `POST /predict` — CSV upload or JSON of recent sessions
- Model persistence via `joblib`
- Return predicted e1RM trajectory + plateau flag as JSON
- Design rule: one shared feature-engineering function used by both training and inference — never duplicate this logic

## Phase 6: CI/CD (Days 11–12)

- pytest: e1RM correctness, feature output shape, API response schema
- GitHub Actions workflow running pytest on every push
- CD is already free via Railway/Vercel auto-deploy

## Phase 7: React Dashboard (Days 12–14)

- Recharts: actual vs. predicted e1RM over time
- Plateau flag indicator
- CSV upload UI

## Phase 8: Deployment (Days 14–15)

- Backend → Railway, frontend → Vercel
- CORS setup, environment variables

## Phase 9: RAG Explanation Layer (Days 15–19)

- Build a small knowledge base (progressive overload, deload principles, plateau causes)
- Embed chunks with `sentence-transformers`, store in Chroma
- On a plateau flag: retrieve relevant chunks, feed them + the model's numbers into an OpenAI call for a grounded explanation
- New Flask field + new React section — additive, doesn't touch Phases 1–8

## Phase 10: Multi-User + Cookie Persistence (Days 19–22)

- Generalize CSV parsing to tolerate other users' export quirks (lift naming, missing RPE, short histories)
- Blend the Phase 3 population-prior model with a stranger's short personal history
- On upload: backend creates a database row for that user's data and sets a cookie in their browser identifying it — no visible ID, no link, the site just recognizes them on return visits (same browser/device)
- Native logging form for ongoing use: exercise, weight, reps, sets, RPE, date — appends to their stored data via the cookie, recomputes rolling features, refreshes predictions. CSV import is only for the initial historical backfill.
- Trade-off: tied to browser/device; clearing cookies loses access. Accepted for this scope.
- Not building: login, passwords, sessions, Redis.

## Timeline

Day 0 + Phases 1–10 ≈ 22 working days at a few hours/day ≈ 3–4 calendar weeks. Checkpoint: if Phases 1–8 aren't done by end of week 3, that's a signal to move faster on Phases 9–10, not to cut them — this scope is locked.

## How we'll work through this together

Concept before code, every phase. Pitfalls get flagged before you hit them. End of each phase: you should be able to explain the core idea out loud in under a minute — if not, slow down before moving on.

---

## Progress Log

**Day 0 (started Sunday, July 5, 2026):** Complete — venv created, packages installed, GitHub repo created and pushed, Strong + Hevy CSVs exported. `setup.sh` still outstanding (discussed, not yet recreated).

**Phase 1 (Days 1–2):** Complete — Strong + Hevy data merged into one schema (`date`, `exercise`, `weight`, `reps`, `set_order`, `source`), date parsing fixed for both apps' differing formats, exercise-name mismatches reconciled (Squat mislabel fix, Hevy→Strong naming map), 1228 total rows confirmed. RPE dropped entirely (100% empty in both sources). Moved exploration work from `explore.py` into `explore.ipynb` (Jupyter notebook, running via VS Code).

**Phase 2 (Days 3–5) — in progress, 3 of 4 features done:**
- e1RM via Epley formula — done, manually verified
- Rolling 4-week trailing average of e1RM — done, manually verified (via independent filter + `.mean()` check)
- Intensity trend (weight ÷ rolling e1RM) — done, manually verified
- Weeks-since-program-change — not started; known program change is Arnold split → PPL, exact date to be pinned down next session

Bodyweight exercises with no logged weight (Pull Up, Leg Raise Parallel Bars — merged into one canonical name) excluded from `modeling_df`/`e1rm_df`, kept in `combined` for completeness. Heaviest set per session selected via sort + `drop_duplicates` (tie-break: highest reps wins on equal weight).

**Phase 3 (Population Baseline Model, Days 6–7) — started early, on Day 3, in progress:**
- Downloaded OpenPowerlifting bulk CSV (~3.96M rows), loaded into `population_baseline.ipynb` with `usecols` to limit memory usage
- Filtered to `Equipment == 'Raw'` (1.88M rows) to match the user's own unequipped training
- Dropped rows missing `Age` or `BodyweightKg` (core model inputs)
- Identified negative values in `Best3SquatKg`/`Best3BenchKg`/`Best3DeadliftKg` as OpenPowerlifting's encoding for failed attempts — to be filtered out per-lift, not globally
- Built a per-lift `experience` feature (nth successful recorded lift for that lifter, via `groupby('Name').cumcount()`), verified correct on sample data
- About to encode `Sex` ('M'/'F' → 0/1) once on `power_data` (not per-lift, to avoid repeating), then rebuild `squat_data` so the encoding carries over via `.copy()`

**Phase 3 — complete.** Trained a shared `train_lift_model()` function (Ridge, 80/20 random split, Age/BodyweightKg/Sex_encoded/experience as features) applied to Squat, Bench, and Deadlift independently. All three beat their naive (predict-the-mean) baseline by a consistent 43-46% reduction in MAE:
- Squat: naive MAE 48.65 → model MAE 27.24, RMSE 35.86
- Bench: naive MAE 38.25 → model MAE 20.50, RMSE 26.93
- Deadlift: naive MAE 50.47 → model MAE 28.92, RMSE 38.07

Data quality issues caught and handled along the way: negative lift values (failed-attempt encoding, filtered per-lift), a third `Sex` category (`'Mx'`) that silently produced `NaN` via `.map()` (dropped, 158 rows). `interview-talking-points.md` created to track design decisions/tradeoffs in interview-ready language — update this alongside every future phase.

**Not yet done:** Phase 4 onward (personal model training/eval, Flask API, CI/CD, React dashboard, deployment, RAG layer, multi-user persistence).

**Timeline note (as of Friday, July 10 / Day 5):** user is traveling for 6 days starting July 11, still working during that time but at a faster pace — no features being cut, just less time spent re-explaining concepts already demonstrated (pandas filtering, groupby, train_test_split, Restart Kernel and Run All discipline, etc.). Remember to `git pull` before starting and `git commit`/`git push` before ending each session if working from a different device while traveling.
