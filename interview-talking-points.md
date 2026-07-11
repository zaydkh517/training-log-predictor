# Interview Talking Points — Training Log Progression Predictor

Living document. Add to this every time a real design decision gets made. Each entry: what I did, why, what I gave up, how I'd say it out loud.

---

## Data merging (Strong + Hevy → one schema)

**What:** Combined two different apps' CSV exports into one common schema (`date`, `exercise`, `weight`, `reps`, `set_order`, `source`).

**Why:** Different apps = different column names, different date formats, different exercise naming conventions for the same real movement (e.g. "Chest Fly (Machine)" in Hevy vs "Pec Deck (Machine)" in Strong). Without reconciling this, the same real exercise would look like two different exercises to any downstream code, silently corrupting every feature built on top of it.

**Tradeoff:** Manual mapping dictionaries (`HEVY_TO_CANONICAL`) don't scale — if a third app got added, or naming changed, this breaks silently unless someone remembers to update it. A more scalable version would use fuzzy string matching (e.g. Levenshtein distance) to auto-suggest matches, with a human confirming — more engineering effort than was justified for a 2-source, ~1200-row personal dataset.

**How to say it:** "I built a normalization layer that reconciles schema and naming differences between data sources before any feature engineering touches the data — this is the same problem you'd hit merging data from multiple third-party APIs in production, just at small scale."

---

## Exclude bodyweight exercises (Pull Up, Leg Raise Parallel Bars) from modeling

**What:** Kept them in the full combined dataset for completeness, but excluded them from the feature-engineering/modeling pipeline (`modeling_df`).

**Why:** e1RM formula requires a `weight` value; bodyweight exercises have none (NaN). Rather than imputing a fake weight (which would be fabricated data) or dropping the exercise from the record entirely (losing information), the decision was to keep the raw log complete but scope the *model* to only exercises where the input actually exists.

**Tradeoff:** This means the model has zero predictive coverage for bodyweight movements — a real limitation, documented rather than silently ignored. A more complete system would track bodyweight-exercise progression via reps-at-bodyweight as a separate metric entirely, which was out of scope given the locked timeline.

**How to say it:** "I made an explicit, documented decision to scope the model to exercises with the data quality needed to support it, rather than force a broken feature onto data that can't support it."

---

## Heaviest-set-per-session (tie-broken by reps)

**What:** For sessions with multiple sets of the same lift, kept only the heaviest set, and reps as an explicit tie-breaker when the same weight appeared in multiple sets.

**Why:** Multiple sets per session mean multiple weight/rep combos on the same day for the same lift — e1RM needs one representative number per session. Heaviest set was chosen (over first set, or an average) as the best signal of that day's true strength capacity. On a tie in weight, more reps is a strictly better proxy for capacity (Epley's formula itself agrees — more reps at equal weight = higher e1RM), so that became the tie-break rule instead of an arbitrary "first logged" default.

**Tradeoff:** Initially implemented with `.groupby().idxmax()`, which only tie-breaks on one column — a real bug caught via manual verification, not by the code failing. Fixed by switching to `.sort_values()` + `.drop_duplicates(keep='last')`, sorting on weight then reps so the correct row survives deterministically.

**How to say it:** "I don't trust code just because it runs without error — I manually verify feature-engineering logic against raw data on at least one example before trusting it downstream. That's exactly how I caught a tie-breaking bug that would've silently picked the wrong set in ambiguous cases."

---

## Rolling 4-week average, not full-history average — and the leakage risk

**What:** Rolling trailing average of e1RM, using a 28-day time-based window (`groupby('exercise').rolling('28D', on='date')`), not a running average since day one.

**Why:** A full-history average gets diluted by outdated data once training splits/rep ranges/programs change — "average since I started lifting" stops reflecting current trend. A trailing window stays responsive to recent performance.

**The leakage risk, explicitly:** a rolling window MUST only look backward from the prediction date, never forward — otherwise the model is training on information it wouldn't actually have at prediction time in a real deployment, producing misleadingly good offline metrics that don't hold up in production. Time-based `.rolling()` in pandas enforces this by construction (trailing by definition), but this was still manually verified against raw data on a specific example before being trusted.

**Tradeoff:** 28 days is a fixed, somewhat arbitrary window — no principled tuning was done to pick that number over, say, 21 or 35 days, given the scope/timeline. A more rigorous version would treat window size as a hyperparameter and tune it against validation performance.

**How to say it:** "Data leakage is the single most common way an ML pipeline looks great offline and fails in production. I specifically used a time-based rolling window, which enforces the 'no future data' constraint structurally rather than relying on discipline, and I verified it by hand on a real example rather than trusting that the code ran without erroring."

---

## Population baseline model (OpenPowerlifting) — why it exists at all

**What:** Training a separate Ridge regression per lift (Squat done, Bench/Deadlift in progress) on ~1.6M rows of real competition data, using Age, Bodyweight, Sex, and a derived `experience` feature to predict competition lift numbers.

**Why:** A personal model trained only on ~9 months of one person's data (dozens of sessions per lift) has almost no ability to generalize — it's a small-sample, single-person time series. Training a population-level model first, on a much larger and broader dataset, lets the system learn general strength-progression relationships (how age/bodyweight/experience typically relate to lift capacity) before ever seeing the individual user's data. The personal model then predicts the *deviation* from this baseline, a smaller and more tractable prediction problem than learning progression from scratch on a handful of data points. This also directly enables the multi-user cold-start case in Phase 10 — a brand-new user with almost no history still gets a reasonable first prediction, blended from the population prior.

**Known limitation, explicitly acknowledged:** competitive powerlifters are a self-selected, non-random population — people who chose to compete, often with competition-specific technique training, skewing away from casual recreational lifters (like the model's actual target user). This is a real representativeness gap, not fully solved — mitigated somewhat by the fact that the personal model only needs the baseline to be a reasonable *starting prior*, not a perfect match, since the deviation-based design self-corrects using the user's actual data over time.

**How to say it:** "I made a deliberate choice to separate 'general population trend' from 'this specific person's trend' into two models, rather than one model trying to do both — this is a standard technique for handling small-sample personalization problems, sometimes called a hierarchical or empirical Bayes-style prior, even though I implemented a simplified version of it here given the timeline. I can also speak directly to the sampling bias in the training data and why it's an acceptable, documented tradeoff rather than something I ignored."

---

## Memory/scale handling on the OpenPowerlifting dataset

**What:** The raw CSV is ~809MB, ~3.96M rows, 41 columns. Used `pd.read_csv(..., usecols=[...])` to load only the 9 columns actually needed, instead of loading everything and dropping columns after.

**Why:** Loading all 41 columns (including large text fields like meet names, federation names, individual per-attempt weights) into memory just to discard 30+ of them afterward wastes both load time and peak RAM — `usecols` skips parsing those columns entirely at the CSV-reading stage, which is meaningfully cheaper than a `pd.read_csv()` + `.drop(columns=[...])` afterward, since the latter still pays the cost of parsing everything first.

**Tradeoff/scalability note:** even with `usecols`, this is still a full in-memory load (~1.6M-1.9M rows after filtering) — fine on a single developer machine for a project of this size, but wouldn't scale to a dataset an order of magnitude larger without moving to chunked reading (`chunksize=`) or a columnar format like Parquet, or pushing the filtering into a database query instead of doing it in pandas after a full load.

**How to say it:** "I was deliberate about not loading the full dataset naively — using `usecols` to only parse the columns I actually needed cut both memory footprint and load time. I know that approach has a ceiling; past a certain data size I'd move to chunked reads or a columnar format instead of a full in-memory pandas load."

---

## Data quality handling: negative values, missing categories

**What:** Discovered `Best3SquatKg`/`Best3BenchKg`/`Best3DeadliftKg` contained negative values (OpenPowerlifting's encoding for a failed/unsuccessful attempt), and that `Sex` had a third category (`'Mx'`) beyond the two the initial encoding accounted for, silently producing `NaN` via `.map()`.

**Why it matters:** Neither of these would have caused a crash — negative weights would have trained a model on nonsense data, and `NaN` from the unmapped `Sex` category would have either broken training outright or silently dropped rows in a way that's easy to miss. Both were caught by explicitly checking (`.min()` per column, `.isna().sum()`), not by the code failing loudly.

**How to say it:** "I don't assume a dataset is clean just because it loads without error. I actively probe for encoding quirks — sentinel values like negative numbers standing in for 'no data', or unexpected categories — before trusting any downstream model built on top of it."

---

## Train/test split strategy: random (population) vs. time-based (personal) — and why they differ

**What:** Used a random 80/20 split for the population baseline model (`train_test_split`), but the personal model (Phase 4, upcoming) is explicitly required to use a time-based split.

**Why they're different:** The population dataset is cross-sectional — each row is a different, independent lifter at a point in time, with no single continuous trajectory to protect. A random split is standard and appropriate. The personal model, in contrast, is a single continuous time series for one person — a random split would let the model train on sessions *after* the date it's supposedly predicting, which is a direct data leakage bug: the model would appear highly accurate offline while being fundamentally incapable of that same accuracy in real deployment, where the future genuinely isn't known yet.

**How to say it:** "The right train/test split strategy depends on whether your data is cross-sectional or a time series — I used a random split where independence between rows actually holds, and I'll use a strictly time-based split for the personal model, because treating a time series like independent samples is one of the most common and dangerous mistakes in applied ML."

---

## Establishing a baseline before trusting any metric

**What:** Before treating a Ridge model's MAE (27.24kg) and RMSE (35.86kg) as "good," computed a naive baseline — always predicting the training mean, regardless of input — and got MAE 48.65kg. The real model beats it by ~44%.

**Why:** A raw error number is meaningless without something to compare it to. Without the naive baseline, there's no way to know whether the model is actually using its features intelligently, or whether it's barely better (or even worse) than doing nothing at all.

**How to say it:** "I don't evaluate a model in isolation — I always compare against a naive baseline first, so I have proof the model is learning a real signal from the features, not just producing plausible-looking numbers."

---

## Refactoring duplicated training code into a shared function

**What:** After building the full train/evaluate pipeline once for Squat, wrapped it into a single `train_lift_model(data, target_column)` function instead of hand-copying the same ~15 lines for Bench and Deadlift.

**Why:** Copy-pasted logic means any future fix (a bug, a new feature, a different eval metric) has to be applied in three places and is easy to forget in one of them. A shared function has exactly one place to fix or extend. This is the same principle the roadmap requires for Phase 5 (one shared feature-engineering function for both training and inference) — applied a phase early, to the model-training step instead of just feature engineering.

**Result:** All three lift models show consistent, meaningful improvement over their naive baselines — Squat ~44%, Bench ~46%, Deadlift ~43% reduction in MAE. That consistency across three independent targets, using the same four features, is stronger evidence the features are genuinely predictive than a single lift's result would be alone.

**How to say it:** "Once I saw myself about to copy-paste the same training pipeline three times, I refactored it into a single function parameterized by target column — same 'don't repeat yourself' principle that applies to any codebase, not just ML pipelines. It also meant I got a consistency check for free: all three lifts showed a similar magnitude of improvement over baseline, which gave me more confidence the features are doing real work, not just fitting noise on one lift."
