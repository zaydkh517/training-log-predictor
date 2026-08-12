from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import joblib

rep_model = joblib.load("rep_model.joblib")

from features import build_dataset
from predict import check_plateau, long_term_outlook

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://training-log-predictor-red.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictSetRequest(BaseModel):
    exercise: str
    weight: float = Field(gt=0)
    rolling_e1rm: float = Field(gt=0)
    set_number: int = Field(ge=1)
    prior_set_reps: float | None = Field(default=None, ge=0)
    prior_set_weight: float | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def check_prior_set_consistency(self):
        both_present = self.prior_set_reps is not None and self.prior_set_weight is not None
        both_absent = self.prior_set_reps is None and self.prior_set_weight is None

        if not (both_present or both_absent):
            raise ValueError('prior_set_reps and prior_set_weight must both be provided together, or both left blank')
        if self.set_number == 1 and both_present:
            raise ValueError('set_number 1 has no prior set -- leave prior_set_reps/prior_set_weight blank')
        if self.set_number >= 2 and not both_present:
            raise ValueError('set_number 2 or higher requires both prior_set_reps and prior_set_weight')
        return self

class PlanSessionRequest(BaseModel):
    exercise: str
    rolling_e1rm: float = Field(gt=0)
    total_sets: int = Field(ge=1)
    first_set_weight: float = Field(gt=0)
    target_min_reps: int = Field(ge=1)
    target_max_reps: int = Field(ge=1)

    @model_validator(mode='after')
    def check_rep_range(self):
        if self.target_min_reps > self.target_max_reps:
            raise ValueError('target_min_reps cannot be greater than target_max_reps')
        return self

SEARCH_BOUND_PCT = 0.20


def find_best_weight(prior_weight, prior_reps, rolling_e1rm, set_number, target_min, target_max):
    best_weight = None
    best_reps = None
    best_distance = None

    # stay within 20% of the prior set -- the model can't extrapolate past its data
    min_weight = prior_weight * (1 - SEARCH_BOUND_PCT)
    candidate = min(prior_weight + 10, prior_weight * (1 + SEARCH_BOUND_PCT))
    while candidate >= min_weight:
        features = [
            candidate,
            set_number,
            prior_reps,
            prior_weight,
            rolling_e1rm,
            candidate / rolling_e1rm,
        ]
        predicted_reps = rep_model.predict([features])[0]

        if target_min <= predicted_reps <= target_max:
            return candidate, round(predicted_reps)

        distance = min(abs(predicted_reps - target_min), abs(predicted_reps - target_max))
        if best_distance is None or distance < best_distance:
            best_weight, best_reps, best_distance = candidate, predicted_reps, distance

        candidate -= 5

    return best_weight, round(best_reps)

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(
    strong_file: UploadFile | None = File(default=None),
    hevy_file: UploadFile | None = File(default=None),
):
    try:
        dataset = build_dataset(
            strong_path=strong_file.file if strong_file else None,
            hevy_path=hevy_file.file if hevy_file else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    e1rm_df = dataset["e1rm_df"]

    results = {}
    for exercise in e1rm_df["exercise"].unique():
        history = e1rm_df[e1rm_df["exercise"] == exercise]
        trajectory = []
        for row in history.itertuples():
            trajectory.append({
                "date": row.date.strftime("%Y-%m-%d"),
                "e1rm": float(row.e1rm),
                "rolling_e1rm": float(row.rolling_e1rm),
            })
        results[exercise] = {
            "trajectory": trajectory,
            "outlook": long_term_outlook(e1rm_df, exercise),
            "plateau": check_plateau(e1rm_df, exercise),
        }

    return {"exercises": results}

@app.post("/predict-set")
def predict_set(request: PredictSetRequest):
    if request.prior_set_reps is None:
        predicted_reps = 30 * (request.rolling_e1rm / request.weight - 1)
        if predicted_reps <= 0:
            return {
                "exercise": request.exercise,
                "predicted_reps": 0,
                "method": "formula",
                "note": "planned weight is at or above your current estimated one-rep max",
            }
        return {
            "exercise": request.exercise,
            "predicted_reps": round(predicted_reps),
            "method": "formula",
        }
    else:
        features = [
            request.weight,
            request.set_number,
            request.prior_set_reps,
            request.prior_set_weight,
            request.rolling_e1rm,
            request.weight / request.rolling_e1rm,
        ]
        predicted_reps = rep_model.predict([features])[0]
        return {
            "exercise": request.exercise,
            "predicted_reps": round(predicted_reps),
            "method": "ridge_model",
        }
        
@app.post("/plan-session")
def plan_session(request: PlanSessionRequest):
    sets = []

    first_reps = 30 * (request.rolling_e1rm / request.first_set_weight - 1)
    first_reps = max(0, round(first_reps))

    if first_reps == 0:
        return {
            "exercise": request.exercise,
            "error": "starting weight is at or above your estimated one-rep max -- pick a lighter starting weight",
        }

    sets.append({"set_number": 1, "weight": request.first_set_weight, "predicted_reps": first_reps})

    prior_weight = request.first_set_weight
    prior_reps = first_reps

    for set_number in range(2, request.total_sets + 1):
        weight, reps = find_best_weight(
            prior_weight, prior_reps, request.rolling_e1rm, set_number,
            request.target_min_reps, request.target_max_reps,
        )

        if reps == 0:
            sets.append({
                "set_number": set_number,
                "note": "predicted to fail this set -- stopping the plan here",
            })
            break

        set_entry = {"set_number": set_number, "weight": weight, "predicted_reps": reps}
        if not (request.target_min_reps <= reps <= request.target_max_reps):
            set_entry["note"] = (
                "target rep range isn't reachable within 20% of the prior set's "
                "weight -- showing the closest supported option instead"
            )
        sets.append(set_entry)
        prior_weight, prior_reps = weight, reps

    return {"exercise": request.exercise, "sets": sets}
