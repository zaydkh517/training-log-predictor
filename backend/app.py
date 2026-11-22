from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from features import build_dataset
from predict import check_plateau, long_term_outlook
from rep_model import FEATURES as REP_MODEL_FEATURES, MAX_WEIGHT_CHANGE, RepModel, epley_reps, fit_personal

DEFAULT_REP_MODEL = RepModel.load()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://training-log-predictor-red.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_PLANNED_SETS = 10
MAX_TARGET_REPS = 30


class RepModelParams(BaseModel):
    """A rep model as returned by /analyze, sent back so planning uses the lifter's own fit."""
    mean: list[float] = Field(min_length=len(REP_MODEL_FEATURES), max_length=len(REP_MODEL_FEATURES))
    scale: list[float] = Field(min_length=len(REP_MODEL_FEATURES), max_length=len(REP_MODEL_FEATURES))
    coef: list[float] = Field(min_length=len(REP_MODEL_FEATURES), max_length=len(REP_MODEL_FEATURES))
    intercept: float
    source: str = 'personal'
    n_sets: int = Field(default=0, ge=0)
    features: list[str]
    _model: RepModel | None = PrivateAttr(default=None)

    @model_validator(mode='after')
    def check_compatible(self):
        self._model = RepModel.from_dict(self.model_dump())  # raises on a mismatched model
        return self

    def to_model(self):
        return self._model


def resolve_model(params):
    return params.to_model() if params is not None else DEFAULT_REP_MODEL


class PredictSetRequest(BaseModel):
    exercise: str
    weight: float = Field(gt=0)
    rolling_e1rm: float = Field(gt=0)
    set_number: int = Field(ge=1)
    prior_set_reps: float | None = Field(default=None, ge=0)
    prior_set_weight: float | None = Field(default=None, ge=0)
    rep_model: RepModelParams | None = None

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
    total_sets: int = Field(ge=1, le=MAX_PLANNED_SETS)
    first_set_weight: float = Field(gt=0)
    target_min_reps: int = Field(ge=1, le=MAX_TARGET_REPS)
    target_max_reps: int = Field(ge=1, le=MAX_TARGET_REPS)
    rep_model: RepModelParams | None = None

    @model_validator(mode='after')
    def check_rep_range(self):
        if self.target_min_reps > self.target_max_reps:
            raise ValueError('target_min_reps cannot be greater than target_max_reps')
        return self

SEARCH_BOUND_PCT = MAX_WEIGHT_CHANGE  # search only where the model was trained


def snap_to_plates(weight):
    """Round to the nearest 5 lb so plans use gym-loadable weights."""
    return 5 * round(weight / 5)


def weight_for_reps(rolling_e1rm, reps):
    """Epley inverted the other way: the weight that should allow `reps` fresh reps."""
    return rolling_e1rm / (1 + reps / 30)


def find_best_weight(prior_weight, prior_reps, rolling_e1rm, set_number, target_min, target_max,
                     model=DEFAULT_REP_MODEL):
    """Heaviest gym-loadable weight within 20% of the prior set that the model expects
    to land in the target range; otherwise the closest miss."""
    best_weight = None
    best_reps = None
    best_distance = None

    # stay within 20% of the prior set -- the model can't extrapolate past its data --
    # and never search above the lifter's estimated max
    min_weight = prior_weight * (1 - SEARCH_BOUND_PCT)
    candidate = min(prior_weight * (1 + SEARCH_BOUND_PCT), rolling_e1rm)

    snapped = snap_to_plates(candidate)
    if min_weight <= snapped <= candidate:
        candidate = snapped
    elif snapped - 5 >= min_weight:
        candidate = snapped - 5

    while candidate >= min_weight:
        predicted_reps = model.predict(candidate, set_number, prior_reps, prior_weight, rolling_e1rm)

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
    # fit the rep model to this lifter (falls back to the default on short histories)
    rep_model = fit_personal(dataset["usable_sets"], DEFAULT_REP_MODEL)

    results = {}
    for exercise in e1rm_df["exercise"].unique():
        history = e1rm_df[e1rm_df["exercise"] == exercise]
        trajectory = []
        for row in history.itertuples():
            trajectory.append({
                "date": row.date.strftime("%Y-%m-%d"),
                "e1rm": float(row.e1rm),
                "rolling_e1rm": float(row.rolling_e1rm),
                "machine": int(row.machine),
            })
        results[exercise] = {
            "trajectory": trajectory,
            "current_machine": int(history["machine"].iloc[-1]),
            "machine_count": int(history["machine"].nunique()),
            "outlook": long_term_outlook(e1rm_df, exercise),
            "plateau": check_plateau(e1rm_df, exercise),
        }

    return {"exercises": results, "rep_model": rep_model.to_dict()}

@app.post("/predict-set")
def predict_set(request: PredictSetRequest):
    if request.prior_set_reps is None:
        predicted_reps = epley_reps(request.weight, request.rolling_e1rm)
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
        model = resolve_model(request.rep_model)
        predicted_reps = model.predict(
            request.weight, request.set_number, request.prior_set_reps,
            request.prior_set_weight, request.rolling_e1rm,
        )
        return {
            "exercise": request.exercise,
            "predicted_reps": round(predicted_reps),
            "method": f"baseline_plus_correction ({model.source})",
        }


@app.post("/plan-session")
def plan_session(request: PlanSessionRequest):
    model = resolve_model(request.rep_model)
    sets = []

    first_reps = max(0, round(epley_reps(request.first_set_weight, request.rolling_e1rm)))

    if first_reps == 0:
        return {
            "exercise": request.exercise,
            "error": "starting weight is at or above your estimated one-rep max -- pick a lighter starting weight",
        }

    first_set = {"set_number": 1, "weight": request.first_set_weight, "predicted_reps": first_reps}
    if not (request.target_min_reps <= first_reps <= request.target_max_reps):
        mid_target = (request.target_min_reps + request.target_max_reps) / 2
        suggested = snap_to_plates(weight_for_reps(request.rolling_e1rm, mid_target))
        first_set["note"] = (
            f"this starting weight puts you outside your target range -- "
            f"try about {suggested} lb to start"
        )
    sets.append(first_set)

    prior_weight = request.first_set_weight
    prior_reps = first_reps

    for set_number in range(2, request.total_sets + 1):
        weight, reps = find_best_weight(
            prior_weight, prior_reps, request.rolling_e1rm, set_number,
            request.target_min_reps, request.target_max_reps, model,
        )

        if reps <= 0:
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

    return {"exercise": request.exercise, "sets": sets, "rep_model_source": model.source}
