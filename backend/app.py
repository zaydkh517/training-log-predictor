from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib

rep_model = joblib.load("rep_model.joblib")

from features import build_dataset
from predict import check_plateau, long_term_outlook

app = FastAPI()


class PredictSetRequest(BaseModel):
    exercise: str
    weight: float
    rolling_e1rm: float
    set_number: int
    prior_set_reps: float | None = None
    prior_set_weight: float | None = None


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(
    strong_file: UploadFile | None = File(default=None),
    hevy_file: UploadFile | None = File(default=None),
):
    dataset = build_dataset(
        strong_path=strong_file.file if strong_file else None,
        hevy_path=hevy_file.file if hevy_file else None,
    )
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
            "predicted_reps": round(predicted_reps, 1),
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
            "predicted_reps": round(predicted_reps, 1),
            "method": "ridge_model",
        }