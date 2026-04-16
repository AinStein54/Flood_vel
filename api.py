# api.py — FloodGuard Clean Pipeline API v7
# Matches flood_prediction_clean_pipeline.ipynb
# 3 classes: Low=0 / Medium=1 / High=2  |  Top-5 towns  |  No scaler needed

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib, json, os, math
from datetime import datetime

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

app = FastAPI(title="FloodGuard API", version="7.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE, "flood_model.pkl")
FEATURES_PATH = os.path.join(BASE, "feature_order.json")
META_PATH     = os.path.join(BASE, "model_metadata.json")
RISK_MAP_PATH = os.path.join(BASE, "risk_mapping.json")

model    = None
FEATURES = []
meta     = {}
RISK_MAP = {}

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model: {type(model).__name__}")
except Exception as e:
    print(f"❌ Model: {e}")

try:
    with open(FEATURES_PATH) as f: FEATURES = json.load(f)
    print(f"✅ Features: {len(FEATURES)}")
except Exception as e:
    print(f"❌ Features: {e}")

try:
    with open(META_PATH) as f: meta = json.load(f)
    print(f"✅ Metadata loaded — towns: {meta.get('towns')}")
except Exception as e:
    print(f"❌ Metadata: {e}")

try:
    with open(RISK_MAP_PATH) as f: RISK_MAP = json.load(f)
    print(f"✅ Risk mapping: {RISK_MAP}")
except Exception as e:
    print(f"❌ Risk mapping: {e}")

# ── Classes & towns (read from metadata, fallback to defaults) ────────────────
CLASS_NAMES  = meta.get('target_order', ['Low', 'Medium', 'High'])
TOWNS        = meta.get('towns', [])
SUIT_TYPES   = meta.get('suitability_types', ['County to Town', 'Town to Street'])

# ── Town coordinates ──────────────────────────────────────────────────────────
# Covers all towns that could appear as top-5 from the dataset
ALL_COORDS = {
    "Leeds":               {"lat": 53.8008, "lon": -1.5491},
    "Sheffield":           {"lat": 53.3811, "lon": -1.4701},
    "Bradford":            {"lat": 53.7938, "lon": -1.7529},
    "Doncaster":           {"lat": 53.5228, "lon": -1.1288},
    "Newcastle upon Tyne": {"lat": 54.9783, "lon": -1.6178},
    "York":                {"lat": 53.9590, "lon": -1.0815},
    "Rotherham":           {"lat": 53.4326, "lon": -1.3635},
    "Barnsley":            {"lat": 53.5527, "lon": -1.4797},
    "Middlesbrough":       {"lat": 54.5742, "lon": -1.2348},
    "Sunderland":          {"lat": 54.9069, "lon": -1.3838},
    "Huddersfield":        {"lat": 53.6450, "lon": -1.7798},
    "Wakefield":           {"lat": 53.6830, "lon": -1.4977},
    "Stockton-on-Tees":    {"lat": 54.5704, "lon": -1.3290},
    "Halifax":             {"lat": 53.7213, "lon": -1.8654},
    "Durham":              {"lat": 54.7761, "lon": -1.5733},
    "Keighley":            {"lat": 53.8678, "lon": -1.9061},
    "Darlington":          {"lat": 54.5237, "lon": -1.5535},
    "Grimsby":             {"lat": 53.5675, "lon": -0.0806},
    "Gateshead":           {"lat": 54.9526, "lon": -1.6030},
}
TOWN_COORDS = {t: ALL_COORDS[t] for t in TOWNS if t in ALL_COORDS}

# ── Feature display names ─────────────────────────────────────────────────────
DISPLAY = {
    "latitude":           "Latitude",
    "longitude":          "Longitude",
    "temp_max":           "Max Temperature (°C)",
    "temp_min":           "Min Temperature (°C)",
    "precipitation_sum":  "Total Precipitation (mm)",
    "wind_speed_max":     "Max Wind Speed (km/h)",
    "wind_gusts_max":     "Wind Gusts (km/h)",
    "humidity_mean":      "Relative Humidity (%)",
    "soil_moisture_mean": "Soil Moisture (m³/m³)",
}
for t in TOWNS:
    DISPLAY[f"town_{t}"] = f"Town: {t}"
for s in SUIT_TYPES:
    DISPLAY[f"suitability_{s}"] = f"Suitability: {s}"

# ── SHAP explanations (context text for each feature) ────────────────────────
CONTEXT = {
    "precipitation_sum":  ("total precipitation", "mm",    "more rain → higher runoff and river levels"),
    "humidity_mean":      ("relative humidity",   "%",     "high humidity = moisture-saturated atmosphere"),
    "soil_moisture_mean": ("soil moisture",        "m³/m³", "saturated soil cannot absorb more rainfall"),
    "wind_speed_max":     ("max wind speed",       "km/h",  "strong winds intensify storm conditions"),
    "wind_gusts_max":     ("wind gusts",           "km/h",  "severe gusts signal extreme storm activity"),
    "temp_max":           ("max temperature",      "°C",    "warm temps drive storm development"),
    "temp_min":           ("min temperature",      "°C",    "cold temps cause snowmelt and drainage issues"),
}

def plain_english(feat, sv, fv):
    ctx = CONTEXT.get(feat)
    if not ctx:
        return None
    label, unit, reason = ctx
    direction = "increases" if sv > 0 else "reduces"
    val_str   = f"{fv:.3f} ({fv*100:.0f}%)" if "soil_moisture" in feat else \
                (f"{fv:.1f} {unit}" if unit else f"{fv:.3f}")
    return f"{label.capitalize()} of {val_str} {direction} flood risk ({reason})"

# ── Build feature vector from request ────────────────────────────────────────
def build_features(day) -> pd.DataFrame:
    """Construct exactly the feature vector the model was trained on."""
    now     = datetime.utcnow()
    coords  = TOWN_COORDS.get(day.town, {"lat": 54.0, "lon": -1.5})

    # Start with all zeros for every feature the model expects
    row = {f: 0 for f in FEATURES}

    # Geographic
    row["latitude"]          = coords["lat"]
    row["longitude"]         = coords["lon"]

    # Weather (raw column names — no wx_ prefix in this pipeline)
    row["temp_max"]           = day.temp_max
    row["temp_min"]           = day.temp_min
    row["precipitation_sum"]  = day.precipitation_sum
    row["wind_speed_max"]     = day.wind_speed_max
    row["wind_gusts_max"]     = day.wind_gusts_max
    row["humidity_mean"]      = day.humidity_mean
    row["soil_moisture_mean"] = day.soil_moisture_mean

    # Suitability OHE — default to 'County to Town' (most common in training)
    suit_key = f"suitability_County to Town"
    if suit_key in row:
        row[suit_key] = 1

    # Town OHE
    town_key = f"town_{day.town}"
    if town_key in row:
        row[town_key] = 1

    return pd.DataFrame([row]).reindex(columns=FEATURES, fill_value=0)

def risk_label(p: float) -> str:
    return "High Risk"     if p >= 0.75 else \
           "Moderate Risk" if p >= 0.50 else \
           "Low Risk"      if p >= 0.25 else "Very Low Risk"

def risk_colour(p: float) -> str:
    return "#ef4444" if p >= 0.75 else \
           "#f59e0b" if p >= 0.50 else \
           "#3b82f6" if p >= 0.25 else "#22c55e"

# ── SHAP computation (3-tier: TreeExplainer → fallback → LOFO) ───────────────
def compute_shap(X_arr, X_df):
    preds = model.predict(X_arr)

    def _extract(sv, preds):
        """Normalise any SHAP output to 2-D (n_samples, n_features)."""
        if isinstance(sv, list):
            return np.array([sv[int(p)][i] for i, p in enumerate(preds)])
        if hasattr(sv, 'ndim') and sv.ndim == 3:
            return np.array([sv[i, :, int(p)] for i, p in enumerate(preds)])
        return sv

    if SHAP_AVAILABLE:
        try:
            expl = shap.TreeExplainer(model)
            sv   = _extract(expl.shap_values(X_df), preds)
            assert sv.ndim == 2 and sv.shape == (len(X_arr), len(FEATURES))
            return sv, "SHAP TreeExplainer"
        except Exception as e:
            print(f"TreeExplainer failed: {e}")

    # LOFO fallback — always works without SHAP library
    col_means = np.zeros(len(FEATURES))
    orig_prob  = model.predict_proba(X_arr)
    matrix     = np.zeros((len(X_arr), len(FEATURES)))
    for j in range(len(FEATURES)):
        Xm = X_arr.copy(); Xm[:, j] = col_means[j]
        mp = model.predict_proba(Xm)
        for i, p in enumerate(preds):
            matrix[i, j] = orig_prob[i, int(p)] - mp[i, int(p)]
    return matrix, "LOFO Proxy"

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ForecastDay(BaseModel):
    town:               str
    temp_max:           float
    temp_min:           float
    precipitation_sum:  float
    wind_speed_max:     float
    wind_gusts_max:     float
    humidity_mean:      float = 80.0
    soil_moisture_mean: float = 0.35
    pub_month:          Optional[int] = None

class PredictionRequest(BaseModel):
    days: List[ForecastDay]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":       "running",
        "model":        meta.get("model_name", type(model).__name__ if model else "not loaded"),
        "classes":      CLASS_NAMES,
        "n_features":   len(FEATURES),
        "towns":        TOWNS,
        "shap":         SHAP_AVAILABLE,
        "test_f1":      meta.get("test_f1_weighted"),
        "test_accuracy":meta.get("test_accuracy"),
    }

@app.get("/towns")
def get_towns():
    return {"towns": TOWNS, "coords": TOWN_COORDS}

@app.post("/predict")
def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        results = []
        for i, day in enumerate(req.days):
            df   = build_features(day)
            X    = df.values
            pred = int(model.predict(X)[0])
            prob = model.predict_proba(X)[0]

            # flood_prob = P(Medium) + P(High) — combined probability of significant flooding
            pm = float(prob[CLASS_NAMES.index("Medium")]) if "Medium" in CLASS_NAMES else 0.0
            ph = float(prob[CLASS_NAMES.index("High")])   if "High"   in CLASS_NAMES else 0.0
            fp = round(pm + ph, 4)

            results.append({
                "day":         i + 1,
                "town":        day.town,
                "prediction":  CLASS_NAMES[pred],
                "flood_prob":  fp,
                "all_probs":   {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, prob)},
                "risk_level":  risk_label(fp),
                "risk_colour": risk_colour(fp),
                "is_high_risk": fp >= 0.50,
            })
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/explain")
def explain(req: PredictionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        all_X, all_df, all_preds, all_probs = [], [], [], []
        for day in req.days:
            df   = build_features(day)
            X    = df.values
            pred = int(model.predict(X)[0])
            prob = model.predict_proba(X)[0]
            all_X.append(X[0]); all_df.append(df)
            all_preds.append(pred); all_probs.append(prob)

        X_arr   = np.array(all_X)
        X_full  = pd.concat(all_df, ignore_index=True)
        shap_m, method = compute_shap(X_arr, X_full)

        explanations = []
        for i, day in enumerate(req.days):
            row_sv  = shap_m[i]
            row_val = X_full.iloc[i]
            feats   = sorted([{
                "feature":      feat,
                "display_name": DISPLAY.get(feat, feat),
                "value":        round(float(row_val[feat]), 4),
                "shap_value":   round(float(row_sv[j]), 4),
                "abs_shap":     round(abs(float(row_sv[j])), 4),
                "direction":    "positive" if row_sv[j] > 0 else "negative",
                "explanation":  plain_english(feat, float(row_sv[j]), float(row_val[feat])),
            } for j, feat in enumerate(FEATURES)
              if not feat.startswith("town_")],   # hide town cols from API response too
            key=lambda x: x["abs_shap"], reverse=True)

            pm = float(all_probs[i][CLASS_NAMES.index("Medium")]) if "Medium" in CLASS_NAMES else 0
            ph = float(all_probs[i][CLASS_NAMES.index("High")])   if "High"   in CLASS_NAMES else 0

            explanations.append({
                "day":          i + 1,
                "town":         day.town,
                "prediction":   CLASS_NAMES[all_preds[i]],
                "flood_prob":   round(pm + ph, 4),
                "all_probs":    {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, all_probs[i])},
                "top_features": feats[:10],
                "bullets":      [f["explanation"] for f in feats if f["explanation"]][:5],
                "method":       method,
            })
        return {"method": method, "explanations": explanations}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=10000, reload=True)
