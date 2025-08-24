#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# ---------- Model def (same as training) ----------
class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        x = self.model(x)
        hazards = torch.sigmoid(x)          # [B, 4]
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S

# ---------- Helpers ----------
def load_json_as_row(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def norm(v):
        if isinstance(v, str) and v.strip().lower() in {"x", "unknown"}:
            return np.nan
        return v
    return {k: norm(v) for k, v in data.items()}

def _norm_missing_to_nan(s: pd.Series) -> pd.Series:

    s = s.replace({pd.NA: np.nan, None: np.nan})

    s = s.apply(lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x)
    return s

def _to_object_str_preserving_nan(s: pd.Series) -> pd.Series:

    s = _norm_missing_to_nan(s)

    mask = s.isna()
    s = s.astype(object)
    s.loc[~mask] = s.loc[~mask].astype(str)
    return s

def align_types_to_preprocessor(df_one, preprocessor,
                                numerical_features, ordinal_features, categorical_features):
    df = df_one.copy()


    for col in numerical_features + ordinal_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    cat_pipeline = preprocessor.named_transformers_["cat"]
    encoder = cat_pipeline.named_steps["encoder"]

    def _is_np_int(x):   return isinstance(x, (np.integer,)) or type(x).__name__ in ("int8","int16","int32","int64","uint8","uint16","uint32","uint64")
    def _is_np_float(x): return isinstance(x, (np.floating,)) or type(x).__name__ in ("float16","float32","float64")

    for j, col in enumerate(categorical_features):
        if col not in df.columns:
            continue

        cats = list(encoder.categories_[j])
        filtered = [c for c in cats if not (isinstance(c, str) and c == "missing")]
        rep = filtered[0] if filtered else None

        if rep is None or isinstance(rep, str):

            df[col] = _to_object_str_preserving_nan(df[col])
        elif _is_np_int(rep) or isinstance(rep, int):

            s = pd.to_numeric(df[col], errors="coerce")
            s = _norm_missing_to_nan(s)
            df[col] = s.astype(object)
        elif _is_np_float(rep) or isinstance(rep, float):

            s = pd.to_numeric(df[col], errors="coerce")
            s = _norm_missing_to_nan(s)
            df[col] = s.astype(object)
        else:

            df[col] = _norm_missing_to_nan(df[col]).astype(object)

    return df

# ---------- Inference ----------
def clinical_main(json_name, model_dir):

    ordinal_features = ["primary_gleason", "secondary_gleason", "ISUP", "tertiary_gleason"]
    numerical_features = ["age_at_prostatectomy", "pre_operative_PSA"]
    categorical_features = [
        "pT_stage", "positive_lymph_nodes", "capsular_penetration",
        "positive_surgical_margins", "invasion_seminal_vesicles",
        "lymphovascular_invasion", "earlier_therapy"
    ]
    trained_feature_cols = numerical_features + ordinal_features + categorical_features


    preprocessor = joblib.load(model_dir / "clinical_preprocessor.joblib")


    raw = load_json_as_row(json_name)
    row_aligned = {col: raw.get(col, np.nan) for col in trained_feature_cols}
    if pd.isna(row_aligned["tertiary_gleason"]):
        row_aligned["tertiary_gleason"] = 2

    df_one = pd.DataFrame([row_aligned], columns=trained_feature_cols)


    df_one = align_types_to_preprocessor(
        df_one, preprocessor, numerical_features, ordinal_features, categorical_features
    )


    X = preprocessor.transform(df_one)  # numpy [1, D]
    input_dim = X.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPBinaryClassifier(input_dim)
    state = torch.load(model_dir / "056.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        hazards, S = model(x_tensor)               # [1, 4]
        survival = torch.sum(S, dim=1).item()


    hazards_np = hazards.squeeze(0).cpu().numpy()
    S_np = S.squeeze(0).cpu().numpy()

    print(f"Inference for: {os.path.basename(json_name)}")
    print(f"  Time to Recurrent: {survival:.6f}")
    print(f"  Hazards (per time bin): {hazards_np}")
    print(f"  Survival S(t): {S_np}")

    return survival


'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-sample clinical inference (type-aligned).")
    parser.add_argument("--json", default="/mnt/SSD3/xulin/phd_research/chimera/"
                                          "data/task1/clinical_data/1301.json", help="Path to single patient JSON")
    parser.add_argument("--preprocessor", default="clinical_preprocessor.joblib",
                        help="Path to saved ColumnTransformer joblib")
    parser.add_argument("--model", default="./models/056.pt", help="Path to trained model .pt")
    parser.add_argument("--out_csv", default="", help="Optional path to save a one-row CSV")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()
    main(args)
'''