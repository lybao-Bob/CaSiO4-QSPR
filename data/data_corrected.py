# -*- coding: utf-8 -*-
"""
PMF anomaly detection and conservative correction pipeline (upgraded)
-------------------------------------------------------------------
Input:
    raw_smi_pmf.csv
Required columns:
    SMILES, PMF

New features:
1. Anomaly scatter plot and before/after correction comparison plot
2. RandomForest vs Huber dual-model comparison
3. Cross-validated (OOF) anomaly score calculation
4. Save:
   - raw_smi_pmf_corrected.csv
   - smi_pmf.csv   (SMILES, PMF_corrected)

Author note:
- Keep original PMF unchanged
- Only create corrected PMF in a new column
- Conservative correction to avoid erasing real structural effects
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.base import clone

from statsmodels.nonparametric.smoothers_lowess import lowess


# =========================================================
# 1. Basic configuration
# =========================================================

INPUT_FILE = "raw_smi_pmf.csv"
OUTPUT_FULL_FILE = "raw_smi_pmf_corrected.csv"
OUTPUT_MIN_FILE = "smi_pmf.csv"

PLOT_DIR = "pmf_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

SMILES_COL = "SMILES"
PMF_COL = "PMF"

# LOWESS smoothing fraction
LOWESS_FRAC = 0.30

# CV setting
N_SPLITS = 5
RANDOM_STATE = 42

# Robust z-score thresholds,值越小更多点被判异常 / 被修正
Z_MILD = 0.8 # 2.5
Z_MODERATE = 1.0 # 3.5
Z_STRONG = 1.2 # 4.5

# Partial shrinkage coefficients，缩系数 λ越小，异常点被“拉回”更明显，corrected = λ * 原始值 + (1-λ) * 预测值
LAMBDA_MODERATE = 0.15 # 0.60
LAMBDA_STRONG = 0.10 # 0.30

# Whether to correct moderate anomalies
CORRECT_MODERATE = True # moderate 只是 flag，不修正; 为 True 时 moderate 也参与修正

# Which final anomaly score to use:
# "ensemble_mean" or "rf" or "huber"
FINAL_SCORE_MODE = "ensemble_mean"


# =========================================================
# 2. Molecular descriptor calculation
# =========================================================

def calc_descriptors(smiles: str):
    result = {
        "MolValid": 0,
        "MW": np.nan,
        "TPSA": np.nan,
        "HBD": np.nan,
        "HBA": np.nan,
        "AromaticRingCount": np.nan,
        "RotatableBonds": np.nan,
        "LogP": np.nan
    }

    if pd.isna(smiles):
        return result

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return result

    result["MolValid"] = 1
    result["MW"] = Descriptors.MolWt(mol)
    result["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
    result["HBD"] = Lipinski.NumHDonors(mol)
    result["HBA"] = Lipinski.NumHAcceptors(mol)
    result["AromaticRingCount"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    result["RotatableBonds"] = Lipinski.NumRotatableBonds(mol)
    result["LogP"] = Crippen.MolLogP(mol)

    return result


# =========================================================
# 3. LOWESS fit: PMF ~ MW
# =========================================================

def fit_lowess_predict(x, y, frac=0.30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fitted = lowess(endog=y, exog=x, frac=frac, return_sorted=True)
    x_fit = fitted[:, 0]
    y_fit = fitted[:, 1]

    x_unique, unique_idx = np.unique(x_fit, return_index=True)
    y_unique = y_fit[unique_idx]

    y_pred = np.interp(x, x_unique, y_unique, left=y_unique[0], right=y_unique[-1])
    return y_pred


def lowess_curve_for_plot(x, y, frac=0.30, n_points=200):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fitted = lowess(endog=y, exog=x, frac=frac, return_sorted=True)
    x_fit = fitted[:, 0]
    y_fit = fitted[:, 1]

    x_unique, unique_idx = np.unique(x_fit, return_index=True)
    y_unique = y_fit[unique_idx]

    x_grid = np.linspace(np.min(x_unique), np.max(x_unique), n_points)
    y_grid = np.interp(x_grid, x_unique, y_unique)
    return x_grid, y_grid


# =========================================================
# 4. Robust z-score
# =========================================================

def robust_zscore(values):
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))

    if np.isclose(mad, 0.0):
        return np.zeros_like(values), med, mad

    z = (values - med) / (1.4826 * mad)
    return z, med, mad


# =========================================================
# 5. Residual models
# =========================================================

def build_rf_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        ))
    ])


def build_huber_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor())
    ])


def get_oof_predictions(model, X, y, n_splits=5, random_state=42):
    """
    Out-of-fold predictions for more reliable anomaly scoring.
    """
    X = X.copy()
    y = np.asarray(y, dtype=float)

    oof_pred = np.full(shape=len(X), fill_value=np.nan, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y[train_idx]

        m = clone(model)
        m.fit(X_train, y_train)
        oof_pred[valid_idx] = m.predict(X_valid)

    final_model = clone(model)
    final_model.fit(X, y)

    return oof_pred, final_model


# =========================================================
# 6. Correction rules
# =========================================================

def classify_anomaly(abs_z):
    if abs_z >= Z_STRONG:
        return "strong"
    elif abs_z >= Z_MODERATE:
        return "moderate"
    elif abs_z >= Z_MILD:
        return "mild"
    else:
        return "normal"


def choose_lambda(abs_z):
    if abs_z >= Z_STRONG:
        return LAMBDA_STRONG
    elif abs_z >= Z_MODERATE:
        return LAMBDA_MODERATE
    else:
        return 1.0


# =========================================================
# 7. Plot functions
# =========================================================

def plot_pmf_vs_mw(df_model, save_path):
    x = df_model["MW"].values
    y = df_model["PMF"].values
    y_corr = df_model["PMF_corrected"].values
    z = df_model["abs_robust_z_final"].values

    x_grid, y_grid = lowess_curve_for_plot(x, y, frac=LOWESS_FRAC, n_points=200)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=z, s=55, alpha=0.85)
    plt.plot(x_grid, y_grid, linewidth=2, label="LOWESS: PMF ~ MW")
    plt.xlabel("MW")
    plt.ylabel("PMF")
    plt.title("PMF vs MW with anomaly intensity")
    cbar = plt.colorbar(sc)
    cbar.set_label("abs_robust_z_final")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # corrected overlay
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=45, alpha=0.70, label="Original PMF")
    plt.scatter(x, y_corr, s=45, alpha=0.70, label="Corrected PMF")
    plt.plot(x_grid, y_grid, linewidth=2, label="LOWESS baseline")
    plt.xlabel("MW")
    plt.ylabel("PMF")
    plt.title("Original vs corrected PMF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pmf_vs_mw_before_after_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_before_after_comparison(df_model, save_path):
    x = df_model["PMF"].values
    y = df_model["PMF_corrected"].values
    flags = df_model["correction_flag"].values

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=55, alpha=0.85)

    min_v = np.nanmin([x.min(), y.min()])
    max_v = np.nanmax([x.max(), y.max()])
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5, label="y = x")

    corrected_idx = np.where(pd.Series(flags).astype(str).str.contains("corrected"))[0]
    if len(corrected_idx) > 0:
        plt.scatter(x[corrected_idx], y[corrected_idx], s=70, alpha=0.95, label="Corrected samples")

    plt.xlabel("Original PMF")
    plt.ylabel("Corrected PMF")
    plt.title("Before vs after correction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dual_model_comparison(df_model, save_path):
    x = df_model["anomaly_score_rf"].values
    y = df_model["anomaly_score_huber"].values

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=55, alpha=0.85)

    min_v = np.nanmin([x.min(), y.min()])
    max_v = np.nanmax([x.max(), y.max()])
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5, label="y = x")

    plt.xlabel("Anomaly score (RF, OOF)")
    plt.ylabel("Anomaly score (Huber, OOF)")
    plt.title("Dual-model anomaly score comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_distribution(df_model, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(df_model["robust_z_rf"].dropna(), bins=20, alpha=0.7, label="RF robust z")
    plt.hist(df_model["robust_z_huber"].dropna(), bins=20, alpha=0.7, label="Huber robust z")
    plt.hist(df_model["robust_z_final"].dropna(), bins=20, alpha=0.7, label="Final robust z")
    plt.xlabel("Robust z-score")
    plt.ylabel("Count")
    plt.title("Distribution of anomaly robust z-scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 8. Main procedure
# =========================================================

def main():
    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    if SMILES_COL not in df.columns or PMF_COL not in df.columns:
        raise ValueError(f"Input file must contain columns: {SMILES_COL}, {PMF_COL}")

    df = df.copy()
    df["row_id"] = np.arange(len(df))

    # -----------------------------------------------------
    # Calculate descriptors
    # -----------------------------------------------------
    desc_df = df[SMILES_COL].apply(calc_descriptors).apply(pd.Series)
    df = pd.concat([df, desc_df], axis=1)

    # -----------------------------------------------------
    # Basic cleaning
    # -----------------------------------------------------
    df["PMF"] = pd.to_numeric(df[PMF_COL], errors="coerce")

    modeling_mask = (
        (df["MolValid"] == 1) &
        df["PMF"].notna() &
        df["MW"].notna()
    )

    df["used_for_model"] = modeling_mask.astype(int)

    if modeling_mask.sum() < 20:
        raise ValueError("Too few valid samples for modeling after SMILES/PMF cleaning.")

    model_df = df.loc[modeling_mask].copy()

    # -----------------------------------------------------
    # First stage: LOWESS PMF ~ MW
    # -----------------------------------------------------
    model_df["PMF_hat_MW"] = fit_lowess_predict(
        x=model_df["MW"].values,
        y=model_df["PMF"].values,
        frac=LOWESS_FRAC
    )
    model_df["residual_MW"] = model_df["PMF"] - model_df["PMF_hat_MW"]

    # -----------------------------------------------------
    # Second stage: OOF residual modeling
    # -----------------------------------------------------
    feature_cols = [
        "TPSA",
        "HBD",
        "HBA",
        "AromaticRingCount",
        "RotatableBonds",
        "LogP"
    ]

    X = model_df[feature_cols].copy()
    y = model_df["residual_MW"].values

    rf_model = build_rf_model()
    huber_model = build_huber_model()

    # Out-of-fold predictions
    oof_rf, final_rf_model = get_oof_predictions(
        model=rf_model, X=X, y=y, n_splits=N_SPLITS, random_state=RANDOM_STATE
    )
    oof_huber, final_huber_model = get_oof_predictions(
        model=huber_model, X=X, y=y, n_splits=N_SPLITS, random_state=RANDOM_STATE
    )

    model_df["residual_hat_struct_rf"] = oof_rf
    model_df["residual_hat_struct_huber"] = oof_huber

    model_df["anomaly_score_rf"] = model_df["residual_MW"] - model_df["residual_hat_struct_rf"]
    model_df["anomaly_score_huber"] = model_df["residual_MW"] - model_df["residual_hat_struct_huber"]

    # robust z for each model
    z_rf, med_rf, mad_rf = robust_zscore(model_df["anomaly_score_rf"].values)
    z_huber, med_huber, mad_huber = robust_zscore(model_df["anomaly_score_huber"].values)

    model_df["robust_z_rf"] = z_rf
    model_df["abs_robust_z_rf"] = np.abs(z_rf)

    model_df["robust_z_huber"] = z_huber
    model_df["abs_robust_z_huber"] = np.abs(z_huber)

    # -----------------------------------------------------
    # Final ensemble anomaly score
    # -----------------------------------------------------
    if FINAL_SCORE_MODE == "rf":
        model_df["anomaly_score_final"] = model_df["anomaly_score_rf"]
        model_df["residual_hat_struct_final"] = model_df["residual_hat_struct_rf"]
    elif FINAL_SCORE_MODE == "huber":
        model_df["anomaly_score_final"] = model_df["anomaly_score_huber"]
        model_df["residual_hat_struct_final"] = model_df["residual_hat_struct_huber"]
    else:
        # ensemble mean
        model_df["anomaly_score_final"] = (
            model_df["anomaly_score_rf"] + model_df["anomaly_score_huber"]
        ) / 2.0
        model_df["residual_hat_struct_final"] = (
            model_df["residual_hat_struct_rf"] + model_df["residual_hat_struct_huber"]
        ) / 2.0

    z_final, med_final, mad_final = robust_zscore(model_df["anomaly_score_final"].values)
    model_df["robust_z_final"] = z_final
    model_df["abs_robust_z_final"] = np.abs(z_final)

    model_df["anomaly_level"] = model_df["abs_robust_z_final"].apply(classify_anomaly)

    # Reasonable target
    model_df["PMF_target_reasonable"] = (
        model_df["PMF_hat_MW"] + model_df["residual_hat_struct_final"]
    )

    model_df["lambda_used"] = model_df["abs_robust_z_final"].apply(choose_lambda)

    # -----------------------------------------------------
    # Correction
    # -----------------------------------------------------
    def apply_correction(row):
        pmf = row["PMF"]
        target = row["PMF_target_reasonable"]
        level = row["anomaly_level"]
        lam = row["lambda_used"]

        if level == "strong":
            return lam * pmf + (1.0 - lam) * target
        elif level == "moderate" and CORRECT_MODERATE:
            return lam * pmf + (1.0 - lam) * target
        else:
            return pmf

    model_df["PMF_corrected"] = model_df.apply(apply_correction, axis=1)

    def correction_flag(row):
        if row["anomaly_level"] == "strong":
            return "corrected_strong"
        elif row["anomaly_level"] == "moderate" and CORRECT_MODERATE:
            return "corrected_moderate"
        elif row["anomaly_level"] in ["moderate", "mild"]:
            return "flag_only"
        else:
            return "unchanged"

    model_df["correction_flag"] = model_df.apply(correction_flag, axis=1)
    model_df["PMF_correction_delta"] = model_df["PMF_corrected"] - model_df["PMF"]

    # -----------------------------------------------------
    # Optional: fit final full-data models for reference
    # -----------------------------------------------------
    final_rf_model.fit(X, y)
    final_huber_model.fit(X, y)

    model_df["residual_hat_struct_rf_fullfit"] = final_rf_model.predict(X)
    model_df["residual_hat_struct_huber_fullfit"] = final_huber_model.predict(X)

    # -----------------------------------------------------
    # Merge back
    # -----------------------------------------------------
    merge_cols = [
        "row_id",
        "PMF_hat_MW",
        "residual_MW",

        "residual_hat_struct_rf",
        "residual_hat_struct_huber",
        "residual_hat_struct_final",

        "anomaly_score_rf",
        "anomaly_score_huber",
        "anomaly_score_final",

        "robust_z_rf",
        "abs_robust_z_rf",
        "robust_z_huber",
        "abs_robust_z_huber",
        "robust_z_final",
        "abs_robust_z_final",

        "anomaly_level",
        "PMF_target_reasonable",
        "lambda_used",
        "PMF_corrected",
        "PMF_correction_delta",
        "correction_flag",

        "residual_hat_struct_rf_fullfit",
        "residual_hat_struct_huber_fullfit"
    ]

    df = df.merge(model_df[merge_cols], on="row_id", how="left")

    # For rows not modeled, keep corrected PMF as original PMF if PMF exists
    df["PMF_corrected"] = df["PMF_corrected"].fillna(df["PMF"])

    # -----------------------------------------------------
    # Save full output
    # -----------------------------------------------------
    output_cols = [
        "row_id",
        SMILES_COL,
        PMF_COL,
        "PMF_corrected",
        "PMF_correction_delta",
        "correction_flag",
        "anomaly_level",

        "robust_z_rf",
        "abs_robust_z_rf",
        "robust_z_huber",
        "abs_robust_z_huber",
        "robust_z_final",
        "abs_robust_z_final",

        "anomaly_score_rf",
        "anomaly_score_huber",
        "anomaly_score_final",

        "PMF_hat_MW",
        "residual_MW",
        "residual_hat_struct_rf",
        "residual_hat_struct_huber",
        "residual_hat_struct_final",
        "PMF_target_reasonable",
        "lambda_used",

        "MolValid",
        "MW",
        "TPSA",
        "HBD",
        "HBA",
        "AromaticRingCount",
        "RotatableBonds",
        "LogP",
        "used_for_model"
    ]

    existing_output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[existing_output_cols].copy()
    df_out.to_csv(OUTPUT_FULL_FILE, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # Save minimal output
    # -----------------------------------------------------
    df_min = df[[SMILES_COL, "PMF_corrected"]].copy()
    df_min.to_csv(OUTPUT_MIN_FILE, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # Save plots
    # -----------------------------------------------------
    plot_pmf_vs_mw(model_df, os.path.join(PLOT_DIR, "pmf_vs_mw_anomaly_scatter.png"))
    plot_before_after_comparison(model_df, os.path.join(PLOT_DIR, "pmf_before_after_comparison.png"))
    plot_dual_model_comparison(model_df, os.path.join(PLOT_DIR, "rf_vs_huber_anomaly_score.png"))
    plot_residual_distribution(model_df, os.path.join(PLOT_DIR, "robust_z_distribution.png"))

    # -----------------------------------------------------
    # Summary table for dual-model comparison
    # -----------------------------------------------------
    summary_rows = []

    for model_name, z_col in [
        ("rf", "abs_robust_z_rf"),
        ("huber", "abs_robust_z_huber"),
        ("final", "abs_robust_z_final")
    ]:
        abs_z = model_df[z_col]
        summary_rows.append({
            "model": model_name,
            "n_mild_or_above": int((abs_z >= Z_MILD).sum()),
            "n_moderate_or_above": int((abs_z >= Z_MODERATE).sum()),
            "n_strong": int((abs_z >= Z_STRONG).sum()),
            "max_abs_robust_z": float(np.nanmax(abs_z)),
            "median_abs_robust_z": float(np.nanmedian(abs_z))
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("model_comparison_summary.csv", index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------
    n_total = len(df)
    n_valid = int(df["used_for_model"].sum())
    n_mild = int((df["anomaly_level"] == "mild").sum())
    n_mod = int((df["anomaly_level"] == "moderate").sum())
    n_strong = int((df["anomaly_level"] == "strong").sum())
    n_corrected = int(df["correction_flag"].astype(str).str.contains("corrected").sum())

    print("=" * 70)
    print("PMF anomaly detection and correction finished")
    print("=" * 70)
    print(f"Total samples                        : {n_total}")
    print(f"Valid samples for modeling           : {n_valid}")
    print(f"Mild anomalies                       : {n_mild}")
    print(f"Moderate anomalies                   : {n_mod}")
    print(f"Strong anomalies                     : {n_strong}")
    print(f"Corrected samples                    : {n_corrected}")
    print(f"Full output                          : {OUTPUT_FULL_FILE}")
    print(f"Minimal output                       : {OUTPUT_MIN_FILE}")
    print(f"Model comparison summary             : model_comparison_summary.csv")
    print(f"Plots directory                      : {PLOT_DIR}")
    print("=" * 70)

    print("\nTop suspicious samples by final score:")
    suspicious = df_out.sort_values("abs_robust_z_final", ascending=False).head(10)
    print(suspicious[
        ["row_id", SMILES_COL, PMF_COL, "PMF_corrected", "abs_robust_z_final", "anomaly_level", "correction_flag"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()