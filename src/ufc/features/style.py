"""Style composite scores — hand-crafted z-score blends + PCA."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ufc.io import paths


_STYLE_DECAY_SUFFIX = "_decay"


def _z_wc(series: pd.Series, weight_class: pd.Series,
          train_mask: pd.Series | None = None) -> pd.Series:
    """Z-score within weight-class groups using TRAIN-fold statistics only.

    If train_mask is None, falls back to whole-sample stats (legacy behavior).
    """
    wc = weight_class.fillna("Unknown")
    if train_mask is None:
        train_mask = pd.Series(True, index=series.index)
    train_series = series[train_mask]
    train_wc = wc[train_mask]

    stats = train_series.groupby(train_wc).agg(["mean", "std"])
    mu_global = float(train_series.mean()) if len(train_series) else 0.0
    sig_global = float(train_series.std()) if len(train_series) and train_series.std() else 1.0

    mu = wc.map(stats["mean"]).fillna(mu_global)
    sig = wc.map(stats["std"]).replace(0, np.nan).fillna(sig_global)
    return ((series - mu) / sig).fillna(0.0)


def compute_style_scores(df: pd.DataFrame, train_mask: pd.Series | None = None) -> pd.DataFrame:
    """Add hand-crafted style composite scores using decay-suffix features."""
    out = df.copy()

    s = _STYLE_DECAY_SUFFIX
    wc = out.get("weight_class", pd.Series("Unknown", index=out.index))

    def _get(col, default=0.0):
        if col in out.columns:
            return out[col].fillna(0.0)
        return pd.Series(default, index=out.index)

    def _z(series):
        return _z_wc(series, wc, train_mask=train_mask)

    # ── Hand-crafted z-score composites (z-scored within weight class) ──
    slpm = _get(f"slpm{s}")
    distance_share = _get(f"distance_share{s}")
    kd_per_15 = _get(f"kd_per_15{s}")
    kd_against_per_15 = _get(f"kd_against_per_15{s}")
    str_def = _get(f"str_def{s}")
    td_per_15 = _get(f"td_per_15{s}")
    ctrl_pct = _get(f"ctrl_pct{s}")
    td_acc = _get(f"td_acc{s}")
    sub_att_per_15 = _get(f"sub_att_per_15{s}")
    ground_share = _get(f"ground_share_grp{s}", 0.0)
    sub_def = _get(f"sub_def{s}")
    clinch_share = _get(f"clinch_share{s}")
    vol_pm = _get(f"vol_attempted_pm{s}")
    td_attempted_per_15 = _get(f"td_attempted_per_15{s}")

    out["striker_score"] = (
        _z(slpm) + _z(distance_share) + _z(kd_per_15) + _z(str_def)
        - _z(td_per_15) - _z(ctrl_pct)
    )
    out["wrestler_score"] = (
        _z(td_per_15) + _z(td_acc) + _z(ctrl_pct) - _z(distance_share)
    )
    out["grappler_score"] = (
        _z(sub_att_per_15) + _z(ground_share) + _z(sub_def)
    )
    out["pressure_score"] = (
        _z(vol_pm) + _z(clinch_share) + _z(slpm)
    )
    out["volume_score"] = (
        _z(vol_pm) + _z(td_attempted_per_15)
    )
    # chin_proxy: absorbed knockdown rate (how often the fighter gets put on the canvas)
    # Used by interactions.py power_vs_chin: kd_per_15_A × chin_proxy_B
    out["chin_proxy"] = kd_against_per_15

    return out


def fit_style_pca(
    df: pd.DataFrame,
    train_mask: pd.Series,
    n_components: int = 5,
    gitsha: str = "latest",
) -> pd.DataFrame:
    """Fit PCA on striking+grappling block (training fold only), add PC columns."""
    pca_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in
        ["slpm_", "sapm_", "str_acc_", "str_def_", "head_share_", "body_share_",
         "leg_share_", "distance_share_", "clinch_share_", "ground_share_",
         "kd_per_15_", "td_per_15_", "td_acc_", "td_def_", "ctrl_pct_",
         "sub_att_per_15_"]
    )]

    if len(pca_cols) < n_components:
        return df

    X = df[pca_cols].fillna(0).values
    X_train = X[train_mask.values]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled[train_mask.values])
    Xpca = pca.transform(X_scaled)

    for i in range(n_components):
        df[f"style_pc{i+1}"] = Xpca[:, i]

    # Persist
    model_dir = paths.outputs_models()
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "pca": pca, "cols": pca_cols},
        model_dir / f"pca_style_{gitsha}.joblib",
        compress=3,
    )

    return df


def apply_style_pca(df: pd.DataFrame, artifact_path) -> pd.DataFrame:
    """Apply saved PCA artifact to a new DataFrame (inference)."""
    art = joblib.load(artifact_path)
    pca_cols = art["cols"]
    cols_present = [c for c in pca_cols if c in df.columns]
    X = df.reindex(columns=pca_cols, fill_value=0).fillna(0).values
    X_scaled = art["scaler"].transform(X)
    Xpca = art["pca"].transform(X_scaled)
    for i in range(Xpca.shape[1]):
        df[f"style_pc{i+1}"] = Xpca[:, i]
    return df
