"""ELO, Glicko-2, and TrueSkill rating systems — chronological single pass."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml

from ufc.io import paths


def _pre_ufc_elo_offset(fighter_id: str, pre_ufc_lookup: "pd.DataFrame | None",
                        k: float = 100.0, cap: float = 120.0) -> float:
    if pre_ufc_lookup is None or fighter_id not in pre_ufc_lookup.index:
        return 0.0
    row = pre_ufc_lookup.loc[fighter_id]
    p = float(row.get("pre_ufc_win_rate_shrunk", 0.5))
    n = float(row.get("pre_ufc_n", 0))
    return float(np.clip(k * (p - 0.5) * np.log1p(n), -cap, cap))


def _load_cfg() -> dict:
    with open(paths.root() / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)["elo"]


def _injury_k() -> float:
    return float(_load_cfg().get("injury_k_factor", 0.25))


# ── ELO ──────────────────────────────────────────────────────────────────────

def compute_elo(ledger: pd.DataFrame,
                pre_ufc_lookup: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Chronological ELO pass.

    Returns ledger with columns added:
        elo_pre       — fighter's ELO rating BEFORE this fight (the feature)
        opp_elo_pre   — opponent's ELO rating before fight
        elo_diff      — elo_pre - opp_elo_pre
    """
    cfg = _load_cfg()
    K_BASE = cfg["k_base"]
    METHOD_MULT = cfg["method_multipliers"]
    INACT_THRESHOLD = cfg["inactivity_threshold_days"]
    INACT_REGRESS = cfg["inactivity_regress_pct"]
    Q_SCALE = cfg["quality_scale"]
    Q_WIDTH = cfg["quality_width"]
    INITIAL = cfg["initial_rating"]

    elo: dict[str, float] = {}  # lazy-init with optional pre-UFC seed
    last_fight_date: dict[str, date | None] = defaultdict(lambda: None)

    def _init_elo(fid: str) -> float:
        return INITIAL + _pre_ufc_elo_offset(fid, pre_ufc_lookup)

    # Sort to get unique fights in chronological order
    _cols = ["fight_id", "event_date", "fighter_id", "opponent_id", "won", "method"]
    if "injury_freak" in ledger.columns:
        _cols = _cols + ["injury_freak"]
    fights = (
        ledger[_cols]
        .drop_duplicates(subset=["fight_id", "fighter_id"])
        .sort_values(["event_date", "fight_id"])
        .copy()
    )

    # Build (fight_id -> (fighter_a_id, fighter_b_id, a_won, method, event_date))
    fight_pairs = {}
    for _, row in fights.iterrows():
        fid = row["fight_id"]
        if fid not in fight_pairs:
            fight_pairs[fid] = {
                "event_date": row["event_date"],
                "method": row["method"],
                "injury": bool(row.get("injury_freak", False)),
                "fighters": [],
            }
        fight_pairs[fid]["fighters"].append((row["fighter_id"], row["won"]))

    # Process fights chronologically
    elo_records: dict[tuple, dict] = {}  # (fight_id, fighter_id) -> {elo_pre, opp_elo_pre}

    for fid, info in sorted(fight_pairs.items(), key=lambda x: (x[1]["event_date"], x[0])):
        event_date = info["event_date"]
        method = info["method"]
        fighters = info["fighters"]

        if len(fighters) == 1:
            # Sentinel row (inference-state carrier): read current rating, no update.
            f_s, _ = fighters[0]
            if f_s not in elo:
                elo[f_s] = _init_elo(f_s)
            elo_records[(fid, f_s)] = {"elo_pre": elo[f_s], "opp_elo_pre": np.nan}
            continue
        if len(fighters) != 2:
            continue

        (f_a, won_a), (f_b, won_b) = fighters[0], fighters[1]

        ed = event_date.date() if hasattr(event_date, "date") else event_date

        # Lazy init + inactivity decay before fight
        for fid_fighter in [f_a, f_b]:
            if fid_fighter not in elo:
                elo[fid_fighter] = _init_elo(fid_fighter)
            prev = last_fight_date[fid_fighter]
            if prev is not None:
                gap_days = (ed - prev).days if isinstance(ed, date) else 0
                if gap_days > INACT_THRESHOLD:
                    current = elo[fid_fighter]
                    elo[fid_fighter] = current + INACT_REGRESS * (INITIAL - current)

        r_a = elo[f_a]
        r_b = elo[f_b]

        # Store pre-fight ratings
        elo_records[(fid, f_a)] = {"elo_pre": r_a, "opp_elo_pre": r_b}
        elo_records[(fid, f_b)] = {"elo_pre": r_b, "opp_elo_pre": r_a}

        # Update only if outcome is known
        if pd.isna(won_a) or won_a not in (0, 1):
            continue

        m_mult = METHOD_MULT.get(method, 0.0)
        if m_mult == 0:
            continue

        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        q_mult_a = 1.0 + Q_SCALE * math.tanh((r_b - INITIAL) / Q_WIDTH)
        q_mult_b = 1.0 + Q_SCALE * math.tanh((r_a - INITIAL) / Q_WIDTH)
        K_a = K_BASE * m_mult * q_mult_a
        K_b = K_BASE * m_mult * q_mult_b
        if info.get("injury"):
            K_a *= _injury_k()
            K_b *= _injury_k()

        elo[f_a] += K_a * (int(won_a) - e_a)
        elo[f_b] += K_b * ((1 - int(won_a)) - (1 - e_a))

        if isinstance(ed, date):
            last_fight_date[f_a] = ed
            last_fight_date[f_b] = ed

    # Merge back into ledger
    elo_df = pd.DataFrame(
        [{"fight_id": fid, "fighter_id": fid_f, **vals}
         for (fid, fid_f), vals in elo_records.items()]
    )
    merged = pd.merge(
        ledger,
        elo_df,
        on=["fight_id", "fighter_id"],
        how="left",
    )
    merged["elo_diff"] = merged["elo_pre"] - merged["opp_elo_pre"]
    return merged


# ── Glicko-2 ─────────────────────────────────────────────────────────────────

def compute_glicko2(ledger: pd.DataFrame,
                    pre_ufc_lookup: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Simplified Glicko-2 for rating uncertainty (RD).

    Returns ledger with columns: glicko_mu_pre, glicko_rd_pre, glicko_phi_pre
    """
    with open(paths.root() / "configs" / "features.yaml") as f:
        gcfg = yaml.safe_load(f)["glicko2"]

    MU0 = gcfg["initial_mu"]
    PHI0 = gcfg["initial_phi"]
    SIGMA0 = gcfg["initial_sigma"]
    TAU = gcfg["tau"]

    # State: mu (rating on 1500 scale), phi (RD), sigma (volatility) — lazy-init
    mu: dict[str, float] = {}
    phi: dict[str, float] = {}
    sigma: dict[str, float] = {}

    def _init_mu(fid: str) -> float:
        return MU0 + _pre_ufc_elo_offset(fid, pre_ufc_lookup)

    def _ensure(fid: str) -> None:
        if fid not in mu:
            mu[fid] = _init_mu(fid)
            phi[fid] = PHI0
            sigma[fid] = SIGMA0

    fight_pairs = {}
    _cols = ["fight_id", "event_date", "fighter_id", "opponent_id", "won", "method"]
    if "injury_freak" in ledger.columns:
        _cols = _cols + ["injury_freak"]
    fights = (
        ledger[_cols]
        .drop_duplicates(subset=["fight_id", "fighter_id"])
        .sort_values(["event_date", "fight_id"])
    )
    for _, row in fights.iterrows():
        fid = row["fight_id"]
        if fid not in fight_pairs:
            fight_pairs[fid] = {
                "event_date": row["event_date"],
                "method": row["method"],
                "injury": bool(row.get("injury_freak", False)),
                "fighters": [],
            }
        fight_pairs[fid]["fighters"].append((row["fighter_id"], row["won"]))

    def _g(phi_j):
        return 1.0 / math.sqrt(1 + 3 * phi_j**2 / math.pi**2)

    def _E(mu_i, mu_j, phi_j):
        return 1.0 / (1 + math.exp(-_g(phi_j) * (mu_i - mu_j) / 400.0))

    glicko_records: dict[tuple, dict] = {}

    for fid, info in sorted(fight_pairs.items(), key=lambda x: (x[1]["event_date"], x[0])):
        fighters = info["fighters"]
        if len(fighters) == 1:
            # Sentinel row: read current rating, no update.
            f_s, _ = fighters[0]
            _ensure(f_s)
            glicko_records[(fid, f_s)] = {
                "glicko_mu_pre": mu[f_s], "glicko_rd_pre": phi[f_s], "glicko_z": np.nan,
            }
            continue
        if len(fighters) != 2:
            continue

        (f_a, won_a), (f_b, won_b) = fighters[0], fighters[1]
        _ensure(f_a); _ensure(f_b)

        # Uncertainty-scaled rating diff: same gap matters more between veterans
        phi_a, phi_b = phi[f_a], phi[f_b]
        denom = math.sqrt(phi_a**2 + phi_b**2) if (phi_a**2 + phi_b**2) > 0 else 1.0
        gz = (mu[f_a] - mu[f_b]) / denom
        glicko_records[(fid, f_a)] = {"glicko_mu_pre": mu[f_a], "glicko_rd_pre": phi_a, "glicko_z": gz}
        glicko_records[(fid, f_b)] = {"glicko_mu_pre": mu[f_b], "glicko_rd_pre": phi_b, "glicko_z": -gz}

        if pd.isna(won_a) or won_a not in (0, 1):
            # Increase uncertainty (no result)
            for ff in [f_a, f_b]:
                phi_star = math.sqrt(phi[ff]**2 + sigma[ff]**2)
                phi[ff] = min(phi_star, PHI0)
            continue

        # Update both fighters (full Glicko-2 volatility update)
        for f_i, f_j, s_ij in [(f_a, f_b, float(won_a)), (f_b, f_a, float(1 - won_a))]:
            g_j = _g(phi[f_j])
            E_ij = _E(mu[f_i], mu[f_j], phi[f_j])
            v = 1.0 / (g_j**2 * E_ij * (1 - E_ij))
            delta = v * g_j * (s_ij - E_ij)

            # Volatility update (simplified)
            a = math.log(sigma[f_i]**2)
            A = a
            f_fn = lambda x: (
                math.exp(x) * (delta**2 - phi[f_i]**2 - v - math.exp(x)) /
                (2 * (phi[f_i]**2 + v + math.exp(x))**2) - (x - a) / TAU**2
            )
            # Illinois algorithm
            B = math.log(delta**2 - phi[f_i]**2 - v) if delta**2 > phi[f_i]**2 + v else a - 6 * TAU
            fA, fB = f_fn(A), f_fn(B)
            for _ in range(100):
                C = A + (A - B) * fA / (fB - fA)
                fC = f_fn(C)
                if fB * fC < 0:
                    A, fA = B, fB
                else:
                    fA /= 2
                B, fB = C, fC
                if abs(B - A) < 1e-6:
                    break
            sigma[f_i] = math.exp(A / 2)

            phi_star = math.sqrt(phi[f_i]**2 + sigma[f_i]**2)
            phi[f_i] = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
            scale = _injury_k() if info.get("injury") else 1.0
            mu[f_i] = mu[f_i] + scale * phi[f_i]**2 * g_j * (s_ij - E_ij)

    glicko_df = pd.DataFrame(
        [{"fight_id": fid, "fighter_id": fid_f, **vals}
         for (fid, fid_f), vals in glicko_records.items()]
    )
    return pd.merge(ledger, glicko_df, on=["fight_id", "fighter_id"], how="left")


# ── TrueSkill ─────────────────────────────────────────────────────────────────

def compute_trueskill(ledger: pd.DataFrame,
                      pre_ufc_lookup: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Online TrueSkill pass (Gaussian factor graph, structurally different from Glicko-2).

    Returns ledger with columns: ts_mu_pre, ts_sigma_pre, ts_z
    """
    try:
        from trueskill import TrueSkill, Rating  # noqa: PLC0415
    except ImportError:
        ledger["ts_mu_pre"] = np.nan
        ledger["ts_sigma_pre"] = np.nan
        ledger["ts_z"] = 0.0
        return ledger

    env = TrueSkill(draw_probability=0.01, backend="scipy")
    MU0, SIGMA0 = env.mu, env.sigma

    ratings: dict[str, Rating] = {}

    def _init_ts(fid: str) -> Rating:
        offset_1500 = _pre_ufc_elo_offset(fid, pre_ufc_lookup)
        mu_seed = MU0 + offset_1500 / 1500.0 * MU0
        return env.create_rating(mu=mu_seed, sigma=SIGMA0)

    _cols = ["fight_id", "event_date", "fighter_id", "opponent_id", "won"]
    if "injury_freak" in ledger.columns:
        _cols = _cols + ["injury_freak"]
    fights = (
        ledger[_cols]
        .drop_duplicates(subset=["fight_id", "fighter_id"])
        .sort_values(["event_date", "fight_id"])
    )
    fight_pairs: dict[str, dict] = {}
    for _, row in fights.iterrows():
        fid = row["fight_id"]
        if fid not in fight_pairs:
            fight_pairs[fid] = {
                "event_date": row["event_date"],
                "injury": bool(row.get("injury_freak", False)),
                "fighters": [],
            }
        fight_pairs[fid]["fighters"].append((row["fighter_id"], row["won"]))

    ts_records: dict[tuple, dict] = {}

    for fid, info in sorted(fight_pairs.items(), key=lambda x: (x[1]["event_date"], x[0])):
        fighters = info["fighters"]
        if len(fighters) == 1:
            # Sentinel row: read current rating, no update.
            f_s, _ = fighters[0]
            if f_s not in ratings:
                ratings[f_s] = _init_ts(f_s)
            ts_records[(fid, f_s)] = {
                "ts_mu_pre": ratings[f_s].mu, "ts_sigma_pre": ratings[f_s].sigma, "ts_z": np.nan,
            }
            continue
        if len(fighters) != 2:
            continue
        (f_a, won_a), (f_b, won_b) = fighters[0], fighters[1]

        if f_a not in ratings:
            ratings[f_a] = _init_ts(f_a)
        if f_b not in ratings:
            ratings[f_b] = _init_ts(f_b)

        ra, rb = ratings[f_a], ratings[f_b]
        denom = math.sqrt(ra.sigma**2 + rb.sigma**2) if (ra.sigma**2 + rb.sigma**2) > 0 else 1.0
        tz = (ra.mu - rb.mu) / denom

        ts_records[(fid, f_a)] = {"ts_mu_pre": ra.mu, "ts_sigma_pre": ra.sigma, "ts_z": tz}
        ts_records[(fid, f_b)] = {"ts_mu_pre": rb.mu, "ts_sigma_pre": rb.sigma, "ts_z": -tz}

        if pd.isna(won_a) or won_a not in (0, 1):
            continue

        if int(won_a) == 1:
            new_ra, new_rb = env.rate_1vs1(ra, rb)
        else:
            new_rb, new_ra = env.rate_1vs1(rb, ra)

        if info.get("injury"):
            f = _injury_k()
            new_ra = env.create_rating(mu=ra.mu + f * (new_ra.mu - ra.mu), sigma=new_ra.sigma)
            new_rb = env.create_rating(mu=rb.mu + f * (new_rb.mu - rb.mu), sigma=new_rb.sigma)

        ratings[f_a] = new_ra
        ratings[f_b] = new_rb

    ts_df = pd.DataFrame(
        [{"fight_id": fid, "fighter_id": fid_f, **vals}
         for (fid, fid_f), vals in ts_records.items()]
    )
    return pd.merge(ledger, ts_df, on=["fight_id", "fighter_id"], how="left")
