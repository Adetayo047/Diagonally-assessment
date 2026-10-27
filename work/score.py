"""Rank every customer by how likely they are to be a commercial loss case.

    python score.py --data-dir ..                   # writes submission.csv
    python score.py --data-dir .. --evaluate        # also prints precision@k against past inspections
    python score.py --readings new_month.csv --customers customers.csv --out submission.csv

Only meter_readings.csv and customers.csv are needed to score. inspections.csv is used
only by --evaluate. The logic and every threshold are explained in analysis.ipynb.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ================================================================ SETTINGS
# Every number that shapes the ranking lives here. Change one, re-run, compare with --evaluate.
# Why each value was chosen is shown in analysis.ipynb (section in brackets).

# How much we trust each pattern (max score it can give). Higher = ranked nearer the top. [§9]
WEIGHTS = {
    "stuck":      1.00,   # same reading to the decimal, month after month   -> near-certain
    "round":      0.95,   # bills are round numbers (estimated, not read)     -> near-certain
    "on_off":     0.85,   # usage flicks between normal and very low          -> tampering
    "bypass":     0.80,   # big drop, readings erratic afterwards             -> likely bypass
    "quiet_drop": 0.35,   # big drop, readings steady afterwards              -> may be genuine
}

# Stuck meter [§5.2]
STUCK_MIN_REPEATS = 3        # this many identical readings in a row = stuck

# Round-number (estimated) bills [§5.3]
ROUND_KWH = 50               # estimated bills are exact multiples of this
ROUND_SHARE_FULL = 0.40      # this share of round bills gives the full "round" score
ROUND_SHARE_IS_ESTIMATED = 0.20  # above this, repeats are estimates, not a stuck meter

# On/off tampering [§5.4]
LOW_MONTH_SHARE = 0.50       # a month is "low" if below this share of the customer's normal usage
FLIPS_START, FLIPS_FULL = 3, 7   # switches: score starts above 3, full score at 7+

# Drops (bypass / quiet drop) [§5.1, §6.2]
MIN_SIDE = 3                 # months needed on each side of a drop before it counts
DROP_START, DROP_FULL = 0.30, 0.55    # drop size: score starts above 30%, full at 55%+
NOISE_START, NOISE_FULL = 1.2, 2.5    # erratic-ness after drop vs before: starts 1.2x, full 2.5x


# ---------------------------------------------------------------- loading
def load(readings_path, customers_path):
    readings = pd.read_csv(readings_path, parse_dates=["reading_date"])
    customers = pd.read_csv(customers_path, parse_dates=["connection_date"])
    readings = readings.drop_duplicates(["customer_id", "reading_date"], keep="last")
    return readings, customers


def build_matrices(readings, customers):
    """Customer x month matrices: raw kWh, status, and seasonally adjusted kWh."""
    r = readings.merge(customers[["customer_id", "premises_type"]], on="customer_id", how="left")
    r["premises_type"] = r.premises_type.fillna("unknown")
    # Seasonal index: median usage relative to each customer's own median, per premises type and month.
    r["rel"] = r.kwh_billed / r.groupby("customer_id").kwh_billed.transform("median")
    season = (r[r.account_status == "active"].groupby(["premises_type", "reading_date"]).rel.median()
              .rename("season_idx"))
    r = r.join(season, on=["premises_type", "reading_date"])
    r["deseason"] = r.kwh_billed / r.season_idx.fillna(1.0)

    def pivot(col):
        return r.pivot(index="customer_id", columns="reading_date", values=col)
    return pivot("kwh_billed"), pivot("account_status"), pivot("deseason")


# ---------------------------------------------------------------- features
def rcv(v):
    """Robust coefficient of variation: median absolute deviation / median."""
    med = np.median(v)
    return np.median(np.abs(v - med)) / med if med > 0 else np.nan


def step_drop(v, index):
    """Largest sustained drop: split month where median(after)/median(before) is lowest."""
    best_ratio, best_k = 1.0, None
    for k in range(MIN_SIDE, len(v) - MIN_SIDE + 1):
        ratio = np.median(v[k:]) / np.median(v[:k])
        if ratio < best_ratio:
            best_ratio, best_k = ratio, k
    if best_k is None:
        return dict(drop_pct=0.0, onset=pd.NaT, months_after=0, noise_ratio=np.nan)
    before, after = v[:best_k], v[best_k:]
    return dict(drop_pct=1 - best_ratio, onset=index[best_k], months_after=len(after),
                noise_ratio=rcv(after) / max(rcv(before), 0.02))


def longest_identical_run(v):
    best = run = 1 if len(v) else 0
    for a, b in zip(v[:-1], v[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def on_off(v):
    """How often usage switches between normal and below half of normal."""
    if len(v) < 4:
        return dict(flips=0, low_months=0)
    normal = np.percentile(v, 75)          # "normal" level; theft months can't drag it down
    low = v < LOW_MONTH_SHARE * normal
    return dict(flips=int((low[1:] != low[:-1]).sum()), low_months=int(low.sum()))


def build_features(raw, status, deseason, cutoffs=None):
    """One row per customer. Vacant months are removed first; cutoffs = optional last date per customer."""
    rows = {}
    for cid in raw.index:
        active = status.loc[cid] == "active"
        rw, ds = raw.loc[cid][active].dropna(), deseason.loc[cid][active].dropna()
        cut = None if cutoffs is None else cutoffs.get(cid)
        if cut is not None and pd.notna(cut):
            rw, ds = rw[rw.index <= cut], ds[ds.index <= cut]
        rows[cid] = dict(n_months=len(rw),
                         last_kwh=rw.iloc[-1] if len(rw) else np.nan,
                         stuck_run=longest_identical_run(rw.values),
                         round_share=float((rw % ROUND_KWH == 0).mean()) if len(rw) else 0.0,
                         **step_drop(ds.values, ds.index), **on_off(ds.values))
    f = pd.DataFrame(rows).T
    num = [c for c in f.columns if c != "onset"]
    f[num] = f[num].astype(float)
    f["onset"] = pd.to_datetime(f.onset)
    return f


# ---------------------------------------------------------------- scoring
def ramp(x, start, full):
    """0 at or below `start`, 1 at or above `full`, straight line in between."""
    return np.clip((x - start) / (full - start), 0, 1)


def detector_scores(f):
    """One column per pattern, each between 0 and its weight."""
    drop_strength = ramp(f.drop_pct, DROP_START, DROP_FULL) * (f.months_after >= MIN_SIDE)
    noise_strength = ramp(f.noise_ratio.fillna(0), NOISE_START, NOISE_FULL)
    d = pd.DataFrame({
        # stuck only if the repeats aren't round estimated bills (those count as "round")
        "stuck": ((f.stuck_run >= STUCK_MIN_REPEATS)
                  & (f.round_share < ROUND_SHARE_IS_ESTIMATED)).astype(float),
        "round": ramp(f.round_share, 0, ROUND_SHARE_FULL),
        "on_off": ramp(f.flips, FLIPS_START, FLIPS_FULL),
        "bypass": drop_strength * noise_strength,      # needs a big drop AND erratic after
        "quiet_drop": drop_strength,                   # big drop alone (weight keeps it low)
    }, index=f.index)
    return d * pd.Series(WEIGHTS)


def score(f):
    d = detector_scores(f)
    # Strongest single pattern decides; a small bonus orders ties by total evidence.
    risk = (d.max(axis=1) + 0.01 * d.sum(axis=1)).clip(0, 1)
    reason = d.idxmax(axis=1).where(d.max(axis=1) > 0, "none")
    return risk, reason


def explain(key, f):
    if key == "stuck":
        return f"meter has shown exactly {f.last_kwh:.0f} kWh for {int(f.stuck_run)} months running"
    if key == "round":
        return f"{f.round_share:.0%} of bills are round estimates, not real meter reads"
    if key == "on_off":
        return f"usage keeps switching between normal and very low ({int(f.flips)} switches)"
    if key == "bypass":
        return f"usage fell {f.drop_pct:.0%} in {f.onset:%b %Y} and has been erratic since"
    if key == "quiet_drop":
        return f"usage fell {f.drop_pct:.0%} in {f.onset:%b %Y}; may be genuine, lower priority"
    if f.n_months == 0:
        return "no billing history to assess"
    return "no unusual pattern in billed usage"


def make_submission(customers, feats):
    feats = feats.reindex(customers.customer_id)      # every customer, even with no readings
    feats["n_months"] = feats.n_months.fillna(0)
    risk, reason = score(feats.fillna({"stuck_run": 0, "round_share": 0, "drop_pct": 0,
                                       "months_after": 0, "flips": 0}))
    sub = pd.DataFrame({
        "customer_id": feats.index,
        "risk_score": risk.fillna(0).round(4).values,
        "primary_reason": [explain(reason[c], feats.loc[c]) for c in feats.index],
        "pattern": reason.values,
    })
    return sub.sort_values("risk_score", ascending=False, kind="stable").reset_index(drop=True)


def validate(sub, customers):
    assert list(sub.columns) == ["customer_id", "risk_score", "primary_reason"]
    assert sub.customer_id.is_unique and set(sub.customer_id) == set(customers.customer_id)
    assert sub.risk_score.notna().all() and sub.risk_score.between(0, 1).all()
    assert (sub.primary_reason.str.split().str.len() < 15).all(), "reason too long"


# ---------------------------------------------------------------- evaluation
def evaluate(inspections_path, customers, raw, status, deseason, feats_now, ks=(10, 25, 50, 100)):
    """Precision@k on past inspections, using only data up to each inspection date."""
    insp = pd.read_csv(inspections_path, parse_dates=["inspection_date"])
    ever_inspected = pd.Index(insp.customer_id)
    insp = insp.merge(customers[["customer_id", "connection_date"]], on="customer_id")
    insp = insp[(insp.outcome != "no_access") & (insp.inspection_date >= insp.connection_date)]
    insp = insp[insp.customer_id.isin(raw.index)].set_index("customer_id")
    y = (insp.outcome == "theft_confirmed").astype(int)

    feats_pit = build_features(raw.loc[y.index], status.loc[y.index], deseason.loc[y.index],
                               insp.inspection_date)
    risk_pit, _ = score(feats_pit)

    # Hard negatives: never-inspected customers with vacant months (vacancy explains their low usage).
    vacant = status.index[(status == "vacant").any(axis=1)].difference(ever_inspected)
    risk_now, _ = score(feats_now)
    y_hard = pd.concat([y, pd.Series(0, index=vacant)])
    risk_hard = pd.concat([risk_pit, risk_now[vacant]])

    def p_at_k(s, lab):
        order = s.loc[lab.index].sort_values(ascending=False, kind="stable").index
        return [lab[order[:k]].mean() for k in ks]

    table = pd.DataFrame({
        f"labelled only (n={len(y)})": p_at_k(risk_pit, y),
        f"+ vacant hard negatives (n={len(y_hard)})": p_at_k(risk_hard, y_hard),
    }, index=[f"precision@{k}" for k in ks])
    table.loc["random guessing"] = [y.mean(), y_hard.mean()]
    print("\nPrecision on past inspections (point-in-time features):")
    print(table.round(2).to_string())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="folder holding the CSVs (default: current folder)")
    ap.add_argument("--readings", help="meter readings CSV (default: <data-dir>/meter_readings.csv)")
    ap.add_argument("--customers", help="customers CSV (default: <data-dir>/customers.csv)")
    ap.add_argument("--inspections", help="inspections CSV for --evaluate (default: <data-dir>/inspections.csv)")
    ap.add_argument("--out", default="submission.csv", help="output path (default: submission.csv)")
    ap.add_argument("--evaluate", action="store_true", help="print precision@k against past inspections")
    args = ap.parse_args()

    data = Path(args.data_dir)
    readings, customers = load(args.readings or data / "meter_readings.csv",
                               args.customers or data / "customers.csv")
    raw, status, deseason = build_matrices(readings, customers)
    feats = build_features(raw, status, deseason)
    sub = make_submission(customers, feats)
    patterns = sub.pop("pattern")
    validate(sub, customers)
    sub.to_csv(args.out, index=False, encoding="utf-8")

    months = readings.reading_date
    print(f"Scored {len(sub)} customers on {months.min():%b %Y}-{months.max():%b %Y} readings -> {args.out}")
    print("Top 100 by pattern:", patterns.head(100).value_counts().to_dict())
    if args.evaluate:
        evaluate(args.inspections or data / "inspections.csv", customers, raw, status, deseason, feats)


if __name__ == "__main__":
    main()
