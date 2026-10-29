"""Run the forecasting pipeline for every country and write results/ for the app and dashboard.

    python build_results.py            # all countries (about 20-30 min on 4 cores)
    python build_results.py India      # just one country, for a quick check
"""
import os

# One BLAS thread per worker: four processes x many threads oversubscribes the CPU badly.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import transition as T

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
OUT = ROOT / "results"
END_YEAR = 2050
N_PATHS = 2000
H = 24              # months of coal / solar forecast shown in the dashboard
MIN_MONTHS = 60
CHART_SOURCES = ["Coal", "Gas", "Other fossil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other renewables"]


clean_series = T.clean_series


def fan(res, h, back=np.exp, exog_cols=0):
    exog = np.zeros((h, exog_cols)) if exog_cols else None
    fc = res.get_forecast(h, exog=exog)
    ci80, ci95 = fc.conf_int(alpha=0.2), fc.conf_int(alpha=0.05)
    return {
        "dates": [d.strftime("%Y-%m") for d in fc.predicted_mean.index],
        "mid": np.round(back(fc.predicted_mean.values), 3).tolist(),
        "lo80": np.round(back(ci80.iloc[:, 0].values), 3).tolist(), "hi80": np.round(back(ci80.iloc[:, 1].values), 3).tolist(),
        "lo95": np.round(back(ci95.iloc[:, 0].values), 3).tolist(), "hi95": np.round(back(ci95.iloc[:, 1].values), 3).tolist(),
    }


def coverage(y, spec, transform, back, h=12):
    """Fit on everything except the last h months, forecast them, report 80%-interval coverage and MAPE."""
    z = transform(y)
    res = T.fit_sarima(z.iloc[:-h], spec["order"], spec["seasonal_order"])
    fc = res.get_forecast(h)
    lo, hi = fc.conf_int(alpha=0.2).values.T
    actual = z.iloc[-h:].values
    out = {"cover80": float(np.mean((actual >= lo) & (actual <= hi)))}
    if back is np.exp:
        out["mape"] = T.metrics(np.exp(actual), np.exp(fc.predicted_mean.values))["MAPE (%)"]
    else:
        out["mae"] = T.metrics(actual, fc.predicted_mean.values)["MAE"]
    return out


def run_country(args):
    name, panel = args
    rec = {"country": name}
    try:
        last12 = panel.iloc[-12:]
        total = last12["Total generation"].sum()
        rec["share12"] = {s: round(float(last12[s].sum() / total * 100), 2) for s in CHART_SOURCES if s in last12}
        rec["prev_share12"] = {s: round(float(panel.iloc[-24:-12][s].sum() / panel.iloc[-24:-12]["Total generation"].sum() * 100), 2)
                               for s in ("Coal", "Solar") if s in panel}
        rec["last_month"] = panel.index[-1].strftime("%Y-%m")
        rec["first_month"] = panel.index[0].strftime("%Y-%m")
        rec["history"] = {"dates": [d.strftime("%Y-%m") for d in panel.index]}
        for s in CHART_SOURCES + ["Total generation"]:
            if s in panel:
                rec["history"][s] = [None if pd.isna(v) else round(float(v), 3) for v in panel[s].values]

        coal_share = rec["share12"].get("Coal", 0)
        if "Solar" not in panel:
            panel["Solar"] = 0.0          # the country reports no solar generation at all
        solar = clean_series(panel["Solar"])
        coal = clean_series(panel["Coal"]) if coal_share >= 1 else None
        rec["models"], rec["forecast"], rec["calibration"] = {}, {}, {}

        for key, y in (("Solar", solar), ("Coal", coal)):
            if y is None or len(y) < MIN_MONTHS:
                continue
            spec = T.auto_sarima(y, small=True)
            res = T.fit_sarima(np.log(y), spec["order"], spec["seasonal_order"])
            rec["models"][key] = {"model": spec["model"], "cv_mape": round(spec["cv"][spec["model"]], 2)}
            rec["forecast"][key] = fan(res, H)
            rec["calibration"][key] = coverage(y, spec, np.log, np.exp)

        if coal is None:
            rec["status"] = "no-coal"
            return rec
        if len(solar) < MIN_MONTHS:
            rec["status"] = "too-short"     # not five continuous years of solar data
            return rec
        r = np.log(clean_series(np.exp(T.log_ratio(solar, coal))))  # same gap/contiguity rule on the ratio
        if len(r) < MIN_MONTHS:
            rec["status"] = "too-short"
            return rec

        spec = T.auto_sarima(r, small=True, transform=T.identity, back=T.identity, score="ae")
        res = T.fit_sarima(r, spec["order"], spec["seasonal_order"])
        rec["models"]["LogRatio"] = {"model": spec["model"], "cv_mae": round(spec["cv"][spec["model"]], 3)}
        rec["calibration"]["LogRatio"] = coverage(r, spec, T.identity, T.identity)

        cx = T.crossover_distribution(r, res, end_year=END_YEAR, n_paths=N_PATHS)
        rec["observed_crossover"] = T.observed_crossover_year(r)
        above = r[r > 0]
        rec["first_month_above"] = above.index[0].strftime("%Y-%m") if len(above) else None
        rec["last12_ratio"] = round(float(np.exp(r.iloc[-12:].mean())), 4)  # solar / coal, geometric mean
        rec["crossover"] = {k: {a: (None if (b is None or not np.isfinite(b)) else round(float(b), 3))
                                for a, b in T.summarise_crossover(cx[k], END_YEAR).items()} for k in ("month", "year")}
        yrs = cx["year"]
        bins = list(range(int(r.index[-1].year), END_YEAR + 1))
        rec["crossover_hist"] = {"years": bins, "p": [round(float(np.mean(yrs == b)), 4) for b in bins],
                                 "p_after": round(float(np.mean(np.isnan(yrs))), 4)}
        annual = cx["paths"].groupby(cx["paths"].index.year).mean()
        rec["ratio_fan"] = {"years": annual.index.tolist(),
                            **{k: np.round(np.exp(annual.quantile(q, axis=1).values), 4).tolist()
                               for k, q in (("p10", .1), ("p50", .5), ("p90", .9))}}
        rec["ratio_hist"] = {"years": sorted(set(r.index.year)),
                             "ratio": np.round(np.exp(r.groupby(r.index.year).mean().values), 4).tolist()}
        if rec["last12_ratio"] > 1:
            rec["status"] = "crossed"
        else:
            rec["status"] = "forecast"
    except Exception as e:  # keep the batch running; report what failed
        rec["status"] = "error"
        rec["error"] = repr(e)
    return rec


def india_target(yearly):
    cap = yearly[yearly["Area"] == "India"].pivot_table(index="Year", columns="Electricity source", values="Capacity (GW)")["Clean"]
    out = {"history": {"years": cap.index.tolist(), "gw": cap.round(2).tolist()}, "target": 500, "target_year": 2030,
           "required_cagr": ((500 / cap.iloc[-1]) ** (1 / (2030 - cap.index[-1])) - 1) * 100,
           "cagr": {f"{a}-{cap.index[-1]}": ((cap.iloc[-1] / cap[a]) ** (1 / (cap.index[-1] - a)) - 1) * 100 for a in (2000, 2015, 2020)},
           "scenarios": {}}
    for label, d in (("Long-run trend", 1), ("Accelerating growth", 2)):
        t = T.target_analysis(cap, 500, 2030, d)
        out["scenarios"][label] = {"model": t["model"], "p_hit": t["p_hit"],
                                   "years": t["fan"].index.tolist(),
                                   **{k: t["fan"][k].round(1).tolist() for k in ("p10", "median", "p90")}}
    return out


def merge_partial():
    """Fold results_partial.json (a re-run of some countries) into results.json."""
    full = json.loads((OUT / "results.json").read_text())
    part = json.loads((OUT / "results_partial.json").read_text())
    new = {c["country"]: c for c in part["countries"]}
    full["countries"] = [new.get(c["country"], c) for c in full["countries"]]
    (OUT / "results.json").write_text(json.dumps(full, separators=(",", ":"), default=float))
    print("merged", sorted(new))


def main(only=None):
    OUT.mkdir(exist_ok=True)
    monthly = T.load_monthly(ROOT / "data" / "ember_monthly_global.csv")
    yearly = T.load_yearly(ROOT / "data" / "ember_yearly_global.csv")
    names = sorted(monthly["Area"].unique()) if not only else only
    jobs = []
    for n in names:
        p = T.country_panel(monthly, n)
        if len(p) >= MIN_MONTHS and p.index[-1] >= pd.Timestamp("2025-01-01"):
            jobs.append((n, p))
    print(f"{len(jobs)} countries with >= {MIN_MONTHS} months of recent data")
    with Pool(4) as pool:
        recs = []
        for i, rec in enumerate(pool.imap_unordered(run_country, jobs), 1):
            recs.append(rec)
            print(f"[{i}/{len(jobs)}] {rec['country']}: {rec['status']} {rec.get('error', '')}", flush=True)
    recs.sort(key=lambda r: r["country"])
    iso = monthly.drop_duplicates("Area").set_index("Area")["ISO 3 code"].to_dict()
    for r in recs:
        r["iso3"] = iso.get(r["country"])
    payload = {"source": "Ember monthly & yearly electricity data (CC-BY-4.0)", "end_year": END_YEAR,
               "data_through": max(r["last_month"] for r in recs if "last_month" in r),
               "countries": recs}
    if not only or "India" in only:
        payload["india_target"] = india_target(yearly)
    fname = "results.json" if not only else "results_partial.json"
    (OUT / fname).write_text(json.dumps(payload, separators=(",", ":"), default=float))
    print("wrote", OUT / fname, f"{(OUT / fname).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rerun"]:
        main(sys.argv[2:])
        merge_partial()
    else:
        main(sys.argv[1:] or None)
