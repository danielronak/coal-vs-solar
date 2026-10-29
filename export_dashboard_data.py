"""Flatten results/results.json into the tidy CSV tables the web dashboard reads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "dashboard_data"
OUT.mkdir(exist_ok=True)
res = json.loads((ROOT / "results" / "results.json").read_text())
countries = [c for c in res["countries"] if "history" in c]

# 1. monthly generation (TWh) for every country, sources grouped as in the app
GROUPS = {"coal": ["Coal"], "solar": ["Solar"], "wind": ["Wind"], "hydro": ["Hydro"],
          "gas_other_fossil": ["Gas", "Other fossil"], "nuclear_bio_other": ["Nuclear", "Bioenergy", "Other renewables"]}
rows = []
for c in countries:
    h = c["history"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "dates"}, index=h["dates"])
    out = pd.DataFrame({g: df[[s for s in cols if s in df]].sum(axis=1, min_count=1) for g, cols in GROUPS.items()})
    out["total"] = df["Total generation"]
    out.insert(0, "month", out.index)
    out.insert(0, "country", c["country"])
    rows.append(out)
gen = pd.concat(rows).round(3)
gen.to_csv(OUT / "generation.csv", index=False)

# 2. one row per country: shares, status and crossover odds
STATUS = {"forecast": "Forecast", "crossed": "Solar already ahead", "no-coal": "Little or no coal",
          "too-short": "Too little solar data", "error": "Not modelled"}
summ = []
for c in countries:
    x = (c.get("crossover") or {}).get("year", {})
    summ.append({
        "country": c["country"], "status": STATUS.get(c["status"], c["status"]),
        "coal_share": c["share12"].get("Coal", 0.0), "solar_share": c["share12"].get("Solar", 0.0),
        "coal_share_prev": c.get("prev_share12", {}).get("Coal"), "solar_share_prev": c.get("prev_share12", {}).get("Solar"),
        "solar_to_coal": c.get("last12_ratio"),
        "median_year": x.get("median"), "p10_year": x.get("p10"), "p90_year": x.get("p90"),
        "p_by_2030": x.get("P(by 2030)"), "p_by_2035": x.get("P(by 2035)"), "p_by_2040": x.get("P(by 2040)"),
        "p_before_2051": x.get("P(before 2050)"),
        "first_full_year_ahead": c.get("observed_crossover"), "first_month_ahead": c.get("first_month_above"),
        "ratio_model": c.get("models", {}).get("LogRatio", {}).get("model"),
        "solar_model": c.get("models", {}).get("Solar", {}).get("model"),
        "coal_model": c.get("models", {}).get("Coal", {}).get("model"),
        "data_from": c["first_month"], "data_to": c["last_month"],
    })
summ = pd.DataFrame(summ)
summ.loc[summ["p90_year"].isin([np.inf]), "p90_year"] = None
summ.to_csv(OUT / "outlook.csv", index=False)

# 3. 24-month forecasts with 80% and 95% intervals
fc = []
for c in countries:
    for src, f in c.get("forecast", {}).items():
        d = pd.DataFrame({k: f[k] for k in ("mid", "lo80", "hi80", "lo95", "hi95")})
        d.insert(0, "month", f["dates"]); d.insert(0, "source", src.lower()); d.insert(0, "country", c["country"])
        fc.append(d)
pd.concat(fc).round(3).to_csv(OUT / "forecast.csv", index=False)

# 4. probability that solar first beats coal (annual) in each year, and by each year
odds = []
for c in countries:
    if "crossover_hist" not in c:
        continue
    h = c["crossover_hist"]
    cum = np.cumsum(h["p"])
    for y, p, q in zip(h["years"], h["p"], cum):
        odds.append({"country": c["country"], "year": y, "p_first": p, "p_by_year": round(float(q), 4)})
pd.DataFrame(odds).to_csv(OUT / "crossover_odds.csv", index=False)

# 5. solar / coal ratio: observed annual means and the simulated fan to 2050
rp = []
for c in countries:
    if "ratio_fan" not in c:
        continue
    for y, r in zip(c["ratio_hist"]["years"], c["ratio_hist"]["ratio"]):
        rp.append({"country": c["country"], "year": y, "kind": "observed", "ratio": r, "p10": None, "p90": None})
    f = c["ratio_fan"]
    for y, a, m, b in zip(f["years"], f["p10"], f["p50"], f["p90"]):
        rp.append({"country": c["country"], "year": y, "kind": "simulated", "ratio": m, "p10": a, "p90": b})
pd.DataFrame(rp).to_csv(OUT / "ratio_path.csv", index=False)

# 6. India non-fossil capacity vs the 500 GW target
t = res["india_target"]
cap = [{"year": y, "scenario": "Observed", "gw": g, "p10": None, "p90": None} for y, g in zip(t["history"]["years"], t["history"]["gw"])]
tgt = []
for k, s in t["scenarios"].items():
    cap += [{"year": y, "scenario": k, "gw": m, "p10": a, "p90": b} for y, m, a, b in zip(s["years"], s["median"], s["p10"], s["p90"])]
    tgt.append({"scenario": k, "model": s["model"], "p_hit": round(s["p_hit"], 4), "median_2030": s["median"][-1],
                "p10_2030": s["p10"][-1], "p90_2030": s["p90"][-1]})
pd.DataFrame(cap).to_csv(OUT / "india_capacity.csv", index=False)
growth = [{"period": "Needed, 2025-2030", "cagr_pct": round(t["required_cagr"], 2)}] + \
         [{"period": f"Achieved, {k}", "cagr_pct": round(v, 2)} for k, v in t["cagr"].items()]
pd.DataFrame(tgt).to_csv(OUT / "india_target.csv", index=False)
pd.DataFrame(growth).to_csv(OUT / "india_growth.csv", index=False)

# 7. calibration: did the held-out 12 months fall inside the 80% interval?
cal = []
NAMES = {"Solar": "Solar generation", "Coal": "Coal generation", "LogRatio": "Solar ÷ coal ratio"}
for c in countries:
    for k, v in c.get("calibration", {}).items():
        cal.append({"country": c["country"], "series": NAMES[k], "inside_80": v["cover80"],
                    "mape_pct": v.get("mape"), "mae_log": v.get("mae")})
pd.DataFrame(cal).round(4).to_csv(OUT / "calibration.csv", index=False)

for f in sorted(OUT.glob("*.csv")):
    print(f"{f.name:22s} {f.stat().st_size / 1e3:8.1f} KB  {sum(1 for _ in open(f)) - 1} rows")
