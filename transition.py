"""Core time-series toolkit for the Solar vs Coal project.

Shared by the analysis notebook, the Streamlit app and the dashboard build script,
so every number shown anywhere comes from the same code.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

MONTHLY_URL = "https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_monthly_global.csv"
YEARLY_URL = "https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_yearly_global.csv"
SOURCES = ["Coal", "Gas", "Other fossil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other renewables"]


# ---------------------------------------------------------------- data
def load_monthly(path):
    """Ember monthly data -> long table of countries only (Date, Area, source, TWh, share)."""
    df = pd.read_csv(path, low_memory=False, parse_dates=["Date"])
    df = df[df["Area type"] == "Country or economy"]
    return df[["Date", "Area", "ISO 3 code", "Electricity source", "Generation (TWh)", "Share of generation (%)"]]


def load_yearly(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["Area type"] == "Country or economy"]
    return df[["Year", "Area", "ISO 3 code", "Electricity source", "Generation (TWh)", "Capacity (GW)"]]


def country_panel(monthly, country, value="Generation (TWh)"):
    """Wide monthly table for one country: index = month start, columns = sources."""
    p = monthly[monthly["Area"] == country].pivot_table(index="Date", columns="Electricity source", values=value)
    p = p.asfreq("MS")
    return p


def clean_series(s, max_gap=2):
    """Missing-value policy: interpolate gaps of up to `max_gap` months (in logs), then keep the
    most recent contiguous run of strictly positive values (logs need positive data, and a
    forecast must start from the latest observation, not from an older, longer stretch)."""
    s = s.copy()
    s[s <= 0] = np.nan
    if s.notna().sum() == 0:
        return s.iloc[:0]
    s = np.exp(np.log(s).interpolate(limit=max_gap, limit_area="inside"))
    ok = s.notna()
    runs = (ok != ok.shift()).cumsum()
    last = runs[ok].iloc[-1]
    return s[(runs == last) & ok].asfreq("MS")


# ---------------------------------------------------------------- statistics
def metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = actual - pred
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE (%)": float(np.mean(np.abs(err / actual)) * 100),
    }


def stationarity_report(s, name):
    """ADF (H0: unit root) and KPSS (H0: stationary). Stationary when ADF rejects and KPSS does not."""
    s = pd.Series(s).dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    return {
        "Series": name, "ADF stat": round(adf_stat, 3), "ADF p-value": round(adf_p, 4),
        "KPSS p-value": round(kpss_p, 4),
        "Verdict": "stationary" if (adf_p < 0.05 and kpss_p >= 0.05) else "non-stationary",
    }


# ---------------------------------------------------------------- models
def _trend_for(d, D):
    # In SARIMAX a constant in a once-differenced model is a drift (linear trend in levels).
    # With two differences a constant would mean a quadratic trend, so it is left out.
    return "c" if d + D <= 1 else "n"


def fit_sarima(y, order, seasonal_order, trend=None, exog=None):
    trend = trend or _trend_for(order[1], seasonal_order[1])
    return SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def is_admissible(res, tol=1.01):
    """True when every AR and MA root (seasonal included) lies outside the unit circle.
    Unconstrained fits are much faster; candidates that land on or inside it (non-stationary or
    non-invertible, e.g. an MA coefficient of -1 from over-differencing) are discarded, as auto-ARIMA does."""
    roots = np.concatenate([np.atleast_1d(res.arroots), np.atleast_1d(res.maroots)])
    return bool(np.all(np.isfinite(roots)) and np.all(np.abs(roots) > tol))


def lockdown_dummy(index):
    """Intervention variable for India's strict COVID-19 lockdown (Apr-May 2020); zero elsewhere and in the future."""
    return pd.DataFrame({"lockdown": index.isin(pd.to_datetime(["2020-04-01", "2020-05-01"])).astype(float)}, index=index)


def seasonal_strength(y, m=12):
    """Hyndman's F_s = max(0, 1 - Var(R) / Var(S + R)) from an STL decomposition."""
    from statsmodels.tsa.seasonal import STL
    stl = STL(pd.Series(y).dropna(), period=m, robust=True).fit()
    return max(0.0, 1 - stl.resid.var() / (stl.resid + stl.seasonal).var())


def differencing_candidates(y, m=12):
    """All (d, D) schemes from {(1,0), (0,1), (1,1)} whose differenced series passes ADF and KPSS.
    The tests often accept more than one scheme; time-series cross-validation then decides."""
    out = []
    for d, D in [(1, 0), (0, 1), (1, 1)]:
        z = y.diff(m) if D else y
        z = z.diff() if d else z
        if stationarity_report(z, "")["Verdict"] == "stationary":
            out.append((d, D))
    return out or [(1, 1)]


def sarima_grid(y, d, D, p=(0, 1, 2), q=(0, 1, 2), P=(0, 1), Q=(0, 1), m=12, exog=None):
    """Fit every candidate SARIMA with fixed differencing; return a table sorted by AIC (BIC alongside)."""
    rows = []
    for pp, qq, PP, QQ in itertools.product(p, q, P, Q):
        dd, DD = d, D
        try:
            r = fit_sarima(y, (pp, dd, qq), (PP, DD, QQ, m), exog=exog)
        except Exception:
            continue
        if not np.isfinite(r.aic) or not is_admissible(r):
            continue
        rows.append({"order": (pp, dd, qq), "seasonal_order": (PP, DD, QQ, m),
                     "model": f"SARIMA({pp},{dd},{qq})({PP},{DD},{QQ})[{m}]",
                     "AIC": r.aic, "BIC": r.bic, "k": len(r.params)})
    cols = ["order", "seasonal_order", "model", "AIC", "BIC", "k"]
    return pd.DataFrame(rows, columns=cols).sort_values("AIC").reset_index(drop=True)


def seasonal_naive(train, h, m=12):
    vals = np.tile(train.iloc[-m:].values, h // m + 1)[:h]
    return pd.Series(vals, index=pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS"))


def holt_winters(train, h, m=12):
    fit = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m).fit()
    return fit.forecast(h)


def rolling_origin_cv(y, forecasters, h=12, n_origins=12, step=1, min_train=48):
    """Expanding-window backtest. forecasters: {name: f(train, h) -> forecast Series}.
    Returns a long table with one row per (model, origin, horizon)."""
    rows = []
    last_origin = len(y) - h
    origins = [o for o in range(last_origin - (n_origins - 1) * step, last_origin + 1, step) if o >= min_train]
    for o in origins:
        train, test = y.iloc[:o], y.iloc[o:o + h]
        for name, f in forecasters.items():
            try:
                fc = np.asarray(f(train, h))[:len(test)]
            except Exception:
                continue
            for k, (a, p_) in enumerate(zip(test.values, fc), start=1):
                rows.append({"model": name, "origin": train.index[-1], "horizon": k, "actual": a, "pred": p_})
    out = pd.DataFrame(rows, columns=["model", "origin", "horizon", "actual", "pred"])
    out["ae"] = (out["actual"] - out["pred"]).abs()
    out["ape"] = out["ae"] / out["actual"].abs() * 100
    return out


# ---------------------------------------------------------------- crossover simulation
def log_ratio(solar, coal):
    """r_t = log(solar_t / coal_t). Solar beats coal in a month exactly when r_t > 0."""
    s = pd.concat([solar, coal], axis=1).dropna()
    s = s[(s.iloc[:, 0] > 0) & (s.iloc[:, 1] > 0)]
    return np.log(s.iloc[:, 0] / s.iloc[:, 1]).asfreq("MS")


def simulate(res, steps, n_paths=2000, seed=0, drift_uncertainty=True, n_draws=100):
    """Future sample paths from a fitted SARIMAX (shape steps x n_paths).
    Exogenous intervention dummies are zero in the future.

    A plain simulation treats the estimated parameters as exact. Over decades the dominant
    uncertainty is the drift (the `intercept`), so when the model has one we draw it from its
    sampling distribution N(c, se^2), re-filter, and simulate an equal share of paths per draw."""
    k = res.model.k_exog
    exog = np.zeros((steps, k)) if k else None
    rng = np.random.default_rng(seed)
    if not (drift_uncertainty and "intercept" in res.params.index and np.isfinite(res.bse["intercept"])):
        sims = res.simulate(nsimulations=steps, repetitions=n_paths, anchor="end", exog=exog, rng=rng)
        return np.asarray(sims).reshape(steps, n_paths)
    i = list(res.params.index).index("intercept")
    per = int(np.ceil(n_paths / n_draws))
    out = []
    for c in rng.normal(res.params["intercept"], res.bse["intercept"], n_draws):
        p = res.params.values.copy()
        p[i] = c
        r = res.model.filter(p)
        out.append(np.asarray(r.simulate(nsimulations=steps, repetitions=per, anchor="end", exog=exog, rng=rng)).reshape(steps, per))
    return np.hstack(out)[:, :n_paths]


def crossover_distribution(r, res, end_year=2050, n_paths=2000, seed=0):
    """First-passage analysis on simulated log-ratio paths.
    Returns per-path crossover years (NaN = not before end_year) for two definitions:
      'month': first month in which solar > coal
      'year' : first calendar year in which solar beats coal on average (mean log-ratio > 0)."""
    start = r.index[-1] + pd.offsets.MonthBegin()
    idx = pd.date_range(start, f"{end_year}-12-01", freq="MS")
    paths = simulate(res, len(idx), n_paths, seed)
    sims = pd.DataFrame(paths, index=idx)

    above = sims.values > 0
    first_month = np.where(above.any(axis=0), above.argmax(axis=0), -1)
    month_year = np.array([idx[i].year + (idx[i].month - 1) / 12 if i >= 0 else np.nan for i in first_month])

    # Calendar years: combine the observed months of the current year with simulated ones.
    hist_part = r[r.index.year == start.year]
    yearly = []
    for year in range(start.year, end_year + 1):
        block = sims[sims.index.year == year]
        if year == start.year and len(hist_part):
            block = pd.concat([pd.DataFrame(np.repeat(hist_part.values[:, None], n_paths, axis=1), index=hist_part.index), block])
        yearly.append(block.mean().values > 0)
    yearly = np.array(yearly)
    years = np.arange(start.year, end_year + 1)
    first_year = np.where(yearly.any(axis=0), years[yearly.argmax(axis=0)], np.nan)

    return {"paths": sims, "month": month_year, "year": first_year}


def summarise_crossover(years, end_year=2050):
    years = np.asarray(years, float)
    p_any = np.mean(~np.isnan(years))
    out = {"P(before %d)" % end_year: p_any}
    for target in (2030, 2035, 2040):
        out[f"P(by {target})"] = float(np.mean(years <= target))
    if p_any >= 0.5:
        q = np.quantile(np.where(np.isnan(years), np.inf, years), [0.1, 0.5, 0.9], method="inverted_cdf")
        out.update({"median": q[1], "p10": q[0], "p90": q[2]})
    else:
        out.update({"median": np.nan, "p10": np.nan, "p90": np.nan})
    return out


# ---------------------------------------------------------------- automatic selection
def identity(x):
    return x


def sarima_forecaster(order, seasonal_order, exog_fn=None, transform=np.log, back=np.exp):
    """Forecast function for rolling_origin_cv: fits on the transformed series, returns the original scale."""
    def f(train, h):
        z = transform(train)
        X = exog_fn(z.index) if exog_fn else None
        Xf = np.zeros((h, X.shape[1])) if X is not None else None
        return back(fit_sarima(z, order, seasonal_order, exog=X).forecast(h, exog=Xf))
    return f


def auto_sarima(y, exog_fn=None, small=False, h=12, n_origins=6, transform=np.log, back=np.exp, score="ape"):
    """Two-stage selection.
    1) For every differencing scheme the stationarity tests accept, pick lags (p,q,P,Q) by AIC.
    2) Compare those finalists with rolling-origin cross-validation on the original scale
       (score='ape' -> MAPE, 'ae' -> MAE) and keep the most accurate one.
    AIC is never compared across different d/D because differencing changes the likelihood's sample."""
    z = transform(y)
    X = exog_fn(z.index) if exog_fn else None
    grid = dict(p=(0, 1), q=(0, 1)) if small else {}
    finalists, tables = [], {}
    for d, D in differencing_candidates(z):
        g = sarima_grid(z, d, D, exog=X, **grid)
        if g.empty:
            continue
        tables[(d, D)] = g
        finalists.append(g.iloc[0])
    scores = {}
    for b in finalists:
        f = sarima_forecaster(b.order, b.seasonal_order, exog_fn, transform, back)
        cv = rolling_origin_cv(y, {b.model: f}, h=h, n_origins=n_origins, step=max(1, 12 // n_origins),
                               min_train=min(48, len(y) - h - n_origins))
        scores[b.model] = float(cv[score].mean()) if len(cv) else np.inf
    best = min(finalists, key=lambda b: scores[b.model])
    return {"order": best.order, "seasonal_order": best.seasonal_order, "model": best.model,
            "cv": scores, "tables": tables}


# ---------------------------------------------------------------- annual capacity vs target
def annual_arima(z, d, max_pq=2):
    """Best ARIMA(p,d,q) by AIC for an annual series with fixed d (drift included when d = 1)."""
    best = None
    for p, q in itertools.product(range(max_pq + 1), range(max_pq + 1)):
        try:
            r = SARIMAX(z, order=(p, d, q), trend="c" if d == 1 else "n").fit(disp=False)
        except Exception:
            continue
        if np.isfinite(r.aic) and (best is None or r.aic < best.aic):
            best = r
    return best


def target_analysis(capacity, target, target_year, d, n_paths=5000, seed=0):
    """Probability that capacity reaches `target` by `target_year` under an ARIMA on log capacity.
    d = 1: growth fluctuates around its long-run average (drift).
    d = 2: the growth rate itself wanders, so recent acceleration carries forward."""
    z = np.log(capacity.astype(float))
    z.index = pd.period_range(str(capacity.index[0]), periods=len(z), freq="Y")
    res = annual_arima(z, d)
    steps = target_year - int(capacity.index[-1])
    sims = np.exp(simulate(res, steps, n_paths, seed))
    years = list(range(int(capacity.index[-1]) + 1, target_year + 1))
    fan = pd.DataFrame({"p10": np.quantile(sims, 0.1, axis=1), "median": np.median(sims, axis=1),
                        "p90": np.quantile(sims, 0.9, axis=1)}, index=years)
    return {"model": f"ARIMA{res.model.order} on log capacity" + (" + drift" if d == 1 else ""),
            "aic": res.aic, "fan": fan, "final": sims[-1], "p_hit": float(np.mean(sims[-1] >= target)),
            "result": res}


def observed_crossover_year(r):
    """First complete calendar year in which solar out-generated coal on average (None if never)."""
    yearly = r.groupby(r.index.year).agg(["mean", "size"])
    hit = yearly[(yearly["size"] == 12) & (yearly["mean"] > 0)]
    return int(hit.index[0]) if len(hit) else None
