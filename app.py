"""Solar vs Coal: interactive time-series app.

    streamlit run app.py

Live tabs refit models on demand with the same code as the notebook (transition.py);
the global tabs read the batch results written by build_results.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

import transition as T

ROOT = Path(__file__).parent
st.set_page_config(page_title="Solar vs Coal", page_icon="☀️", layout="wide")

COL = {"Coal": "#2a78d6", "Solar": "#eb6834", "Wind": "#1baf7a", "Fossil (gas & other)": "#eda100",
       "Hydro": "#e87ba4", "Other clean": "#008300", "Total generation": "#6250d6"}
INK, MUTED, GRID = "#52514e", "#898781", "rgba(137,135,129,0.25)"
MIX = {"Coal": ["Coal"], "Solar": ["Solar"], "Wind": ["Wind"], "Fossil (gas & other)": ["Gas", "Other fossil"],
       "Hydro": ["Hydro"], "Other clean": ["Nuclear", "Bioenergy", "Other renewables"]}


def style(fig, h=380, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), **kw)
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(show_spinner="Loading Ember data…")
def load():
    m = T.load_monthly(ROOT / "data" / "ember_monthly_global.csv")
    res = json.loads((ROOT / "results" / "results.json").read_text())
    return m, res


@st.cache_data(show_spinner=False)
def panel(country):
    p = T.country_panel(MONTHLY, country)
    for s in ["Coal", "Solar", "Total generation"]:   # a source a country never used is absent from the file
        if s not in p:
            p[s] = 0.0
    return p


@st.cache_data(show_spinner="Selecting the model (AIC per differencing scheme, then cross-validation)…")
def auto_select(country, source, n, use_dummy):
    y = T.clean_series(panel(country)[source]).iloc[:n]
    sel = T.auto_sarima(y, exog_fn=T.lockdown_dummy if use_dummy else None, small=True)
    tables = {f"d={d}, D={D}": t[["model", "AIC", "BIC"]].head(8) for (d, D), t in sel["tables"].items()}
    return sel["order"], sel["seasonal_order"], sel["cv"], tables


MONTHLY, RESULTS = load()
BY_NAME = {c["country"]: c for c in RESULTS["countries"]}
countries = sorted(MONTHLY["Area"].unique())

# ------------------------------------------------------------------ sidebar
st.sidebar.title("☀️ Solar vs Coal")
country = st.sidebar.selectbox("Country", countries, index=countries.index("India"))
st.sidebar.caption(f"Data: Ember monthly electricity data (CC-BY-4.0), through {RESULTS['data_through']}. "
                   "Models: SARIMA / SARIMAX with statsmodels.")
p = panel(country)
rec = BY_NAME.get(country, {})

st.title(f"{country}: the race between solar and coal")
tab_over, tab_lab, tab_cross, tab_target, tab_world = st.tabs(
    ["Overview", "Forecast lab (live)", "Solar overtakes coal?", "India 2030 target", "All countries"])

# ------------------------------------------------------------------ overview
with tab_over:
    last12 = p.iloc[-12:].sum()
    tot = last12.get("Total generation", np.nan)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coal share, last 12 months", f"{last12.get('Coal', 0) / tot * 100:.1f}%")
    c2.metric("Solar share, last 12 months", f"{last12.get('Solar', 0) / tot * 100:.1f}%")
    if rec.get("status") == "crossed":
        c3.metric("Solar vs coal", "Solar ahead", f"solar = {rec['last12_ratio']:.2f}× coal")
    elif rec.get("status") == "forecast":
        y = rec["crossover"]["year"]
        c3.metric("Solar overtakes coal (median year)", "after 2050" if y["median"] is None else f"{y['median']:.0f}",
                  f"P(by 2035) = {y['P(by 2035)']:.0%}", delta_color="off")
    else:
        c3.metric("Solar vs coal", {"no-coal": "Little or no coal", "too-short": "Too little solar data"}.get(rec.get("status"), "n/a"))
    c4.metric("Months of data", f"{len(p)}", f"{p.index[0]:%b %Y} – {p.index[-1]:%b %Y}", delta_color="off")

    fig = go.Figure()
    for grp, cols in MIX.items():
        v = p[[c for c in cols if c in p]].sum(axis=1, min_count=1)
        fig.add_trace(go.Scatter(x=p.index, y=v, name=grp, stackgroup="mix", line=dict(width=0.5, color=COL[grp]),
                                 hovertemplate="%{y:.1f} TWh"))
    st.plotly_chart(style(fig, 420, title="Monthly generation by source (TWh)"), width="stretch")

    share = p[["Coal", "Solar"]].div(p["Total generation"], axis=0).mul(100).rolling(12).mean()
    fig = go.Figure([go.Scatter(x=share.index, y=share[s], name=s, line=dict(color=COL[s], width=2),
                                hovertemplate="%{y:.1f}%") for s in ["Coal", "Solar"]])
    st.plotly_chart(style(fig, 320, title="Share of generation, 12-month rolling (%)"), width="stretch")

# ------------------------------------------------------------------ forecast lab
with tab_lab:
    st.markdown("Fit a SARIMA model **live** on any series. Choose automatic selection or set the orders yourself; "
                "the last months are held out to test accuracy against Holt-Winters and a seasonal-naive baseline.")
    srcs = [s for s in ["Coal", "Solar", "Wind", "Hydro", "Gas", "Nuclear", "Total generation"]
            if s in p and len(T.clean_series(p[s])) >= 48]
    a, b, c = st.columns(3)
    source = a.selectbox("Series", srcs, index=srcs.index("Solar") if "Solar" in srcs else 0)
    hold = b.slider("Hold-out months", 6, 24, 12)
    horizon = c.slider("Forecast horizon (months)", 6, 36, 24)
    mode = st.radio("Model orders", ["Automatic (AIC/BIC + cross-validation)", "Manual"], horizontal=True)
    dummy = st.checkbox("Add COVID-lockdown intervention dummy (Apr–May 2020)", value=country == "India" and source != "Solar")

    y = T.clean_series(p[source])
    if len(y) < 48:
        st.warning("This series is too short for seasonal modelling (needs ≥ 48 months).")
    else:
        z = np.log(y)
        train = y.iloc[:-hold]

        if mode.startswith("Automatic"):
            order, sorder, cv, tables = auto_select(country, source, len(train), dummy)
            st.success(f"Selected **SARIMA{order}{sorder[:3]}[12]**: cross-validated MAPE per finalist: "
                       + ", ".join(f"{k} → {v:.2f}%" for k, v in cv.items()))
            with st.expander("AIC / BIC tables (lags chosen within each differencing scheme)"):
                for k, t in tables.items():
                    st.markdown(f"**{k}**")
                    st.dataframe(t.round(2), hide_index=True, width="stretch")
        else:
            cols = st.columns(6)
            o = [cols[i].number_input(n, 0, 2 if i in (0, 2) else 1, v) for i, (n, v) in
                 enumerate([("p", 1), ("d", 1), ("q", 1), ("P", 0), ("D", 1), ("Q", 1)])]
            order, sorder = (o[0], o[1], o[2]), (o[3], o[4], o[5], 12)

        ex = T.lockdown_dummy if dummy else None
        with st.spinner("Fitting…"):
            f_sarima = T.sarima_forecaster(order, sorder, ex)(train, hold)
            f_hw = np.exp(T.holt_winters(np.log(train), hold))
            f_sn = T.seasonal_naive(train, hold)
            X = ex(z.index) if ex else None
            full = T.fit_sarima(z, order, sorder, exog=X)
            fc = full.get_forecast(horizon, exog=np.zeros((horizon, X.shape[1])) if X is not None else None)
            mid, ci80, ci95 = np.exp(fc.predicted_mean), np.exp(fc.conf_int(alpha=0.2)), np.exp(fc.conf_int(alpha=0.05))

        test = y.iloc[-hold:]
        met = pd.DataFrame({"SARIMA": T.metrics(test, f_sarima), "Holt-Winters": T.metrics(test, f_hw),
                            "Seasonal naive": T.metrics(test, f_sn)}).T.round(2)
        left, right = st.columns([1.6, 1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y.index, y=y, name="Observed", line=dict(color=INK, width=1.6)))
            fig.add_trace(go.Scatter(x=list(ci95.index) + list(ci95.index[::-1]), y=list(ci95.iloc[:, 1]) + list(ci95.iloc[:, 0][::-1]),
                                     fill="toself", fillcolor="rgba(235,104,52,0.12)", line=dict(width=0), name="95% interval", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=list(ci80.index) + list(ci80.index[::-1]), y=list(ci80.iloc[:, 1]) + list(ci80.iloc[:, 0][::-1]),
                                     fill="toself", fillcolor="rgba(235,104,52,0.25)", line=dict(width=0), name="80% interval", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=mid.index, y=mid, name="Forecast", line=dict(color="#eb6834", width=2.2)))
            fig.add_trace(go.Scatter(x=test.index, y=f_sarima, name="SARIMA on hold-out", line=dict(color="#e34948", width=1.6)))
            fig.add_trace(go.Scatter(x=test.index, y=f_hw, name="Holt-Winters on hold-out", line=dict(color="#6250d6", width=1.2)))
            st.plotly_chart(style(fig, 430, title=f"{source}: hold-out test and {horizon}-month forecast (TWh)"), width="stretch")
        with right:
            st.markdown("**Hold-out accuracy**")
            st.dataframe(met, width="stretch")
            lb = acorr_ljungbox(full.resid.iloc[13 + 12 * sorder[1]:], lags=[12], return_df=True)["lb_pvalue"].iloc[0]
            st.markdown(f"**AIC** {full.aic:.1f} · **BIC** {full.bic:.1f}  \n**Ljung-Box p (lag 12)** = {lb:.3f} "
                        + ("✅ residuals look like white noise" if lb > 0.05 else "⚠️ autocorrelation left in residuals"))
            coef = pd.DataFrame({"coef": full.params, "p-value": full.pvalues}).round(4)
            st.dataframe(coef, width="stretch")

        with st.expander("Stationarity tests and ACF / PACF"):
            rows = []
            for d_, D_ in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                w = z.diff(12) if D_ else z
                w = w.diff() if d_ else w
                rows.append({**T.stationarity_report(w, f"log {source}"), "d": d_, "D": D_})
            st.dataframe(pd.DataFrame(rows)[["d", "D", "ADF p-value", "KPSS p-value", "Verdict"]], hide_index=True)
            w = z
            w = w.diff(12) if sorder[1] else w
            w = w.diff() if order[1] else w
            w = w.dropna()
            nl = min(36, len(w) // 2 - 1)
            fig = make_subplots(1, 2, subplot_titles=(f"ACF (d={order[1]}, D={sorder[1]})", "PACF"))
            for i, vals in enumerate([acf(w, nlags=nl)[1:], pacf(w, nlags=nl, method="ywm")[1:]], start=1):
                fig.add_trace(go.Bar(x=list(range(1, nl + 1)), y=vals, marker_color="#2a78d6", showlegend=False), 1, i)
                band = 1.96 / np.sqrt(len(w))
                for s_ in (band, -band):
                    fig.add_hline(y=s_, line=dict(color=MUTED, width=1), row=1, col=i)
            st.plotly_chart(style(fig, 300), width="stretch")

# ------------------------------------------------------------------ crossover
with tab_cross:
    st.markdown(r"We model $r_t=\log(\text{solar}_t/\text{coal}_t)$ with SARIMA and simulate thousands of future paths. "
                "Solar beats coal when $r_t>0$; the share of paths that have crossed by each year is the probability.")
    if "Coal" not in p or (p["Coal"].iloc[-12:].sum() / p["Total generation"].iloc[-12:].sum()) < 0.01:
        st.info(f"{country} generates (almost) no electricity from coal, so there is no crossover to forecast.")
    else:
        n_paths = st.select_slider("Simulated paths", [500, 1000, 2000, 5000], value=2000)
        r = np.log(T.clean_series(np.exp(T.log_ratio(p["Solar"], p["Coal"]))))
        if len(r) < 48:
            st.warning("Not enough continuous months with both solar and coal to model the ratio.")
        else:
            with st.spinner("Selecting the ratio model and simulating…"):
                sel = T.auto_sarima(r, small=True, transform=T.identity, back=T.identity, score="ae")
                res = T.fit_sarima(r, sel["order"], sel["seasonal_order"])
                cx = T.crossover_distribution(r, res, end_year=2050, n_paths=n_paths)
            summ = T.summarise_crossover(cx["year"])
            obs = T.observed_crossover_year(r)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Model", sel["model"].replace("SARIMA", ""))
            k2.metric("Median crossover year", "already" if np.exp(r.iloc[-12:].mean()) > 1 else
                      ("after 2050" if np.isnan(summ["median"]) else f"{summ['median']:.0f}"))
            k3.metric("P(by 2030)", f"{summ['P(by 2030)']:.0%}")
            k4.metric("P(by 2035)", f"{summ['P(by 2035)']:.0%}")
            if obs:
                st.caption(f"Solar first out-generated coal over a full calendar year in {obs}.")
            ann = cx["paths"].groupby(cx["paths"].index.year).mean()
            hist = r.groupby(r.index.year).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(ann.index) + list(ann.index[::-1]),
                                     y=list(np.exp(ann.quantile(.9, axis=1))) + list(np.exp(ann.quantile(.1, axis=1))[::-1]),
                                     fill="toself", fillcolor="rgba(235,104,52,0.2)", line=dict(width=0), name="80% of paths"))
            fig.add_trace(go.Scatter(x=ann.index, y=np.exp(ann.median(axis=1)), name="Median path", line=dict(color="#eb6834", width=2.2)))
            fig.add_trace(go.Scatter(x=hist.index, y=np.exp(hist), name="Observed", mode="lines+markers", line=dict(color=INK)))
            fig.add_hline(y=1, line=dict(color="#2a78d6", width=1.5), annotation_text="solar = coal")
            fig.update_yaxes(type="log", title="solar ÷ coal")
            st.plotly_chart(style(fig, 400, title="Solar ÷ coal generation (annual mean, log scale)"), width="stretch")
            yrs = cx["year"]
            b = np.arange(int(np.nanmin(yrs)) if np.isfinite(np.nanmin(yrs)) else 2026, 2051)
            fig = go.Figure(go.Bar(x=b, y=[np.mean(yrs == v) for v in b], marker_color="#eb6834",
                                   hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style(fig, 300, title="Probability that solar first beats coal (annual) in each year"), width="stretch")

# ------------------------------------------------------------------ India target
with tab_target:
    t = RESULTS.get("india_target")
    if not t:
        st.info("Run build_results.py to compute the target analysis.")
    else:
        st.markdown(f"India's goal is **{t['target']} GW of non-fossil capacity by {t['target_year']}**. Hitting it needs "
                    f"**{t['required_cagr']:.1f}% growth a year**; history: "
                    + ", ".join(f"{k}: {v:.1f}%/yr" for k, v in t["cagr"].items()) + ".")
        cols = st.columns(len(t["scenarios"]))
        for col, (k, s) in zip(cols, t["scenarios"].items()):
            col.metric(f"{k}: P(≥ {t['target']} GW)", f"{s['p_hit']:.0%}", f"median {s['median'][-1]:.0f} GW in {t['target_year']}", delta_color="off")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t["history"]["years"], y=t["history"]["gw"], name="Observed", mode="lines+markers", line=dict(color=INK)))
        x0, y0 = t["history"]["years"][-1], t["history"]["gw"][-1]
        for (k, s), c, rgba in zip(t["scenarios"].items(), ["#2a78d6", "#eb6834"], ["rgba(42,120,214,0.18)", "rgba(235,104,52,0.18)"]):
            xs = [x0] + s["years"]
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[y0] + s["p90"] + ([y0] + s["p10"])[::-1], fill="toself",
                                     fillcolor=rgba, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xs, y=[y0] + s["median"], name=f"{k} ({s['model']})", line=dict(color=c, width=2.2)))
        fig.add_hline(y=t["target"], line=dict(color="#e34948", width=1.5), annotation_text=f"{t['target']} GW target")
        st.plotly_chart(style(fig, 440, title="Non-fossil installed capacity (GW), median and 80% interval"), width="stretch")
        st.caption("Long-run trend = ARIMA with drift on log capacity (growth reverts to its 25-year average). "
                   "Accelerating = ARIMA with d = 2 (the growth rate itself can keep rising). Source: Ember yearly data.")

# ------------------------------------------------------------------ world
with tab_world:
    rows = []
    for c in RESULTS["countries"]:
        if c.get("status") not in ("forecast", "crossed"):
            continue
        y = c["crossover"]["year"]
        rows.append({"Country": c["country"], "Coal %": c["share12"].get("Coal"), "Solar %": c["share12"].get("Solar"),
                     "Solar ÷ coal": c["last12_ratio"],
                     "Median year solar > coal": "already" if c["status"] == "crossed" else (str(int(y["median"])) if y["median"] else "after 2050"),
                     "P(by 2030)": y["P(by 2030)"], "P(by 2035)": y["P(by 2035)"]})
    df = pd.DataFrame(rows).sort_values("Solar ÷ coal", ascending=False)
    st.markdown(f"**{len(df)} countries** with meaningful coal generation, ranked by how close solar is to coal (last 12 months).")
    st.dataframe(df, hide_index=True, width="stretch", height=520,
                 column_config={"P(by 2030)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                "P(by 2035)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    cal = pd.DataFrame([{"Series": k, "cover": v["cover80"]} for c in RESULTS["countries"] for k, v in c.get("calibration", {}).items()])
    if len(cal):
        st.markdown("**Calibration check:** share of held-out months that fell inside the 80% prediction interval "
                    "(a well-calibrated model scores ≈ 80%).")
        st.dataframe(cal.groupby("Series").cover.agg(["count", "mean"]).rename(columns={"count": "countries", "mean": "coverage"}).round(3))
