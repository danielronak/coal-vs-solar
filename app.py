"""Solar vs Coal: interactive time-series app.

    streamlit run app.py

Live tabs refit models on demand with the same code as the notebook (transition.py);
the global tabs read the batch results written by build_results.py.
"""
import html
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
       "Hydro": "#e87ba4", "Other clean": "#008300"}
INK, INK2, MUTED, GRID, AXIS = "#1f1e1c", "#52514e", "#898781", "#e9e7e1", "#c3c2b7"
FONT = "Arial, Helvetica, sans-serif"   # a font every browser has, so Plotly measures legends correctly
MIX = {"Coal": ["Coal"], "Solar": ["Solar"], "Wind": ["Wind"], "Fossil (gas & other)": ["Gas", "Other fossil"],
       "Hydro": ["Hydro"], "Other clean": ["Nuclear", "Bioenergy", "Other renewables"]}

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }
  h1 { font-size: 2rem !important; font-weight: 700 !important; margin-bottom: 0 !important; }
  .sc-sub { color: #52514e; font-size: 1rem; margin: 0.15rem 0 1.4rem; }
  .sc-tile { border: 1px solid #e3e1da; border-radius: 10px; padding: 0.9rem 1.05rem; background: #ffffff; min-height: 8.4rem; }
  .sc-tile .lab { color: #6b6a65; font-size: 0.82rem; line-height: 1.25; min-height: 2.5em; }
  .sc-tile .val { color: #1f1e1c; font-size: 1.75rem; font-weight: 600; line-height: 1.2; margin-top: 0.2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .sc-tile .val.sm { font-size: 1.3rem; padding-block: 0.22rem; }
  .sc-tile .note { color: #898781; font-size: 0.8rem; margin-top: 0.15rem; }
  .sc-h { font-size: 1.08rem; font-weight: 600; color: #1f1e1c; margin: 1.6rem 0 0.1rem; }
  .sc-cap { color: #6b6a65; font-size: 0.86rem; margin: 0 0 0.3rem; }
  .sc-good { color: #0f7b0f; } .sc-warn { color: #b25e00; }
  div[data-testid="stTabs"] button p { font-size: 0.95rem; }
  div[data-testid="stExpander"] details { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ presentation helpers
def tiles(items):
    """A row of equal cards: (label, value, note). Long values get a smaller size so nothing wraps."""
    for col, (label, value, note) in zip(st.columns(len(items)), items):
        size = " sm" if len(value) > 12 else ""
        col.markdown(f'<div class="sc-tile"><div class="lab">{html.escape(label)}</div>'
                     f'<div class="val{size}" title="{html.escape(value)}">{html.escape(value)}</div>'
                     f'<div class="note">{html.escape(note or "")}&nbsp;</div></div>', unsafe_allow_html=True)


def heading(title, caption=None):
    st.markdown(f'<div class="sc-h">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="sc-cap">{html.escape(caption)}</div>', unsafe_allow_html=True)


def show(fig, h=380, legend=True, hover="x unified"):
    for tr in fig.data:   # Plotly adds markers to short scatter traces unless a mode is set
        if tr.type == "scatter" and tr.mode is None:
            tr.mode = "lines"
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=36 if legend else 8, b=8), template="none",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode=hover,
        font=dict(family=FONT, size=13, color=INK2), showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", traceorder="normal",
                    font=dict(size=12.5, color=INK2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, size=12.5, color=INK)),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=AXIS, ticks="outside", tickcolor=AXIS, zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, showline=False, zeroline=False, ticks="", automargin=True)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def band(fig, x, lo, hi, rgba, name, legend=True):
    fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=rgba,
                             line=dict(width=0), name=name, hoverinfo="skip", showlegend=legend))


def year_str(v):
    return "after 2050" if v is None or not np.isfinite(v) else f"{v:.0f}"


# ------------------------------------------------------------------ data
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
    tables = {f"d = {d}, D = {D}": t[["model", "AIC", "BIC"]].head(8) for (d, D), t in sel["tables"].items()}
    return sel["order"], sel["seasonal_order"], sel["cv"], tables


@st.cache_data(show_spinner="Selecting the ratio model and simulating future paths…")
def crossover(country, n_paths):
    p = panel(country)
    r = np.log(T.clean_series(np.exp(T.log_ratio(p["Solar"], p["Coal"]))))
    if len(r) < 48:
        return None
    sel = T.auto_sarima(r, small=True, transform=T.identity, back=T.identity, score="ae")
    res = T.fit_sarima(r, sel["order"], sel["seasonal_order"])
    cx = T.crossover_distribution(r, res, end_year=2050, n_paths=n_paths)
    ann = cx["paths"].groupby(cx["paths"].index.year).mean()
    return {"model": sel["model"], "years": cx["year"], "summary": T.summarise_crossover(cx["year"]),
            "observed": T.observed_crossover_year(r), "last12": float(np.exp(r.iloc[-12:].mean())),
            "hist": np.exp(r.groupby(r.index.year).mean()),
            "fan": pd.DataFrame({"p10": np.exp(ann.quantile(.1, axis=1)), "p50": np.exp(ann.median(axis=1)),
                                 "p90": np.exp(ann.quantile(.9, axis=1))})}


MONTHLY, RESULTS = load()
BY_NAME = {c["country"]: c for c in RESULTS["countries"]}
countries = sorted(MONTHLY["Area"].unique())

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Solar vs Coal")
    country = st.selectbox("Country", countries, index=countries.index("India"))
    st.caption(f"Data: Ember monthly and yearly electricity data (CC-BY-4.0), through {RESULTS['data_through']}.")
    st.caption("Models: SARIMA on log generation, selected by AIC within each differencing scheme and by "
               "time-series cross-validation across schemes.")
p = panel(country)
rec = BY_NAME.get(country, {})

st.title(f"{country}: solar vs coal")
st.markdown(f'<div class="sc-sub">Monthly electricity generation, forecasts and the odds that solar overtakes coal · '
            f'{p.index[0]:%b %Y} to {p.index[-1]:%b %Y}</div>', unsafe_allow_html=True)
tab_over, tab_lab, tab_cross, tab_target, tab_world = st.tabs(
    ["Overview", "Forecast lab", "Solar overtakes coal?", "India 2030 target", "All countries"])

# ------------------------------------------------------------------ overview
with tab_over:
    last12, prev12 = p.iloc[-12:].sum(), p.iloc[-24:-12].sum()
    share = lambda s, w: w.get(s, 0) / w["Total generation"] * 100
    dc, ds = share("Coal", last12) - share("Coal", prev12), share("Solar", last12) - share("Solar", prev12)
    status = rec.get("status")
    if status == "crossed":
        third = ("Solar vs coal, last 12 months", f"{rec['last12_ratio']:.2f}× coal", "solar already generates more")
    elif status == "forecast":
        y = rec["crossover"]["year"]
        third = ("Most likely first year solar beats coal", year_str(y["median"]),
                 f"{y['P(by 2035)']:.0%} chance by 2035")
    else:
        third = ("Solar vs coal", {"no-coal": "No coal to beat", "too-short": "Not modelled"}.get(status, "n/a"),
                 {"too-short": "too little solar data"}.get(status, ""))
    tiles([("Coal share of generation, last 12 months", f"{share('Coal', last12):.1f}%", f"{dc:+.1f} points vs the year before"),
           ("Solar share of generation, last 12 months", f"{share('Solar', last12):.1f}%", f"{ds:+.1f} points vs the year before"),
           third,
           ("Total generation, last 12 months", f"{last12['Total generation']:,.0f} TWh",
            f"{(last12['Total generation'] / prev12['Total generation'] - 1) * 100:+.1f}% vs the year before")])

    heading("Monthly generation by source", "Terawatt-hours, stacked. Coal at the bottom, then solar, wind, gas and other fossil, hydro, and other clean sources.")
    fig = go.Figure()
    for grp, cols in MIX.items():
        v = p[[c for c in cols if c in p]].sum(axis=1, min_count=1).fillna(0)
        fig.add_trace(go.Scatter(x=p.index, y=v, name=grp, stackgroup="mix", line=dict(width=0, color=COL[grp]),
                                 fillcolor=COL[grp], opacity=1, hovertemplate="%{y:,.1f} TWh"))
    fig.update_yaxes(title=None, rangemode="tozero")
    show(fig, 420)

    heading("Share of generation", "Rolling 12-month share, in percent.")
    sh = p[["Coal", "Solar"]].div(p["Total generation"], axis=0).mul(100).rolling(12).mean().dropna()
    fig = go.Figure([go.Scatter(x=sh.index, y=sh[s], name=s, line=dict(color=COL[s], width=2.4),
                                hovertemplate="%{y:.1f}%") for s in ["Coal", "Solar"]])
    fig.update_yaxes(ticksuffix="%", range=[0, min(100, float(sh.max().max()) * 1.15)])
    show(fig, 320)

# ------------------------------------------------------------------ forecast lab
with tab_lab:
    st.markdown("Fit a SARIMA model **live** on any series. The last months are held out to test it against "
                "Holt-Winters and a seasonal-naive baseline, then the model is refitted on all data to forecast.")
    srcs = [s for s in ["Coal", "Solar", "Wind", "Hydro", "Gas", "Nuclear", "Total generation"]
            if s in p and len(T.clean_series(p[s])) >= 48]
    with st.container(border=True):
        a, b, c = st.columns(3)
        source = a.selectbox("Series", srcs, index=srcs.index("Solar") if "Solar" in srcs else 0)
        hold = b.slider("Hold-out months", 6, 24, 12)
        horizon = c.slider("Forecast horizon (months)", 6, 36, 24)
        a, b = st.columns([2, 1.3])
        mode = a.radio("Model orders", ["Automatic: AIC/BIC + cross-validation", "Manual"], horizontal=True)
        dummy = b.checkbox("COVID-lockdown dummy (Apr–May 2020)", value=country == "India" and source != "Solar")
        if mode == "Manual":
            cols = st.columns(6)
            o = [cols[i].number_input(n, 0, 2 if i in (0, 2) else 1, v) for i, (n, v) in
                 enumerate([("p", 1), ("d", 1), ("q", 1), ("P", 0), ("D", 1), ("Q", 1)])]

    y = T.clean_series(p[source]) if srcs else pd.Series(dtype=float)
    if len(y) < 48:
        st.warning("No series here has the 48 continuous months needed for seasonal modelling.")
    else:
        z = np.log(y)
        train = y.iloc[:-hold]
        if mode.startswith("Automatic"):
            order, sorder, cv, tables = auto_select(country, source, len(train), dummy)
        else:
            order, sorder = (o[0], o[1], o[2]), (o[3], o[4], o[5], 12)
        name = f"SARIMA({order[0]},{order[1]},{order[2]})({sorder[0]},{sorder[1]},{sorder[2]})[12]"

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
        met = {"SARIMA": T.metrics(test, f_sarima), "Holt-Winters": T.metrics(test, f_hw), "Seasonal naive": T.metrics(test, f_sn)}
        lb = acorr_ljungbox(full.resid.iloc[13 + 12 * sorder[1]:], lags=[12], return_df=True)["lb_pvalue"].iloc[0]
        best = min(met, key=lambda k: met[k]["MAPE (%)"])

        tiles([("SARIMA model", name.replace("SARIMA", ""), "chosen automatically" if mode.startswith("Automatic") else "set manually"),
               ("SARIMA hold-out error (MAPE)", f"{met['SARIMA']['MAPE (%)']:.1f}%", f"best on hold-out: {best}"),
               ("AIC · BIC", f"{full.aic:.0f} · {full.bic:.0f}", "lower is better, same differencing only"),
               ("Ljung-Box p, lag 12", f"{lb:.3f}", "residuals look like white noise" if lb > 0.05 else "autocorrelation left in residuals")])

        heading(f"{source}: hold-out test and {horizon}-month forecast", "Terawatt-hours. Bands are the SARIMA 80% and 95% prediction intervals.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y, name="Observed", line=dict(color=INK, width=1.6), hovertemplate="%{y:,.1f}"))
        band(fig, ci95.index, ci95.iloc[:, 0], ci95.iloc[:, 1], "rgba(235,104,52,0.12)", "95% interval")
        band(fig, ci80.index, ci80.iloc[:, 0], ci80.iloc[:, 1], "rgba(235,104,52,0.26)", "80% interval")
        fig.add_trace(go.Scatter(x=mid.index, y=mid, name="Forecast", line=dict(color="#eb6834", width=2.4), hovertemplate="%{y:,.1f}"))
        fig.add_trace(go.Scatter(x=test.index, y=f_sarima, name="SARIMA on hold-out", line=dict(color="#e34948", width=2), hovertemplate="%{y:,.1f}"))
        fig.add_trace(go.Scatter(x=test.index, y=f_hw, name="Holt-Winters on hold-out", line=dict(color="#6250d6", width=1.6, dash="dot"), hovertemplate="%{y:,.1f}"))
        show(fig, 430)

        left, right = st.columns(2, gap="large")
        with left:
            heading("Hold-out accuracy", f"Last {hold} months, original scale.")
            st.dataframe(pd.DataFrame(met).T.rename_axis("Model").reset_index(), hide_index=True, width="stretch",
                         column_config={"RMSE": st.column_config.NumberColumn(format="%.2f"),
                                        "MAE": st.column_config.NumberColumn(format="%.2f"),
                                        "MAPE (%)": st.column_config.NumberColumn(format="%.2f")})
            if mode.startswith("Automatic"):
                heading("Cross-validated error of each finalist", "Best-AIC model of each accepted differencing scheme.")
                st.dataframe(pd.DataFrame({"Finalist": list(cv), "CV MAPE (%)": list(cv.values())}), hide_index=True,
                             width="stretch", column_config={"CV MAPE (%)": st.column_config.NumberColumn(format="%.2f")})
        with right:
            heading("Coefficients", "Fitted on all data, log scale.")
            st.dataframe(pd.DataFrame({"Term": full.params.index, "Estimate": full.params.values, "p-value": full.pvalues.values}),
                         hide_index=True, width="stretch",
                         column_config={"Estimate": st.column_config.NumberColumn(format="%.4f"),
                                        "p-value": st.column_config.NumberColumn(format="%.4f")})

        if mode.startswith("Automatic"):
            with st.expander("AIC / BIC tables: lags chosen within each differencing scheme"):
                cols = st.columns(len(tables))
                for col, (k, t) in zip(cols, tables.items()):
                    col.markdown(f"**{k}**")
                    col.dataframe(t, hide_index=True, width="stretch",
                                  column_config={"AIC": st.column_config.NumberColumn(format="%.1f"),
                                                 "BIC": st.column_config.NumberColumn(format="%.1f")})

        with st.expander("Stationarity tests and ACF / PACF"):
            rows = []
            for d_, D_ in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                w = z.diff(12) if D_ else z
                w = w.diff() if d_ else w
                rows.append({**T.stationarity_report(w, ""), "d": d_, "D": D_})
            st.dataframe(pd.DataFrame(rows)[["d", "D", "ADF p-value", "KPSS p-value", "Verdict"]], hide_index=True, width="stretch",
                         column_config={"ADF p-value": st.column_config.NumberColumn(format="%.4f"),
                                        "KPSS p-value": st.column_config.NumberColumn(format="%.4f")})
            w = z.diff(12) if sorder[1] else z
            w = (w.diff() if order[1] else w).dropna()
            nl = min(36, len(w) // 2 - 1)
            fig = make_subplots(1, 2, subplot_titles=(f"ACF of log {source}, d = {order[1]}, D = {sorder[1]}", "PACF"), horizontal_spacing=0.08)
            bound = 1.96 / np.sqrt(len(w))
            for i, vals in enumerate([acf(w, nlags=nl)[1:], pacf(w, nlags=nl, method="ywm")[1:]], start=1):
                fig.add_trace(go.Bar(x=list(range(1, nl + 1)), y=vals, marker_color="#2a78d6", width=0.5,
                                     hovertemplate="lag %{x}: %{y:.2f}<extra></extra>"), 1, i)
                for s_ in (bound, -bound):
                    fig.add_hline(y=s_, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=i)
            fig.update_annotations(font=dict(size=13, color=INK2))
            show(fig, 300, legend=False, hover="closest")

# ------------------------------------------------------------------ crossover
with tab_cross:
    st.markdown(r"We model $r_t=\log(\text{solar}_t/\text{coal}_t)$ with SARIMA and simulate thousands of future paths, "
                "redrawing the trend from its estimated uncertainty. Solar beats coal when $r_t>0$, and the share of "
                "paths that have crossed by a year is the probability.")
    coal_share = p["Coal"].iloc[-12:].sum() / p["Total generation"].iloc[-12:].sum()
    if coal_share < 0.01:
        st.info(f"{country} generates little or no electricity from coal, so there is no crossover to forecast.")
    else:
        n_paths = st.select_slider("Simulated paths", [500, 1000, 2000, 5000], value=2000)
        cx = crossover(country, n_paths)
        if cx is None:
            st.warning("There are not enough continuous months with both solar and coal to model the ratio.")
        else:
            s = cx["summary"]
            ahead = cx["last12"] > 1
            tiles([("Ratio model", cx["model"].replace("SARIMA", ""), "SARIMA on log(solar ÷ coal)"),
                   ("Most likely first full year solar beats coal", "already" if ahead else year_str(s["median"]),
                    f"solar = {cx['last12']:.2f}× coal, last 12 months"),
                   ("Chance by 2030", f"{s['P(by 2030)']:.0%}", ""),
                   ("Chance by 2035", f"{s['P(by 2035)']:.0%}", f"{s['P(before 2050)']:.0%} before 2051")])
            if cx["observed"]:
                st.caption(f"Solar first out-generated coal over a full calendar year in {cx['observed']}.")

            heading("Solar output as a multiple of coal", "Annual mean, log scale. Above the red line, solar out-generates coal.")
            fan, hist = cx["fan"], cx["hist"]
            fig = go.Figure()
            band(fig, fan.index, fan["p10"], fan["p90"], "rgba(235,104,52,0.2)", "80% of simulated paths")
            fig.add_trace(go.Scatter(x=fan.index, y=fan["p50"], name="Median simulated path",
                                     line=dict(color="#eb6834", width=2.4, dash="dash"), hovertemplate="%{y:.2f}×"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Observed", mode="lines+markers",
                                     line=dict(color=INK, width=2), marker=dict(size=6), hovertemplate="%{y:.2f}×"))
            fig.add_hline(y=1, line=dict(color="#e34948", width=1.5))
            lo = max(1e-3, min(hist.min(), 1) / 3)
            hi = max(lo * 1e3, 30)
            ticks = [v for v in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100) if lo <= v <= hi]
            fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=ticks,
                             ticktext=[f"{v:g}×" for v in ticks])
            show(fig, 400)

            heading("When does solar first beat coal?", "Probability that each year is the first full year with more solar than coal.")
            yrs = cx["years"]
            first = int(np.nanmin(yrs)) if np.isfinite(np.nanmin(yrs)) else 2026
            b = np.arange(first, 2051)
            fig = go.Figure(go.Bar(x=b, y=[np.mean(yrs == v) for v in b], marker_color="#eb6834", width=0.65,
                                   hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(dtick=2 if len(b) > 12 else 1)
            show(fig, 300, legend=False, hover="closest")

# ------------------------------------------------------------------ India target
with tab_target:
    t = RESULTS.get("india_target")
    if not t:
        st.info("Run build_results.py to compute the target analysis.")
    else:
        st.markdown(f"India's goal is **{t['target']} GW of non-fossil capacity by {t['target_year']}**. Its growth rate has "
                    "been rising, so two ARIMA models on log capacity encode two views of the future, each simulated 5,000 times.")
        sc = t["scenarios"]
        tiles([("Chance of 500 GW: growth returns to its long-run trend", f"{sc['Long-run trend']['p_hit']:.0%}",
                f"median {sc['Long-run trend']['median'][-1]:.0f} GW in 2030"),
               ("Chance of 500 GW: recent acceleration continues", f"{sc['Accelerating growth']['p_hit']:.0%}",
                f"median {sc['Accelerating growth']['median'][-1]:.0f} GW in 2030"),
               ("Growth needed, 2025 to 2030", f"{t['required_cagr']:.1f}% a year",
                f"achieved 2015 to 2025: {t['cagr'].get('2015-2025', float('nan')):.1f}% a year")])

        heading("Non-fossil installed capacity", "Gigawatts. Lines are median paths; bands hold 80% of simulations.")
        fig = go.Figure()
        x0, y0 = t["history"]["years"][-1], t["history"]["gw"][-1]
        for (k, s), c, rgba in zip(sc.items(), ["#2a78d6", "#eb6834"], ["rgba(42,120,214,0.16)", "rgba(235,104,52,0.16)"]):
            xs = [x0] + s["years"]
            band(fig, xs, [y0] + s["p10"], [y0] + s["p90"], rgba, f"{k}: 80% band", legend=False)
            fig.add_trace(go.Scatter(x=xs, y=[y0] + s["median"], name=k, line=dict(color=c, width=2.4, dash="dash"),
                                     hovertemplate="%{y:,.0f} GW"))
        fig.add_trace(go.Scatter(x=t["history"]["years"], y=t["history"]["gw"], name="Observed", mode="lines+markers",
                                 line=dict(color=INK, width=2), marker=dict(size=5), hovertemplate="%{y:,.0f} GW"))
        fig.add_hline(y=t["target"], line=dict(color="#e34948", width=1.5),
                      annotation_text=f"{t['target']} GW target", annotation_position="top left",
                      annotation_font=dict(color="#e34948", size=12.5))
        fig.update_yaxes(rangemode="tozero", ticksuffix=" GW")
        show(fig, 440)
        st.caption("Long-run trend: ARIMA with drift on log capacity, so growth reverts to its 25-year average. "
                   "Accelerating growth: ARIMA with d = 2, so the growth rate itself can keep rising. Source: Ember yearly data.")

# ------------------------------------------------------------------ world
with tab_world:
    rows = []
    for c in RESULTS["countries"]:
        if c.get("status") not in ("forecast", "crossed"):
            continue
        y = c["crossover"]["year"]
        rows.append({"Country": c["country"], "Coal share": c["share12"].get("Coal"), "Solar share": c["share12"].get("Solar"),
                     "Solar ÷ coal": c["last12_ratio"],
                     "First year solar > coal": "already" if c["status"] == "crossed" else year_str(y["median"]),
                     "Chance by 2030": 100 * y["P(by 2030)"], "Chance by 2035": 100 * y["P(by 2035)"]})
    df = pd.DataFrame(rows).sort_values("Solar ÷ coal", ascending=False)
    heading(f"{len(df)} countries with coal in their mix",
            "Ranked by solar output as a multiple of coal over the last 12 months. Shares are of total generation.")
    st.dataframe(df, hide_index=True, width="stretch", height=35 * (len(df) + 1) + 3,
                 column_config={"Coal share": st.column_config.NumberColumn(format="%.1f%%"),
                                "Solar share": st.column_config.NumberColumn(format="%.1f%%"),
                                "Solar ÷ coal": st.column_config.NumberColumn(format="%.2f×"),
                                "Chance by 2030": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                "Chance by 2035": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})

    cal = pd.DataFrame([{"Series": k, "cover": v["cover80"]} for c in RESULTS["countries"] for k, v in c.get("calibration", {}).items()])
    if len(cal):
        heading("Can the probabilities be trusted?",
                "Each model was refitted without its last 12 months; a well-calibrated model puts about 80% of those months inside its 80% interval.")
        names = {"Solar": "Solar generation", "Coal": "Coal generation", "LogRatio": "Solar ÷ coal ratio"}
        g = cal.groupby("Series").cover.agg(["count", "mean"])
        tiles([(names[k], f"{g.loc[k, 'mean']:.0%}", f"inside the 80% interval · {g.loc[k, 'count']} countries")
               for k in ["Solar", "Coal", "LogRatio"] if k in g.index])
