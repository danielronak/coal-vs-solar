# Solar vs Coal: forecasting the energy transition

An applied time series project. It models monthly electricity generation by source with SARIMA, estimates **when solar will overtake coal** as a probability distribution, tests whether **India will reach 500 GW of non-fossil capacity by 2030**, and runs the same pipeline automatically on every country in the dataset.

Data: [Ember](https://ember-energy.org) monthly and yearly electricity data (CC-BY-4.0).

## What's here

| File | Purpose |
|---|---|
| `Solar_vs_Coal_Time_Series_Analysis.ipynb` | The report. Follows the rubric order: data, EDA, missing values, ADF/KPSS, transformations, STL, ACF/PACF, model identification, AIC/BIC, evaluation, forecasts, then the extensions |
| `transition.py` | Shared library: stationarity tests, two-stage SARIMA selection, rolling-origin CV, crossover simulation, capacity-target analysis |
| `build_results.py` | Runs the pipeline for every country and writes `results/results.json` (about 1 hour on 4 cores) |
| `app.py` | Streamlit app: refits SARIMA live for any country and series, shows the AIC/BIC tables, diagnostics, forecasts, crossover odds and the India 2030 tracker |
| `export_dashboard_data.py` | Flattens the results into the CSV tables used by the published web dashboard (`dashboard_data/`) |
| `dashboard/index.html` | Source of the published web dashboard: a d3 page that reads the `dashboard_data/` tables |
| `results/results.json` | Precomputed results for every country, read by the app and the dashboard |
| `data/` | The two Ember CSV files |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild every result from the raw data:

```bash
python build_results.py
python export_dashboard_data.py
jupyter nbconvert --to notebook --execute --inplace Solar_vs_Coal_Time_Series_Analysis.ipynb
```

## Deploy the app for free (Streamlit Community Cloud)

1. Push this folder to a **public** GitHub repository.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **Create app**, pick the repository and branch, and set the main file to `app.py`.
4. Click **Deploy**. The app gets a public `*.streamlit.app` address. It sleeps after a few days without visitors and wakes in about 30 seconds on the next visit.

## Method in brief

- **Transform:** log of generation (Box-Cox λ ≈ 0 for solar), so seasonal swings that grow with the level become additive and errors become percentages.
- **Differencing:** ADF and KPSS are run on every scheme (d=1, D=1, both). Every scheme that both tests accept is kept.
- **Lags:** within each accepted scheme, SARIMA(p,d,q)(P,D,Q)₁₂ orders are ranked by AIC (BIC reported). AIC is never compared across schemes, since differencing changes the sample the likelihood is computed on.
- **Scheme choice:** rolling-origin cross-validation (MAPE on the original scale) picks between the AIC-best models of each scheme.
- **Benchmarks:** seasonal naive and damped Holt-Winters.
- **Crossover:** SARIMA on log(solar ÷ coal) produces 2,000 simulated paths. The drift is redrawn from its sampling distribution to include trend uncertainty. The first year in which each path's annual mean exceeds 0 is recorded.
- **Calibration:** for every country, the last 12 months are held out and the share inside the 80% interval is counted.

## Limitations

- The models extrapolate statistical trends. Policy, prices, storage costs or new coal plants can change the path.
- Monthly data for most countries starts between 2015 and 2019, so long-horizon intervals are wide. That is intended.
- India's coal generation fell in 2025 and rebounded in 2026. Damped Holt-Winters handles this break better than SARIMA, and the notebook reports that.
