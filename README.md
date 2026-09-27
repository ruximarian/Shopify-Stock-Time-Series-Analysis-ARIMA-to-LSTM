# Shopify (SHOP) Stock Price: Advanced Time Series Analysis — from ARIMA to LSTM

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-econometrics-4B8BBE)
![arch](https://img.shields.io/badge/arch-GARCH-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%20LSTM-FF6F00?logo=tensorflow&logoColor=white)

An end-to-end time series study of Shopify's daily stock price (May 2015 – May 2026), combining classical econometrics, volatility modeling and deep learning to answer one question:

> **How much of Shopify's price movement is explained by the broad market (S&P 500), interest rates (10Y Treasury yield) and market fear (VIX) — and how much is company-specific?**

Built as a team project for the *Advanced Time Series Analysis* course, MSc in Applied Statistics and Data Science, Bucharest University of Economic Studies (ASE).

---

## Key findings

- **Shopify is driven mainly by company-specific factors.** In the VAR(2) model, 88% of SHOP's 24-month forecast error variance comes from its own shocks; the S&P 500 explains about 10%.
- **Interest rates matter more for SHOP than for the market.** 10Y yields Granger-cause SHOP (p = 0.021). In the structural VAR, a rate shock hits SHOP immediately (−0.27 in month 1) and fades over 8–10 months, while the effect on the S&P 500 is roughly ten times smaller — consistent with how rising rates discount the future earnings of growth stocks.
- **A structural break in January 2022**, identified endogenously by the Zivot–Andrews test (p = 0.042), coincides with the Fed's pivot toward aggressive rate hikes — about seven weeks *before* the invasion of Ukraine.
- **Correlations are not constant.** DCC-GARCH shows the SHOP–S&P 500 correlation moving between ~0.25 in calm periods and ~0.65 during stress, so a single static correlation understates risk exactly when it matters.
- **No long-run equilibrium** between SHOP, the S&P 500 and yields: Johansen (r = 0) and the ARDL Bounds Test (F = 2.06 < 3.10) agree, so the multivariate models are estimated on returns rather than as a VECM.
- **The price is essentially a random walk**: I(1), no seasonality (STL seasonal strength = 0.00), flat periodogram, fat-tailed returns with strong ARCH effects.

---

## Data

| Series | Description | Source |
|---|---|---|
| `Close_SHOP` | Shopify daily close (split-adjusted) | Yahoo Finance |
| `Close_SP500` | S&P 500 index daily close | Yahoo Finance |
| `VIXCLS` | CBOE Volatility Index | FRED |
| `DGS10` | 10-Year Treasury constant maturity yield | FRED |
| `Volume_SHOP` | Shopify daily trading volume | Yahoo Finance (`yfinance`, downloaded in the notebook) |

- **Period:** May 2015 – May 2026, ~2,750 daily observations
- **Frequency:** daily for univariate, DCC-GARCH and LSTM models; aggregated to monthly for VAR, ARDL and SVAR
- **Split:** chronological 80% train / 20% test (test period ≈ March 2024 – May 2026)

---

## Methodology

### Part I — Univariate analysis (`notebooks/01_univariate_analysis.ipynb`)

| Step | Methods |
|---|---|
| Decomposition | STL (period = 252 trading days), season and subseries plots |
| Stationarity | Log transform, ADF, KPSS, Zivot–Andrews structural break test |
| Identification | ACF / PACF on log prices and log returns |
| Exponential smoothing | Holt linear trend (daily), Holt–Winters (monthly) |
| ARIMA | Candidate orders from ACF/PACF vs. `auto_arima`; selected ARIMA(3,1,0) by AIC; Ljung–Box and Jarque–Bera diagnostics |
| Volatility | ARCH-LM test → GARCH(1,1) Normal → Student-t → EGARCH and GJR-GARCH; Sign Bias test |
| State space | Local Linear Trend model estimated with the Kalman filter / smoother |
| Spectral analysis | Periodogram of daily returns |

### Part II — Multivariate analysis (`notebooks/02_multivariate_analysis.ipynb`)

| Step | Methods |
|---|---|
| Exploration | Correlation matrices (levels vs. returns), 60-day rolling mean and volatility, VIF |
| Feature selection | VIX excluded from linear models (r = −0.74 with S&P 500 returns; VIX is built from S&P 500 options) |
| Cointegration | Engle–Granger and Johansen trace test |
| VAR(2) | Lag selection, Granger causality, impulse response functions, FEVD, stability and residual diagnostics |
| DCC-GARCH | Two-step estimation: univariate GARCH(1,1) per series, then dynamic conditional correlations (custom likelihood optimized with `scipy`) |
| ARDL | Order selection by AIC, error-correction form, Pesaran–Shin–Smith Bounds Test |
| SVAR | Recursive (Cholesky) identification: DGS10 → S&P 500 → SHOP; structural IRFs with Monte Carlo bands, structural FEVD |
| LSTM | 5 input features, 60-day window, 2 × LSTM(64) + Dropout(0.2) + Dense(32) + Dense(1), Adam, MSE loss, early stopping |

---

## Results

### Univariate forecasts (multi-step, full test horizon)

Each model is fit once on the training set and forecasts the entire ~2-year test period without seeing any test data.

| Model | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|
| **State Space (Local Linear Trend)** | 19.77 | **26.08** | **16.94** |
| Holt Linear Trend | **19.54** | 26.56 | 19.88 |
| ARIMA(3,1,0) | 37.69 | 47.21 | 30.29 |
| EGARCH(1,1)-t\* | 55.56 | 73.38 | 50.28 |

\*EGARCH models volatility, not the price level; it is included for completeness and is the relevant model for risk measures such as VaR.

![Univariate forecasts on the test set](images/univariate_forecasts.png)

Adaptive models (state space, Holt) beat ARIMA, which reverts to a flat forecast over long horizons.

### Volatility models (in-sample fit)

| Model | AIC | BIC |
|---|---|---|
| GARCH(1,1)-Normal | 11853.9 | 11876.7 |
| GARCH(1,1)-t | 11630.2 | 11658.7 |
| **EGARCH(1,1)-t** | **11589.7** | **11629.6** |
| GJR-GARCH(1,1)-t | 11604.7 | 11638.9 |

### Multivariate models

| Model | Setup | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|---|
| VAR(2) | Monthly, multi-step out-of-sample | 17.46 | 20.91 | 17.06 |
| LSTM (5 features) | Daily, one-step-ahead out-of-sample | 10.62 | 13.69 | 9.01 |

ARDL(1,1,1) reaches an in-sample MAPE of 7.42%, but this is a training-set fit driven by the lagged dependent variable (coefficient ≈ 1.006) and is not comparable to out-of-sample forecasts.

![LSTM one-step-ahead forecast](images/lstm_forecast.png)

### Dynamic correlations (DCC-GARCH)

DCC parameters: α = 0.019, β = 0.953 (α + β = 0.972) — correlations react slowly to new shocks but are highly persistent once they shift.

![DCC-GARCH conditional correlations](images/dcc_correlations.png)

### Shock transmission (VAR impulse responses)

![VAR impulse response functions](images/var_impulse_response.png)

---

## Limitations and honest comparison

- **Forecast setups differ, so errors are not directly comparable across all models.** The univariate models and VAR forecast many steps ahead from the end of the training set, while the LSTM predicts one day ahead using the actual previous 60 days. One-step-ahead prediction is a much easier task.
- **A naive persistence baseline** (tomorrow's price = today's price) reaches an MAE of about $2.67 on the same daily test period, which is lower than the LSTM's $10.62. The LSTM's forecast visibly lags and smooths the actual series. This is a well-known pattern in stock price prediction and supports the econometric conclusion that SHOP behaves close to a random walk: the useful outputs of this project are the structural insights (drivers, shock transmission, volatility and correlation regimes), not point price forecasts.
- Results depend on a single train/test split; walk-forward (rolling-origin) validation would give more robust error estimates.
- The notebooks' commentary and chart labels are in Romanian; this README summarizes them in English.

## Possible extensions

- Walk-forward validation and comparison of every model against naive and random-walk-with-drift baselines
- Predicting returns or direction instead of price levels
- Technical indicators (e.g., RSI) and earnings-date features for the LSTM
- Value-at-Risk backtesting using the EGARCH-t and DCC-GARCH volatility forecasts

---

## Repository structure

```
.
├── data/
│   ├── shop_sp500_daily.xlsx        # SHOP + S&P 500 daily closes (univariate notebook)
│   └── shop_macro_daily.xlsx        # SHOP, S&P 500, VIX, DGS10 daily (multivariate notebook)
├── notebooks/
│   ├── 01_univariate_analysis.ipynb
│   └── 02_multivariate_analysis.ipynb
├── images/                          # Figures used in this README
├── presentation/
│   └── shopify_time_series_presentation_RO.pdf
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/ruximarian/Shopify-Stock-Time-Series-Analysis-ARIMA-to-LSTM.git
cd Shopify-Stock-Time-Series-Analysis-ARIMA-to-LSTM
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run the notebooks from the `notebooks/` folder so the relative data paths resolve. The multivariate notebook downloads Shopify trading volume with `yfinance`, so it needs an internet connection. The LSTM uses a fixed TensorFlow seed (42), but results may still vary slightly across hardware and library versions.

## Tech stack

Python · pandas · NumPy · SciPy · statsmodels · pmdarima · arch · scikit-learn · TensorFlow / Keras · matplotlib · seaborn · yfinance

## Authors

- **Ruxandra-Elena Mărian** — [GitHub](https://github.com/ruximarian)
- **Diana-Maria Lazăr**

MSc Applied Statistics and Data Science, Bucharest University of Economic Studies (ASE)

## References

- Engle, R.F. (2002). Dynamic Conditional Correlation. *Journal of Business & Economic Statistics*, 20(3).
- Engle, R.F. & Granger, C.W.J. (1987). Co-integration and Error Correction. *Econometrica*, 55(2).
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors. *Econometrica*, 59(6).
- Nelson, D.B. (1991). Conditional Heteroskedasticity in Asset Returns. *Econometrica*, 59(2).
- Pesaran, M.H., Shin, Y. & Smith, R.J. (2001). Bounds Testing Approaches to the Analysis of Level Relationships. *Journal of Applied Econometrics*, 16(3).
- Sims, C.A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1).
- Zivot, E. & Andrews, D.W.K. (1992). Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis. *Journal of Business & Economic Statistics*, 10(3).

*Full reference list in the presentation.*
