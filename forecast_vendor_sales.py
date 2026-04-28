import sqlite3
import pandas as pd
import numpy as np
import logging
import time
import os
import warnings

from sqlalchemy import create_engine
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving plots
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH          = "inventory.db"
TOP_N_VENDORS    = 10          # forecast only the top N vendors by revenue
FORECAST_DAYS    = 90          # how many days ahead to forecast
TARGET           = "SalesDollars"   # or "SalesQuantity"
SEASONALITY_MODE = "multiplicative" # "additive" for stable, "multiplicative" for growing trends

OUTPUT_DIR       = "forecasts"
PLOT_DIR         = os.path.join(OUTPUT_DIR, "plots")

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs("logs",     exist_ok=True)

logging.basicConfig(
    filename="logs/forecast_vendor_sales.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

# ─────────────────────────────────────────────
# Step 1: Load & Aggregate
# ─────────────────────────────────────────────
def load_daily_sales(db_path: str) -> pd.DataFrame:
    """
    Pull daily aggregated sales per vendor from the raw `sales` table.
    Returns columns: VendorNo, VendorName, ds (date), SalesDollars, SalesQuantity
    """
    logging.info("Loading daily sales from DB...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            VendorNo,
            VendorName,
            DATE(SalesDate)     AS ds,
            SUM(SalesDollars)   AS SalesDollars,
            SUM(SalesQuantity)  AS SalesQuantity
        FROM sales
        GROUP BY VendorNo, VendorName, DATE(SalesDate)
        ORDER BY VendorNo, ds
    """, conn)
    conn.close()

    df["ds"] = pd.to_datetime(df["ds"])
    logging.info(f"Loaded {len(df):,} vendor-day rows, "
                 f"{df['VendorNo'].nunique()} unique vendors")
    return df


def get_top_vendors(df: pd.DataFrame, n: int) -> list:
    """Return VendorNo list for top N vendors by total SalesDollars."""
    top = (
        df.groupby("VendorNo")["SalesDollars"]
        .sum()
        .nlargest(n)
        .index.tolist()
    )
    logging.info(f"Top {n} vendors selected: {top}")
    return top


# ─────────────────────────────────────────────
# Step 2: Prophet Forecasting
# ─────────────────────────────────────────────
def build_prophet_model(seasonality_mode: str) -> Prophet:
    """Instantiate a Prophet model with sensible defaults for retail sales."""
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,            # 95% confidence interval
        changepoint_prior_scale=0.05,   # controls trend flexibility
    )
    # Add US holidays (tweak if your data is from a different region)
    model.add_country_holidays(country_name="US")
    return model


def forecast_vendor(vendor_df: pd.DataFrame,
                    vendor_no,
                    vendor_name: str,
                    target: str,
                    forecast_days: int,
                    seasonality_mode: str) -> pd.DataFrame:
    """
    Fit Prophet on one vendor's daily time series and return forecast df.
    Prophet requires columns named `ds` (date) and `y` (target).
    """
    ts = vendor_df[["ds", target]].rename(columns={target: "y"}).copy()
    ts = ts.sort_values("ds").drop_duplicates("ds")

    # Need at least 2 full periods to fit seasonalities
    if len(ts) < 14:
        logging.warning(f"Skipping vendor {vendor_no} — only {len(ts)} data points")
        return pd.DataFrame()

    model = build_prophet_model(seasonality_mode)

    # Suppress Prophet's verbose stdout
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(ts)

    future = model.make_future_dataframe(periods=forecast_days, freq="D")
    forecast = model.predict(future)

    # Clip negative predictions (sales can't be negative)
    forecast["yhat"]       = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    # Tag with vendor info
    forecast["VendorNo"]   = vendor_no
    forecast["VendorName"] = vendor_name
    forecast["Target"]     = target

    # Save plot
    fig = model.plot(forecast, figsize=(14, 5))
    plt.title(f"{vendor_name} — {target} Forecast ({forecast_days}d ahead)")
    plt.tight_layout()
    safe_name = str(vendor_no).replace("/", "_")
    fig.savefig(os.path.join(PLOT_DIR, f"{safe_name}_{target}.png"), dpi=100)
    plt.close(fig)

    # Save components plot
    fig2 = model.plot_components(forecast, figsize=(14, 8))
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOT_DIR, f"{safe_name}_{target}_components.png"), dpi=100)
    plt.close(fig2)

    return forecast[["ds", "VendorNo", "VendorName", "Target",
                      "yhat", "yhat_lower", "yhat_upper"]]


# ─────────────────────────────────────────────
# Step 3: Run All Vendors
# ─────────────────────────────────────────────
def run_forecasts(df: pd.DataFrame,
                  top_vendors: list,
                  target: str,
                  forecast_days: int,
                  seasonality_mode: str) -> pd.DataFrame:
    all_forecasts = []

    for i, vendor_no in enumerate(top_vendors, 1):
        vendor_df   = df[df["VendorNo"] == vendor_no]
        vendor_name = vendor_df["VendorName"].iloc[0]

        print(f"  [{i}/{len(top_vendors)}] Forecasting: {vendor_name} (#{vendor_no})")
        logging.info(f"Forecasting vendor {vendor_no} — {vendor_name}")

        start = time.time()
        forecast_df = forecast_vendor(
            vendor_df, vendor_no, vendor_name,
            target, forecast_days, seasonality_mode
        )
        elapsed = round(time.time() - start, 2)

        if forecast_df.empty:
            continue

        # Save per-vendor CSV
        safe_name = str(vendor_no).replace("/", "_")
        csv_path  = os.path.join(OUTPUT_DIR, f"{safe_name}_{target}_forecast.csv")
        forecast_df.to_csv(csv_path, index=False)

        # Print last actual date + first forecast date
        history_end  = df[df["VendorNo"] == vendor_no]["ds"].max().date()
        forecast_end = forecast_df["ds"].max().date()
        print(f"       History ends: {history_end} | "
              f"Forecast to: {forecast_end} | "
              f"Time: {elapsed}s")
        logging.info(f"  Done in {elapsed}s | forecast to {forecast_end}")

        all_forecasts.append(forecast_df)

    return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()


# ─────────────────────────────────────────────
# Step 4: Persist to DB
# ─────────────────────────────────────────────
def save_forecasts_to_db(forecast_df: pd.DataFrame, db_path: str):
    """Write combined forecast results to SQLite."""
    engine = create_engine(f"sqlite:///{db_path}")
    forecast_df["ds"] = forecast_df["ds"].astype(str)
    forecast_df.to_sql("vendor_sales_forecast", con=engine,
                       if_exists="replace", index=False)
    logging.info(f"Saved {len(forecast_df):,} rows to 'vendor_sales_forecast' table")
    print(f"\n  ✔ {len(forecast_df):,} rows saved to 'vendor_sales_forecast' in {db_path}")


# ─────────────────────────────────────────────
# Step 5: Summary Table
# ─────────────────────────────────────────────
def print_forecast_summary(forecast_df: pd.DataFrame, df_actual: pd.DataFrame):
    """Print a clean summary of forecasted revenue per vendor."""
    last_actual_date = df_actual["ds"].max()
    future_only = forecast_df[pd.to_datetime(forecast_df["ds"]) > last_actual_date]

    summary = (
        future_only.groupby(["VendorNo", "VendorName"])["yhat"]
        .sum()
        .reset_index()
        .rename(columns={"yhat": f"Forecasted_{TARGET}_{FORECAST_DAYS}d"})
        .sort_values(f"Forecasted_{TARGET}_{FORECAST_DAYS}d", ascending=False)
    )

    print(f"\n{'='*65}")
    print(f"  Forecast Summary — Next {FORECAST_DAYS} Days ({TARGET})")
    print(f"{'='*65}")
    print(f"  {'VendorName':<35} {'Forecasted ($)':>15}")
    print(f"  {'-'*35} {'-'*15}")
    for _, row in summary.iterrows():
        val = row[f"Forecasted_{TARGET}_{FORECAST_DAYS}d"]
        print(f"  {row['VendorName']:<35} {val:>15,.2f}")

    logging.info(f"\nForecast Summary:\n{summary.to_string()}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    total_start = time.time()
    logging.info("========== forecast_vendor_sales.py START ==========")
    print("========== Vendor Sales Time-Series Forecasting ==========")
    print(f"  Target        : {TARGET}")
    print(f"  Top N Vendors : {TOP_N_VENDORS}")
    print(f"  Forecast Days : {FORECAST_DAYS}")
    print(f"  Seasonality   : {SEASONALITY_MODE}\n")

    # 1. Load
    df = load_daily_sales(DB_PATH)

    # 2. Select top vendors
    top_vendors = get_top_vendors(df, TOP_N_VENDORS)

    # 3. Forecast
    print(f"{'='*65}")
    print("  Running Prophet forecasts...")
    print(f"{'='*65}")
    forecast_df = run_forecasts(df, top_vendors, TARGET, FORECAST_DAYS, SEASONALITY_MODE)

    if forecast_df.empty:
        print("No forecasts generated. Check logs.")
    else:
        # 4. Save to DB
        save_forecasts_to_db(forecast_df, DB_PATH)

        # 5. Summary
        print_forecast_summary(forecast_df, df)

        print(f"\n  ✔ Per-vendor CSVs  → {OUTPUT_DIR}/")
        print(f"  ✔ Forecast plots   → {PLOT_DIR}/")

    total_time = round((time.time() - total_start) / 60, 2)
    logging.info(f"========== TOTAL TIME: {total_time} minutes ==========")
    print(f"\n========== Done in {total_time} min ==========")