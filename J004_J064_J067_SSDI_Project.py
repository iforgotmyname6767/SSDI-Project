import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import requests
import streamlit as st
from io import StringIO


# ---------------- FETCH DATA ---------------- #

def fetch_exoplanet_data():
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = "SELECT pl_name, pl_bmassj, pl_radj, default_flag FROM ps"
    params = {"query": query, "format": "csv"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    return df


# ---------------- CLEAN DATA ---------------- #

def preprocess(df):
    df = df[df["default_flag"] == 1].copy()
    df = df.dropna(subset=["pl_bmassj", "pl_radj"])
    df = df[(df["pl_bmassj"] > 0) & (df["pl_radj"] > 0)]
    df = df[(df["pl_bmassj"] < 80) & (df["pl_radj"] < 3)]

    df["log_mass"] = np.log10(df["pl_bmassj"])
    df["log_radius"] = np.log10(df["pl_radj"])

    return df


# ---------------- REGRESSION ---------------- #

def run_regression(df):
    X = df[["log_mass"]].values
    y = df["log_radius"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    slope = model.coef_[0]
    intercept = model.intercept_

    return model, X_train, X_test, y_train, y_test, y_pred, slope, intercept, r2, rmse


# ---------------- PREDICTION ---------------- #

def predict_radius(model, mass_jupiter):
    log_mass = np.log10(mass_jupiter)
    log_radius = model.predict([[log_mass]])[0]
    return 10 ** log_radius


# ---------------- PLOTTING ---------------- #

def plot_results(df, model, X_test, y_test, y_pred, slope, intercept, r2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Log-log ---
    ax = axes[0]
    ax.scatter(df["log_mass"], df["log_radius"], alpha=0.3, s=20, color="steelblue")
    ax.scatter(X_test, y_test, alpha=0.6, s=25, color="orange")

    x_line = np.linspace(df["log_mass"].min(), df["log_mass"].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))

    ax.plot(x_line, y_line, color="red", linewidth=2)

    ax.set_xlabel("log₁₀(Mass)")
    ax.set_ylabel("log₁₀(Radius)")
    ax.set_title(f"Log-Log Fit | R² = {r2:.3f}")
    ax.grid(True)

    # --- Plot 2: Original scale ---
    ax2 = axes[1]
    mass_range = np.logspace(df["log_mass"].min(), df["log_mass"].max(), 300)
    radius_pred = 10 ** model.predict(np.log10(mass_range).reshape(-1, 1))

    ax2.scatter(df["pl_bmassj"], df["pl_radj"], alpha=0.3, s=20)
    ax2.plot(mass_range, radius_pred, color="red")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Mass")
    ax2.set_ylabel("Radius")
    ax2.set_title("Original Scale")
    ax2.grid(True)

    plt.tight_layout()
    return fig


# ---------------- STREAMLIT APP ---------------- #

st.title("Exoplanet Mass-Radius Regression")
st.write("Predict planet radius from mass using NASA data")

with st.spinner("Fetching data..."):
    raw_df = fetch_exoplanet_data()

df = preprocess(raw_df)

st.write(f"Dataset size: {len(df)} planets")


model, X_train, X_test, y_train, y_test, y_pred, slope, intercept, r2, rmse = run_regression(df)


col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{r2:.4f}")
col2.metric("RMSE (log)", f"{rmse:.4f}")
col3.metric("Exponent", f"{slope:.4f}")

st.write(f"Equation: log₁₀(R) = {slope:.4f} × log₁₀(M) + {intercept:.4f}")


fig = plot_results(df, model, X_test, y_test, y_pred, slope, intercept, r2)
st.pyplot(fig)

# Prediction
st.subheader("Predict Radius")
mass_input = st.number_input(
    "Mass (Jupiter masses)", min_value=0.001, max_value=79.0, value=1.0
)

predicted = predict_radius(model, mass_input)
st.write(f"Predicted radius: {predicted:.4f} Jupiter radii")