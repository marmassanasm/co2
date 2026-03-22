import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="CO2 Simulator", layout="wide")

st.title("🌍 Simulador d'Emissions de CO₂ - País UABERS")
st.write("Modifica les variables per veure com canvien les emissions")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    BASE = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(BASE, "master_dataset.csv")
    
    # Intentar diferents separadors
    separators = [',', ';', '\t']
    
    for sep in separators:
        try:
            df = pd.read_csv(filepath, sep=sep, engine='python')
            if len(df.columns) > 1:
                return df
        except:
            continue
    
    st.error("❌ Error carregant dataset!")
    st.stop()

df = load_data()

# -----------------------------
# DEBUG INFO
# -----------------------------
with st.expander("🔍 Informació del dataset"):
    st.write(f"**Columnes:** {len(df.columns)}")
    st.write(f"**Files:** {len(df):,}")
    st.write(f"**Anys:** {df['Year'].min():.0f} - {df['Year'].max():.0f}")
    st.dataframe(df.head(3))

# -----------------------------
# VARIABLES
# -----------------------------
features = ["gdp_pc", "hdi", "gini", "pct_lowcarbon", "population"]
features = [f for f in features if f in df.columns]

target = "co2_prod_pc"

# Verificar columnes
if not all(col in df.columns for col in [target] + features):
    st.error("❌ Falten columnes necessàries!")
    st.stop()

# -----------------------------
# FILTRE PER ANY
# -----------------------------
year_selected = st.sidebar.slider(
    "Any de referència", 
    int(df['Year'].min()), 
    int(df['Year'].max()), 
    2019
)

df_year = df[df["Year"] == year_selected].copy()
df_clean = df_year.dropna(subset=features + [target]).copy()  # ← .copy() evita warnings

if len(df_clean) == 0:
    st.error(f"❌ No hi ha dades per {year_selected}!")
    st.stop()

st.sidebar.success(f"✅ {len(df_clean)} països amb dades")

# -----------------------------
# PREPARAR DADES
# -----------------------------
X = df_clean[features].copy()  # ← .copy() evita warnings
y = df_clean[target].copy()

# -----------------------------
# MODEL
# -----------------------------
model = LinearRegression()
model.fit(X, y)
r2 = model.score(X, y)

# -----------------------------
# INPUTS (SLIDERS)
# -----------------------------
st.sidebar.header("🎛️ Configuració UABERS")

input_values = {}

labels = {
    "gdp_pc": "PIB per càpita (USD)",
    "hdi": "HDI",
    "gini": "Gini",
    "pct_lowcarbon": "% Renovables",
    "population": "Població",
}

for feature in features:
    min_val = float(df_clean[feature].min())
    max_val = float(df_clean[feature].max())
    median_val = float(df_clean[feature].median())
    
    label = labels.get(feature, feature)
    
    if feature == "population":
        value = st.sidebar.number_input(
            f"{label} (M)",
            min_value=min_val / 1e6,
            max_value=max_val / 1e6,
            value=median_val / 1e6,
            step=0.1,
            format="%.1f"
        ) * 1e6
    elif feature == "hdi":
        value = st.sidebar.slider(label, min_val, max_val, median_val, format="%.3f")
    else:
        value = st.sidebar.slider(label, min_val, max_val, median_val, format="%.0f")
    
    input_values[feature] = value

# -----------------------------
# PREDICCIÓ (sense warnings)
# -----------------------------
input_df = pd.DataFrame([input_values], columns=features)  # ← Usa DataFrame en lloc de array
prediction = model.predict(input_df)[0]

# -----------------------------
# RESULTATS
# -----------------------------
st.subheader("📊 Resultat de la Simulació")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("CO₂ per càpita UABERS", f"{prediction:.2f} t/any")

with col2:
    avg = y.mean()
    st.metric("Mitjana mundial", f"{avg:.2f} t/any", delta=f"{prediction - avg:+.2f}")

with col3:
    percentile = (y < prediction).mean() * 100
    st.metric("Percentil", f"{percentile:.0f}%")

# Emissions totals
if "population" in input_values:
    total_gt = prediction * input_values["population"] / 1e9
    st.info(f"**Emissions totals UABERS:** {total_gt:.2f} Gt CO₂/any")

# -----------------------------
# GRÀFIC 1: PIB vs CO₂
# -----------------------------
if "gdp_pc" in features:
    st.subheader("📈 Posició de UABERS en el context global")
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Scatter països
    scatter = ax1.scatter(
        df_clean["gdp_pc"], 
        y,
        alpha=0.6,
        s=df_clean["population"] / 1e6 + 10 if "population" in df_clean.columns else 50,
        c=df_clean["hdi"] if "hdi" in df_clean.columns else "blue",
        cmap="RdYlGn",
        edgecolors="gray",
        linewidth=0.3
    )
    
    # UABERS
    ax1.scatter(
        input_values["gdp_pc"],
        prediction,
        s=500,
        marker="*",
        color="red",
        edgecolors="black",
        linewidths=2,
        zorder=10,
        label="UABERS"
    )
    
    ax1.set_xlabel("PIB per càpita (USD)", fontsize=12)
    ax1.set_ylabel("CO₂ per càpita (t/any)", fontsize=12)
    ax1.set_title(f"PIB vs Emissions CO₂ ({year_selected})", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    if "hdi" in df_clean.columns:
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("HDI", rotation=270, labelpad=20)
    
    st.pyplot(fig1)

# -----------------------------
# GRÀFIC 2: HDI vs CO₂
# -----------------------------
if "hdi" in features:
    st.subheader("🌍 HDI vs Emissions")
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    ax2.scatter(df_clean["hdi"], y, alpha=0.6, s=60, color="#1f77b4", edgecolors="gray", linewidth=0.3)
    ax2.scatter(input_values["hdi"], prediction, s=500, marker="*", color="red", 
                edgecolors="black", linewidths=2, zorder=10, label="UABERS")
    
    ax2.axhline(y=avg, color="green", linestyle="--", alpha=0.5, label="Mitjana")
    ax2.axvline(x=0.8, color="orange", linestyle="--", alpha=0.5, label="HDI alt")
    
    ax2.set_xlabel("HDI", fontsize=12)
    ax2.set_ylabel("CO₂ per càpita (t/any)", fontsize=12)
    ax2.set_title(f"Desenvolupament vs Emissions ({year_selected})", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    st.pyplot(fig2)

# -----------------------------
# IMPORTÀNCIA VARIABLES
# -----------------------------
st.subheader("📊 Importància de Variables")

coef_df = pd.DataFrame({
    "Variable": features,
    "Coeficient": model.coef_
}).sort_values("Coeficient", key=abs, ascending=False)

fig3, ax3 = plt.subplots(figsize=(10, 5))

colors = ["#d62728" if c > 0 else "#2ca02c" for c in coef_df["Coeficient"]]
ax3.barh(coef_df["Variable"], coef_df["Coeficient"], color=colors, edgecolor="black")
ax3.axvline(0, color="black", linewidth=1.5)
ax3.set_xlabel("Coeficient", fontsize=11)
ax3.set_title("Impacte de cada variable", fontsize=13, fontweight="bold")
ax3.grid(axis="x", alpha=0.3)

st.pyplot(fig3)

st.dataframe(coef_df.style.format({"Coeficient": "{:.6f}"}), use_container_width=True)

# -----------------------------
# RENDIMENT MODEL
# -----------------------------
st.subheader("🎯 Rendiment del Model")

from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.2f} t")
col3.metric("MAE", f"{mae:.2f} t")

# -----------------------------
# PAÏSOS SIMILARS
# -----------------------------
st.subheader("🗺️ Països Similars a UABERS")

df_similarity = df_clean.copy()  # ← .copy() evita warnings

# Calcular similaritat
df_similarity["similarity"] = np.sqrt(
    ((df_similarity["gdp_pc"] - input_values["gdp_pc"]) / df_similarity["gdp_pc"].std())**2 +
    ((df_similarity["hdi"] - input_values.get("hdi", df_similarity["hdi"].mean())) / df_similarity["hdi"].std())**2 +
    ((df_similarity["population"] - input_values.get("population", df_similarity["population"].mean())) / df_similarity["population"].std())**2
)

similar = df_similarity.nsmallest(5, "similarity")[["Entity", target, "gdp_pc", "hdi", "population"]]

st.dataframe(
    similar.style.format({
        target: "{:.2f} t",
        "gdp_pc": "${:,.0f}",
        "hdi": "{:.3f}",
        "population": "{:,.0f}"
    }),
    use_container_width=True
)

# -----------------------------
# DESCÀRREGA
# -----------------------------
st.subheader("💾 Exportar Resultats")

result_data = {
    "Any": year_selected,
    **{f"{k}": v for k, v in input_values.items()},
    "CO2_per_capita": prediction,
    "Mitjana_mundial": avg,
    "Percentil": percentile,
    "R2": r2
}

result_df = pd.DataFrame([result_data])
csv = result_df.to_csv(index=False)

st.download_button(
    "📥 Descarregar configuració",
    csv,
    f"uabers_{year_selected}.csv",
    "text/csv"
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(f"📅 {year_selected} | 🌐 {len(df_clean)} països | 🔬 R² = {r2:.3f}")