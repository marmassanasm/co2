import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Predictor Temporal CO₂", layout="wide")

st.title("🔮 Predictor d'Emissions i Desenvolupament Global")
st.write("Previsions a 10, 25, 50, 75 i 100 anys per als països més significants")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    BASE = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE, "master_dataset.csv"))
    return df

df = load_data()

# -----------------------------
# SELECCIÓ DE PAÏSOS SIGNIFICANTS
# -----------------------------
st.sidebar.header("⚙️ Configuració")

# Automàtic: Top emissors actuals + Top HDI + Representants regionals
df_recent = df[df['Year'] >= 2015].copy()

top_emissors = df_recent.groupby('Entity')['co2_prod'].sum().nlargest(15).index.tolist()
top_hdi = df_recent.dropna(subset=['hdi']).groupby('Entity')['hdi'].mean().nlargest(10).index.tolist()

# Països representatius per regió
representatius = [
    "China", "United States", "India", "Russia", "Japan",  # Top emissors
    "Germany", "United Kingdom", "France", "Brazil", "Indonesia",  # Grans economies
    "Norway", "Switzerland", "Sweden", "Denmark",  # HDI alt + nets
    "Nigeria", "Ethiopia", "Bangladesh", "Pakistan",  # Poblacions grans en desenvolupament
    "Qatar", "Saudi Arabia", "United Arab Emirates",  # Petro-estats
    "Spain", "Italy", "Canada", "Australia", "South Korea"  # Desenvolupats variats
]

paisos_significants = list(set(top_emissors + top_hdi + representatius))
paisos_significants = [p for p in paisos_significants if p in df['Entity'].unique()]

# Permetre selecció manual
paisos_seleccionats = st.sidebar.multiselect(
    "Selecciona països (màxim 30)",
    sorted(paisos_significants),
    default=sorted(paisos_significants[:15])
)

if len(paisos_seleccionats) == 0:
    st.error("❌ Selecciona almenys 1 país!")
    st.stop()

# -----------------------------
# VARIABLES A PREDIR
# -----------------------------
variables_predir = {
    'co2_prod_pc': 'CO₂ per càpita (t)',
    'gdp_pc': 'PIB per càpita (USD)',
    'hdi': 'HDI',
    'life_exp': 'Esperança de vida (anys)',
    'population': 'Població',
    'pct_lowcarbon': '% Electricitat baixa emissió'
}

# -----------------------------
# ANYS DE PREDICCIÓ
# -----------------------------
any_base = 2023  # Últim any amb dades reals
anys_prediccio = [10, 25, 50, 75, 100]
anys_futurs = [any_base + a for a in anys_prediccio]

st.sidebar.info(f"**Any base:** {any_base}")
st.sidebar.info(f"**Prediccions:** {', '.join(map(str, anys_prediccio))} anys")

# -----------------------------
# MÈTODE DE PREDICCIÓ
# -----------------------------
metode = st.sidebar.selectbox(
    "Mètode de predicció",
    ["Tendència lineal", "Tendència polinòmica", "Random Forest"]
)

# -----------------------------
# FUNCIÓ DE PREDICCIÓ
# -----------------------------
def predir_futur(df_pais, variable, metode='lineal'):
    """
    Prediu valors futurs d'una variable per a un país.
    """
    # Filtrar dades amb la variable disponible
    data = df_pais[['Year', variable]].dropna()
    
    if len(data) < 5:  # Mínim 5 punts per fer predicció
        return None
    
    X = data[['Year']].values
    y = data[variable].values
    
    # Entrenar model segons mètode
    if metode == 'lineal':
        model = LinearRegression()
    elif metode == 'polinòmica':
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    else:  # Random Forest
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    model.fit(X, y)
    
    # Predir anys futurs
    X_futur = np.array(anys_futurs).reshape(-1, 1)
    prediccions = model.predict(X_futur)
    
    # Aplicar límits realistes
    if variable == 'hdi':
        prediccions = np.clip(prediccions, 0, 1)
    elif variable == 'co2_prod_pc':
        prediccions = np.maximum(prediccions, 0)  # No pot ser negatiu
    elif variable == 'life_exp':
        prediccions = np.clip(prediccions, 40, 120)
    elif variable == 'population':
        prediccions = np.maximum(prediccions, 0)
    elif variable == 'pct_lowcarbon':
        prediccions = np.clip(prediccions, 0, 100)
    
    return prediccions

# -----------------------------
# GENERAR PREDICCIONS
# -----------------------------
st.subheader("📊 Generant prediccions...")

resultats = []

progress_bar = st.progress(0)
total_paisos = len(paisos_seleccionats)

for idx, pais in enumerate(paisos_seleccionats):
    df_pais = df[df['Entity'] == pais].sort_values('Year')
    
    prediccions_pais = {'País': pais, 'Any_base': any_base}
    
    for var_key, var_name in variables_predir.items():
        preds = predir_futur(df_pais, var_key, 
                           'lineal' if metode == 'Tendència lineal' 
                           else 'polinòmica' if metode == 'Tendència polinòmica' 
                           else 'rf')
        
        if preds is not None:
            for i, any_futur in enumerate(anys_futurs):
                prediccions_pais[f'{var_key}_{any_futur}'] = preds[i]
        
        # Valor actual (any base)
        val_actual = df_pais[df_pais['Year'] == any_base][var_key].values
        if len(val_actual) > 0:
            prediccions_pais[f'{var_key}_{any_base}'] = val_actual[0]
    
    resultats.append(prediccions_pais)
    progress_bar.progress((idx + 1) / total_paisos)

df_prediccions = pd.DataFrame(resultats)

st.success(f"✅ Prediccions generades per {len(paisos_seleccionats)} països!")

# -----------------------------
# MOSTRAR RESULTATS
# -----------------------------

# Selector de variable
variable_mostrar = st.selectbox(
    "Selecciona variable a visualitzar",
    list(variables_predir.values())
)

var_key = [k for k, v in variables_predir.items() if v == variable_mostrar][0]

# -----------------------------
# GRÀFIC 1: EVOLUCIÓ TEMPORAL
# -----------------------------
st.subheader(f"📈 Evolució de {variable_mostrar}")

fig1, ax1 = plt.subplots(figsize=(14, 8))

colors = plt.cm.tab20(np.linspace(0, 1, len(paisos_seleccionats)))

for i, pais in enumerate(paisos_seleccionats):
    # Dades històriques
    df_pais = df[(df['Entity'] == pais) & (df['Year'] >= 1990)].sort_values('Year')
    
    if var_key in df_pais.columns:
        ax1.plot(df_pais['Year'], df_pais[var_key], 
                color=colors[i], linewidth=2, alpha=0.7, label=pais)
    
    # Prediccions futures
    fila_pais = df_prediccions[df_prediccions['País'] == pais]
    
    if not fila_pais.empty:
        anys_plot = [any_base] + anys_futurs
        valors_plot = [
            fila_pais[f'{var_key}_{any}'].values[0] 
            for any in anys_plot 
            if f'{var_key}_{any}' in fila_pais.columns
        ]
        
        if len(valors_plot) == len(anys_plot):
            ax1.plot(anys_plot, valors_plot, 
                    color=colors[i], linewidth=2, linestyle='--', alpha=0.9)
            ax1.scatter(anys_futurs, valors_plot[1:], 
                       color=colors[i], s=80, zorder=10, edgecolors='black', linewidth=1)

ax1.axvline(x=any_base, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Any base')
ax1.set_xlabel("Any", fontsize=12)
ax1.set_ylabel(variable_mostrar, fontsize=12)
ax1.set_title(f"Evolució històrica i prediccions de {variable_mostrar}", 
             fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(alpha=0.3)

st.pyplot(fig1)

# -----------------------------
# GRÀFIC 2: COMPARACIÓ PER ANY
# -----------------------------
st.subheader(f"📊 Comparació entre països per any")

any_comparar = st.select_slider(
    "Selecciona any per comparar",
    options=[any_base] + anys_futurs,
    value=2048  # 25 anys
)

fig2, ax2 = plt.subplots(figsize=(12, 8))

valors_any = []
paisos_any = []

for pais in paisos_seleccionats:
    fila = df_prediccions[df_prediccions['País'] == pais]
    col_name = f'{var_key}_{any_comparar}'
    
    if not fila.empty and col_name in fila.columns:
        valor = fila[col_name].values[0]
        if not np.isnan(valor):
            valors_any.append(valor)
            paisos_any.append(pais)

# Ordenar per valor
ordre = np.argsort(valors_any)[::-1]
valors_any = [valors_any[i] for i in ordre]
paisos_any = [paisos_any[i] for i in ordre]

# Colors segons valor
norm_vals = (np.array(valors_any) - min(valors_any)) / (max(valors_any) - min(valors_any) + 0.001)
colors_bars = plt.cm.RdYlGn_r(norm_vals)

bars = ax2.barh(paisos_any, valors_any, color=colors_bars, edgecolor='black', linewidth=0.5)

# Afegir valors a les barres
for bar, val in zip(bars, valors_any):
    ax2.text(bar.get_width() + max(valors_any)*0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=9)

ax2.set_xlabel(variable_mostrar, fontsize=12)
ax2.set_title(f"{variable_mostrar} previst per l'any {any_comparar}", 
             fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

st.pyplot(fig2)

# -----------------------------
# TAULA DE PREDICCIONS
# -----------------------------
st.subheader("📋 Taula de Prediccions Detallades")

# Preparar taula per mostrar
taula_mostrar = df_prediccions[['País']].copy()

for any in [any_base] + anys_futurs:
    col_name = f'{var_key}_{any}'
    if col_name in df_prediccions.columns:
        taula_mostrar[f'{any}'] = df_prediccions[col_name]

# Ordenar per valor actual
if f'{any_base}' in taula_mostrar.columns:
    taula_mostrar = taula_mostrar.sort_values(f'{any_base}', ascending=False)

st.dataframe(
    taula_mostrar.style.format({col: "{:.2f}" for col in taula_mostrar.columns if col != 'País'}),
    use_container_width=True,
    height=600
)

# -----------------------------
# CANVIS PERCENTUALS
# -----------------------------
st.subheader("📈 Canvis Percentuals Respecte Any Base")

canvis = pd.DataFrame()
canvis['País'] = df_prediccions['País']

for any in anys_futurs:
    col_base = f'{var_key}_{any_base}'
    col_futur = f'{var_key}_{any}'
    
    if col_base in df_prediccions.columns and col_futur in df_prediccions.columns:
        canvis[f'{any} (+{any-any_base} anys)'] = (
            (df_prediccions[col_futur] - df_prediccions[col_base]) / 
            df_prediccions[col_base] * 100
        )

st.dataframe(
    canvis.style.format({col: "{:+.1f}%" for col in canvis.columns if col != 'País'}),
    use_container_width=True
)

# -----------------------------
# HEATMAP DE CANVIS
# -----------------------------
st.subheader("🔥 Heatmap de Canvis Percentuals")

fig3, ax3 = plt.subplots(figsize=(12, len(paisos_seleccionats)*0.4 + 2))

# Preparar dades per heatmap
heatmap_data = canvis.set_index('País').values

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    center=0,
    xticklabels=[f'+{a}a' for a in anys_prediccio],
    yticklabels=canvis['País'],
    cbar_kws={'label': 'Canvi %'},
    ax=ax3
)

ax3.set_title(f"Canvi percentual de {variable_mostrar} respecte {any_base}", 
             fontsize=13, fontweight='bold')

st.pyplot(fig3)

# -----------------------------
# ESTADÍSTIQUES GLOBALS
# -----------------------------
st.subheader("🌍 Estadístiques Globals")

stats_globals = pd.DataFrame()

for any in [any_base] + anys_futurs:
    col_name = f'{var_key}_{any}'
    if col_name in df_prediccions.columns:
        vals = df_prediccions[col_name].dropna()
        stats_globals[str(any)] = {
            'Mitjana': vals.mean(),
            'Mediana': vals.median(),
            'Mínim': vals.min(),
            'Màxim': vals.max(),
            'Desv. Est.': vals.std()
        }

stats_globals = stats_globals.T

st.dataframe(
    stats_globals.style.format("{:.2f}"),
    use_container_width=True
)

# -----------------------------
# DESCÀRREGA DE RESULTATS
# -----------------------------
st.subheader("💾 Exportar Prediccions")

csv_complert = df_prediccions.to_csv(index=False)

st.download_button(
    label="📥 Descarregar totes les prediccions (CSV)",
    data=csv_complert,
    file_name=f"prediccions_co2_{metode.lower().replace(' ', '_')}.csv",
    mime="text/csv"
)

# Només la variable seleccionada
csv_variable = taula_mostrar.to_csv(index=False)

st.download_button(
    label=f"📥 Descarregar {variable_mostrar} (CSV)",
    data=csv_variable,
    file_name=f"prediccions_{var_key}_{metode.lower().replace(' ', '_')}.csv",
    mime="text/csv"
)

# -----------------------------
# ASSUMPTIONS I LIMITACIONS
# -----------------------------
with st.expander("⚠️ Assumptions i Limitacions del Model"):
    st.markdown(f"""
    ### Mètode utilitzat: **{metode}**
    
    #### Assumptions:
    - Les tendències històriques es mantenen en el futur
    - No hi ha esdeveniments disruptius (guerres, pandèmies, revolucions tecnològiques)
    - Les polítiques climàtiques actuals continuen sense canvis radicals
    - El creixement poblacional segueix projeccions demogràfiques estàndard
    
    #### Limitacions:
    - **Incertesa creixent:** Prediccions a 100 anys són extremadament incertes
    - **Dades històriques limitades:** Alguns països tenen menys de 30 anys de dades
    - **Factors no capturats:** Innovacions tecnològiques, canvis geopolítics, desastres naturals
    - **Correlacions espúries:** El model pot capturar relacions que no són causals
    
    #### Recomanacions:
    - Utilitza prediccions a **10-25 anys** com a orientatives
    - Prediccions a **50+ anys** són escenaris il·lustratius, no pronòstics reals
    - Combina amb altres models (IPCC, IEA, UN) per a decisions crítiques
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(f"""
🔮 **Predictor Temporal CO₂** | Mètode: {metode} | Països: {len(paisos_seleccionats)}  
📅 Any base: {any_base} | Horitzons: {', '.join([f'+{a}a' for a in anys_prediccio])}
""")