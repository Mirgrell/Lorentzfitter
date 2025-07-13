import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import pandas as pd

# -----------------------------------------------
# Wissenschaftliches Runden
# -----------------------------------------------

def sci_round(value, uncertainty):
    """
    Wissenschaftliche Rundung eines Messwerts mit Unsicherheit.
    """
    if uncertainty == 0 or np.isnan(uncertainty):
        return value, uncertainty, f"{value}"

    exponent = int(np.floor(np.log10(abs(uncertainty))))
    leading_digit = int(uncertainty / 10**exponent)

    sig_digits = 2 if leading_digit in [1, 2] else 1
    unc_rounded = round(uncertainty, -exponent + (sig_digits - 1))
    digits = max(0, -exponent + (sig_digits - 1))
    val_rounded = round(value, digits)

    fmt = f"{{0:.{digits}f}} ± {{1:.{digits}f}}"
    result_str = fmt.format(val_rounded, unc_rounded)

    return val_rounded, unc_rounded, result_str

# -----------------------------------------------
# Lorentzfunktionen
# -----------------------------------------------

def lorentz(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

def multi_lorentz(x, *params):
    y = np.zeros_like(x)
    num_peaks = len(params) // 3
    for i in range(num_peaks):
        x0 = params[i * 3]
        gamma = params[i * 3 + 1]
        A = params[i * 3 + 2]
        y += lorentz(x, x0, gamma, A)
    return y

# -----------------------------------------------
# Streamlit UI
# -----------------------------------------------

st.title("Multi-Peak Lorentz Fit")

# Spektrum laden oder simulieren?
source = st.radio(
    "Spektrum wählen:",
    ["Simuliertes Spektrum", "Spektrum laden (CSV)"]
)

# -----------------------------
# Simuliertes Spektrum
# -----------------------------

num_peaks = st.number_input(
    "Anzahl Peaks", min_value=1, max_value=10, value=3, step=1
)
noise_level = st.slider("Noise Level (σ)", 0.0, 1.0, 0.1, 0.01)

x0_list = []
gamma_list = []
A_list = []

st.subheader("Startwerte für Peaks")

for i in range(num_peaks):
    st.markdown(f"### Peak {i+1}")
    col1, col2, col3 = st.columns(3)

    with col1:
        x0 = st.slider(f"x₀ {i+1}", 0.0, 10.0, float(2 + i * 2), 0.1)
    with col2:
        gamma = st.slider(f"γ {i+1}", 0.1, 3.0, 0.5, 0.1)
    with col3:
        A = st.slider(f"A {i+1}", 0.1, 5.0, 1.0, 0.1)

    x0_list.append(x0)
    gamma_list.append(gamma)
    A_list.append(A)

if source == "Simuliertes Spektrum":
    x = np.linspace(0, 10, 1000)
    y_data = np.zeros_like(x)

    x00 = [1, 3, 6, 9]
    gamma0 = [0.1, 2.5, 0.3, 1.5]
    A0 = [3, 2.6, 0.7, 1.4]

    for x0, gamma, A in zip(x00, gamma0, A0):
        y_data += lorentz(x, x0, gamma, A)

    y_data += np.random.normal(0, noise_level, size=len(x))
    #np.savetxt("test.csv", np.column_stack((x, y_data)), delimiter=",", fmt='%s')
# -----------------------------
# Spektrum laden
# -----------------------------
else:
    st.info("Lade eine CSV-Datei mit zwei Spalten: x und y.")

    uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Minimalprüfung
        if df.shape[1] < 2:
            st.error("Die CSV muss mindestens zwei Spalten haben!")
            st.stop()

        st.write("Vorschau der Daten:")
        st.dataframe(df.head())

        x = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values

    else:
        st.warning("Bitte zuerst eine Datei hochladen.")
        st.stop()

# -----------------------------
# Plot Original Spektrum
# -----------------------------

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_data, mode='lines', name='Original Spektrum'))
for i, (x0, gamma, A) in enumerate(zip(x0_list, gamma_list, A_list)):
    y_peak = lorentz(x, x0, gamma, A)
    fig.add_trace(go.Scatter(
        x=x, y=y_peak,
        mode='lines',
        name=f"Peak {i+1}",
        line=dict(dash='dot')
    ))

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Fit durchführen?
# -----------------------------

st.subheader("Fit durchführen")


start_params = []
for x0, gamma, A in zip(x0_list, gamma_list, A_list):
    start_params += [x0, gamma, A]

if st.button("Fit starten"):
    try:
        popt, pcov = curve_fit(
            multi_lorentz,
            x,
            y_data,
            p0=start_params,
            maxfev=10000
        )
        y_fit = multi_lorentz(x, *popt)
        perr = np.sqrt(np.diag(pcov))
        ci_lower = multi_lorentz(x, *(popt - 2 * perr))
        ci_upper = multi_lorentz(x, *(popt + 2 * perr))

        fig.add_trace(go.Scatter(
            x=x, y=y_fit,
            mode='lines',
            name='Fit',
            line=dict(color='red', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

        # Ergebnisse ausgeben
        st.success("Fit erfolgreich!")
        result_table = ""
        for i in range(len(popt) // 3):
            x0_val, x0_unc, x0_str = sci_round(popt[i * 3], perr[i * 3])
            gamma_val, gamma_unc, gamma_str = sci_round(popt[i * 3 + 1], perr[i * 3 + 1])
            A_val, A_unc, A_str = sci_round(popt[i * 3 + 2], perr[i * 3 + 2])

            result_table += f"**Peak {i + 1}**\n"
            result_table += f"- x₀ = {x0_str}\n"
            result_table += f"- γ = {gamma_str}\n"
            result_table += f"- A = {A_str}\n\n"

        st.markdown(result_table)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler beim Fit: {e}")
