import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

df_p1 = pd.read_csv("P1_completo.csv", parse_dates=["time"])
df_p2 = pd.read_csv("P2_completo.csv", parse_dates=["time"])

# P1
time_p1 = df_p1["time"]
lat_p1 = df_p1["latitude"]
lon_p1 = df_p1["longitude"]
vhm0_p1 = df_p1["VHM0"]
vtpk_p1 = df_p1["VTPK"]
vmdr_p1 = df_p1["VMDR"]

# P2
time_p2 = df_p2["time"]
lat_p2 = df_p2["latitude"]
lon_p2 = df_p2["longitude"]
vhm0_p2 = df_p2["VHM0"]
vtpk_p2 = df_p2["VTPK"]
vmdr_p2 = df_p2["VMDR"]

def media_circular(grados):
    rad = np.deg2rad(grados.dropna())
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))
    angulo = np.rad2deg(np.arctan2(sin_mean, cos_mean))
    return angulo % 360  # angulo entre 0 y 360 grados 

def analizar_punto(df):
    # mAR MEDIO
    hm0_medio = df["VHM0"].mean()
    tp_medio = df["VTPK"].mean()
    dir_media = media_circular(df["VMDR"]) if "VMDR" in df.columns else np.nan

    # CLIMA EXTREMO
    idx_max = df["VHM0"].idxmax()
    fila_ext = df.loc[idx_max]

    return {
        "hm0_medio": round(hm0_medio, 4),
        "tp_medio": round(tp_medio, 4),
        "dir_media": round(dir_media, 4),
        "hm0_ext": round(fila_ext["VHM0"], 4),
        "tp_ext": round(fila_ext["VTPK"], 4),
        "dir_ext": round(fila_ext["VMDR"], 4) if "VMDR" in df.columns else np.nan,
        "fecha_ext": fila_ext["time"],
    }


resultados = {}

for punto, df in [("P1", df_p1), ("P2", df_p2)]:
    res = analizar_punto(df)
    resultados[punto] = res

    print(f"=== {punto} ===")
    print(f"Mar Medio: Hm0 = {res['hm0_medio']:.4f} m | Tp = {res['tp_medio']:.4f} s | Dir = {res['dir_media']:.4f}°")
    print(f"Mar Extremo: Hm0 = {res['hm0_ext']:.4f} m | Tp = {res['tp_ext']:.4f} s | Dir = {res['dir_ext']:.4f}° (Fecha: {res['fecha_ext']})\n")

# P1
hm0_medio_p1 = resultados["P1"]["hm0_medio"]
tp_medio_p1 = resultados["P1"]["tp_medio"]
dir_media_p1 = resultados["P1"]["dir_media"]
hm0_ext_p1 = resultados["P1"]["hm0_ext"]
tp_ext_p1 = resultados["P1"]["tp_ext"]
dir_ext_p1 = resultados["P1"]["dir_ext"]
fecha_ext_p1 = resultados["P1"]["fecha_ext"]

# P2
hm0_medio_p2 = resultados["P2"]["hm0_medio"]
tp_medio_p2 = resultados["P2"]["tp_medio"]
dir_media_p2 = resultados["P2"]["dir_media"]
hm0_ext_p2 = resultados["P2"]["hm0_ext"]
tp_ext_p2 = resultados["P2"]["tp_ext"]
dir_ext_p2 = resultados["P2"]["dir_ext"]
fecha_ext_p2 = resultados["P2"]["fecha_ext"]

g = 9.81 # gravedad
V_kn = 18 # velocidad en nudos
V = V_kn * 0.514 # Velocidad en m/s

escenarios = {
    "P1 - Mar Medio":   {"Hm0": hm0_medio_p1, "Tp": tp_medio_p1,  "Dir": dir_media_p1},
    "P1 - Mar Extremo": {"Hm0": hm0_ext_p1, "Tp": tp_ext_p1, "Dir": dir_ext_p1},
    "P2 - Mar Medio":   {"Hm0": hm0_medio_p2, "Tp": tp_medio_p2,  "Dir": dir_media_p2},
    "P2 - Mar Extremo": {"Hm0": hm0_ext_p2, "Tp": tp_ext_p2, "Dir": dir_ext_p2},
}

# Rubo de Evergreen Ever Ace para el tramo de ruta de Róterdam -> Hamburgo
rumbos = {"P1": 25.0, "P2": 75.0} # para poder calcular la frecuencia de encuentro del buque con lals olas

def jonswap_spectrum(w, Hm0, Tp, gamma=3.3):
    wp = 2 * np.pi / Tp
    sigma = np.where(w <= wp, 0.07, 0.09)
    a = np.exp(-((w - wp)**2) / (2 * (sigma**2) * (wp**2)))
    S_unscaled = (wp**4 / w**5) * np.exp(-1.25 * (wp / w)**4) * (gamma**a)
    m0_unscaled = np.trapezoid(S_unscaled, w)
    alpha = (Hm0**2) / (16.0 * m0_unscaled)
    return alpha * S_unscaled

w_fine = np.linspace(0.05, 2.5, 25000)
dw_fine = w_fine[1] - w_fine[0]
we_grid = np.linspace(0.05, 2.2, 450)
we_centers = 0.5 * (we_grid[:-1] + we_grid[1:])

espectros_encuentro = {}

for nombre, d in escenarios.items():
    punto = nombre.split(" - ")[0]
    rumbo = rumbos[punto]
    
    dir_prop = (d["Dir"] + 180.0) % 360.0
    diff = abs(rumbo - dir_prop) % 360.0
    mu_deg = 360.0 - diff if diff > 180.0 else diff
    mu_rad = np.deg2rad(mu_deg)
    
    # Espectro JONSWAP S(w)
    S_w = jonswap_spectrum(w_fine, d["Hm0"], d["Tp"])
    
    # Frecuencia de encuentro y conservación de energía S(we)*dwe = S(w)*dw
    w_e = np.abs(w_fine - (w_fine**2 * V / g) * np.cos(mu_rad))
    S_we_hist, _ = np.histogram(w_e, bins=we_grid, weights=S_w * dw_fine)
    S_we = S_we_hist / np.diff(we_grid)
    
    espectros_encuentro[nombre] = {"we": we_centers, "S_we": S_we, "mu": mu_deg}

    # Escala automática según la zona con energía significativa
    y_max = max(float(np.max(S_w)), float(np.max(S_we)))
    umbral = 0.005 * y_max
    w_activos = w_fine[S_w > umbral]
    we_activos = we_centers[S_we > umbral]
    x_min = max(0.0, min(float(w_activos[0]), float(we_activos[0])) - 0.15)
    x_max = min(2.2, max(float(w_activos[-1]), float(we_activos[-1])) + 0.15)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(w_fine, S_w, label=f"S(ω) Fijo (Hm0={d['Hm0']:.4f} m)", color="#1f77b4", lw=2.0)
    ax.plot(we_centers, S_we, label=f"S(ωe) Encuentro (μ={mu_deg:.2f}°, U={V_kn:.0f} kn)", color="#d62728", ls="--", lw=2.0)
    ax.set_title(f"{nombre} (Tp={d['Tp']:.4f} s, Dir={d['Dir']:.4f}°)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Frecuencia angular [rad/s]", fontsize=10.5)
    ax.set_ylabel("Densidad espectral [m²·s/rad]", fontsize=10.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max * 1.12)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15, ls=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, frameon=True)

    slug = nombre.lower().replace(" - ", "_").replace(" ", "_")
    plt.tight_layout()
    plt.savefig(f"step2_jonswap_{slug}.png", dpi=200, bbox_inches="tight")


# RAO

rho = 1025.0

# PARÁMETROS DEL EVER ACE
L = 399.98       # Eslora [m]
B = 61.50        # Manga [m]
T = 14.50        # Calado [m]
CB = 0.76        # Coeficiente de bloque estimado
CWP = 0.82       # Coeficiente de flotación
nabla = L * B * T * CB          # Volumen de carena [m3]
M = rho * nabla                 # Masa del buque [kg]
Delta = M * g                   # Desplazamiento en peso [N]
Awp = L * B * CWP               # Área de flotación [m2]

print(f'Volumen de carena: {nabla:.4f} m3')
print(f'Desplazamiento en peso: {Delta:.4f} N')
print(f'Área de flotación: {Awp:.4f} m2')

# Coeficientes hidrodinámicos y frecuencias naturales
A33 = 0.85 * M                  # Masa añadida en heave
C33 = rho * g * Awp             # Rigidez hidrostática en heave [N/m]
wn_heave = np.sqrt(C33 / (M + A33))
zeta_heave = 0.18

kyy = 0.25 * L
I55 = M * kyy**2
A55 = 0.80 * I55
GML = 520.0                     # Altura metacéntrica longitudinal [m]
C55 = Delta * GML
wn_pitch = np.sqrt(C55 / (I55 + A55))
zeta_pitch = 0.16

kxx = 0.38 * B
I44 = M * kxx**2
A44 = 0.25 * I44
GMT = 2.5                      # Altura metacéntrica transversal [m]
C44 = Delta * GMT
wn_roll = np.sqrt(C44 / (I44 + A44))
zeta_roll = 0.08


def calc_raos(we, mu_deg):
    mu = np.deg2rad(mu_deg)
    k = we**2 / g  # Número de onda
    
    arg_L = np.maximum(0.5 * k * L * np.abs(np.cos(mu)), 1e-6)
    f_L = np.abs(np.sin(arg_L) / arg_L)
    arg_B = np.maximum(0.5 * k * B * np.abs(np.sin(mu)), 1e-6)
    f_B = np.abs(np.sin(arg_B) / arg_B)
    
    # RAO Heave [m² / m²]
    r_h = we / wn_heave
    H_heave = (f_L * f_B) / np.sqrt((1 - r_h**2)**2 + (2 * zeta_heave * r_h)**2)
    RAO_heave = H_heave**2
    
    # RAO Pitch [deg² / m²]
    r_p = we / wn_pitch
    chi_pitch = np.where(arg_L < 1e-3, 1.0, 3.0 * np.abs(np.sin(arg_L) - arg_L * np.cos(arg_L)) / (arg_L**3))
    H_pitch_rad = (k * np.abs(np.cos(mu)) * chi_pitch * f_B) / np.sqrt((1 - r_p**2)**2 + (2 * zeta_pitch * r_p)**2)
    RAO_pitch = (np.rad2deg(H_pitch_rad))**2
    
    # RAO Roll [deg² / m²]
    r_r = we / wn_roll
    H_roll_rad = (k * np.abs(np.sin(mu)) * f_B * f_L) / np.sqrt((1 - r_r**2)**2 + (2 * zeta_roll * r_r)**2)
    RAO_roll = (np.rad2deg(H_roll_rad))**2
    
    return RAO_heave, RAO_pitch, RAO_roll

# 1. Verificación asintótica de RAOs en baja frecuencia (we -> 0)
rh0, rp0, rr0 = calc_raos(np.array([1e-4]), 128.9)
print(f"Check RAO(we->0): Heave = {rh0[0]:.4f} (esperado: 1.0) | Pitch = {rp0[0]:.2e} (esperado: 0)")

# 2. Gráficas individuales de los RAOs bien escaladas para el Punto 2 del Assignment
we_plot = np.linspace(0.01, 1.4, 2000)
raos_mu = [
    (128.9, "P1 Extremo (μ=128.9°)", "#1f77b4"),
    (72.7,  "P2 Extremo (μ=72.7°)",  "#ff7f0e"),
]
rao_configs = [
    ("Heave", "RAO Heave |H₃(ωe)|²", "[m²/m²]",   0, wn_heave, (0.0, 1.3), "step2_rao_heave.png"),
    ("Pitch", "RAO Pitch |H₅(ωe)|²", "[deg²/m²]", 1, wn_pitch, (0.0, 1.2), "step2_rao_pitch.png"),
    ("Roll",  "RAO Roll |H₄(ωe)|²",  "[deg²/m²]", 2, wn_roll,  (0.0, 0.5), "step2_rao_roll.png"),
]

# Precalcular RAOs para cada ángulo de encuentro
raos_por_mu = [(etiqueta, color, calc_raos(we_plot, mu_val)) for mu_val, etiqueta, color in raos_mu]

for nombre_dof, titulo, unidades, idx_dof, wn_dof, xlim, filename in rao_configs:
    fig, ax = plt.subplots(figsize=(8, 5))
    y_max = 0.0
    for etiqueta, color, rao_tuple in raos_por_mu:
        y = rao_tuple[idx_dof]
        ax.plot(we_plot, y, lw=2.0, color=color, label=etiqueta)
        y_max = max(y_max, float(np.max(y[we_plot <= xlim[1]])))

    ax.axvline(wn_dof, color="#555555", ls="--", lw=1.3, label=f"ωₙ {nombre_dof} = {wn_dof:.3f} rad/s")
    ax.set_title(f"{titulo} {unidades}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frecuencia de encuentro ωe [rad/s]", fontsize=10.5)
    ax.set_ylabel(f"|H(ωe)|² {unidades}", fontsize=10.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, y_max * 1.12)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15, ls=":")
    ax.legend(fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)

# ESPECTROS JONSWAP Y ESPECTROS DE RESPUESTA
U = 18.0 * 0.514444 
escenarios = {
    "P1 - Mar Medio":   {"Hm0": 1.31, "Tp": 6.35,  "Dir": 302.8},
    "P1 - Mar Extremo": {"Hm0": 6.19, "Tp": 10.76, "Dir": 333.9},
    "P2 - Mar Medio":   {"Hm0": 1.30, "Tp": 6.40,  "Dir": 305.1},
    "P2 - Mar Extremo": {"Hm0": 6.53, "Tp": 13.22, "Dir": 327.7},
}
rumbos = {"P1": 25.0, "P2": 75.0}

def jonswap_spectrum(w, Hm0, Tp, gamma=3.3):
    wp = 2 * np.pi / Tp
    sigma = np.where(w <= wp, 0.07, 0.09)
    a = np.exp(-((w - wp)**2) / (2 * (sigma**2) * (wp**2)))
    S_unscaled = (wp**4 / w**5) * np.exp(-1.25 * (wp / w)**4) * (gamma**a)
    m0_unscaled = np.trapezoid(S_unscaled, w)
    alpha = (Hm0**2) / (16.0 * m0_unscaled)
    return alpha * S_unscaled

w_fine = np.linspace(0.05, 2.5, 25000)
dw_fine = w_fine[1] - w_fine[0]
we_grid = np.linspace(0.05, 2.0, 450)
we = 0.5 * (we_grid[:-1] + we_grid[1:])

resultados = []
espectros_resp = {}

for nombre, d in escenarios.items():
    punto = nombre.split(" - ")[0]
    rumbo = rumbos[punto]
    dir_prop = (d["Dir"] + 180.0) % 360.0
    diff = abs(rumbo - dir_prop) % 360.0
    mu_deg = 360.0 - diff if diff > 180.0 else diff
    mu_rad = np.deg2rad(mu_deg)
    
    # Espectro de encuentro S(we)
    S_w = jonswap_spectrum(w_fine, d["Hm0"], d["Tp"])
    w_e_vals = np.abs(w_fine - (w_fine**2 * U / g) * np.cos(mu_rad))
    S_we_hist, _ = np.histogram(w_e_vals, bins=we_grid, weights=S_w * dw_fine)
    S_we = S_we_hist / np.diff(we_grid)
    
    # RAOs del buque
    RAO_h, RAO_p, RAO_r = calc_raos(we, mu_deg)
    
    # Multiplicación punto a punto: S_respuesta(we) = |H(we)|^2 * S_ola(we)
    S_resp_h = RAO_h * S_we
    S_resp_p = RAO_p * S_we
    S_resp_r = RAO_r * S_we
    
    # Espectros de velocidad y aceleración en Heave
    S_vel_h = (we**2) * S_resp_h
    S_acel_h = (we**4) * S_resp_h
    
    # Momentos espectrales m0 y amplitudes significativas (2 * sqrt(m0))
    m0_h = np.trapezoid(S_resp_h, we)
    m0_p = np.trapezoid(S_resp_p, we)
    m0_r = np.trapezoid(S_resp_r, we)
    m0_vh = np.trapezoid(S_vel_h, we)
    m0_ah = np.trapezoid(S_acel_h, we)
    
    resultados.append({
        "Escenario": nombre,
        "mu [°]": round(mu_deg, 1),
        "Heave sig [m]": round(2 * np.sqrt(m0_h), 3),
        "Vel Heave sig [m/s]": round(2 * np.sqrt(m0_vh), 3),
        "Acel Heave sig [m/s²]": round(2 * np.sqrt(m0_ah), 3),
        "Pitch sig [°]": round(2 * np.sqrt(m0_p), 2),
        "Roll sig [°]": round(2 * np.sqrt(m0_r), 2),
    })
    
    # Gráficas individuales de los espectros de respuesta bien escaladas
    slug_esc = nombre.lower().replace(" - ", "_").replace(" ", "_")
    for dof_slug, S_resp, titulo, ylabel, color in [
        ("heave", S_resp_h, f"{nombre} - Heave", "S_z [m²·s/rad]",   "#9467bd"),
        ("pitch", S_resp_p, f"{nombre} - Pitch", "S_θ [deg²·s/rad]", "#2ca02c"),
        ("roll",  S_resp_r, f"{nombre} - Roll",  "S_φ [deg²·s/rad]", "#ff7f0e"),
    ]:
        y_max = float(np.max(S_resp))
        activos = we[S_resp > 0.005 * y_max] if y_max > 0 else we
        x_min = max(0.0, float(activos[0]) - 0.12)
        x_max = min(1.6, float(activos[-1]) + 0.12)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(we, S_resp, color=color, lw=2.0)
        ax.fill_between(we, S_resp, color=color, alpha=0.2)
        ax.set_title(titulo, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Frecuencia de encuentro ωe [rad/s]", fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max * 1.12 if y_max > 0 else 1.0)
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15, ls=":")
        plt.tight_layout()
        plt.savefig(f"step3_resp_{slug_esc}_{dof_slug}.png", dpi=200)
        plt.close(fig)

    espectros_resp[nombre] = (S_resp_h, S_resp_p, S_resp_r)

for punto, esc_list, out_name in [
    ("Punto 1", ["P1 - Mar Medio", "P1 - Mar Extremo"], "step3_espectros_respuesta_p1.png"),
    ("Punto 2", ["P2 - Mar Medio", "P2 - Mar Extremo"], "step3_espectros_respuesta_p2.png"),
]:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row_idx, esc in enumerate(esc_list):
        S_h, S_p, S_r = espectros_resp[esc]
        for col_idx, (S_resp, dof_name, ylabel, color) in enumerate([
            (S_h, "Heave", "S_z [m²·s/rad]",   "#9467bd"),
            (S_p, "Pitch", "S_θ [deg²·s/rad]", "#2ca02c"),
            (S_r, "Roll",  "S_φ [deg²·s/rad]", "#ff7f0e"),
        ]):
            ax = axes[row_idx, col_idx]
            ax.plot(we, S_resp, color=color, lw=1.8)
            ax.fill_between(we, S_resp, color=color, alpha=0.2)
            ax.set_title(f"{esc} - {dof_name}", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlim(0.0, 1.6)
            ax.grid(True, alpha=0.3)
            if row_idx == 1:
                ax.set_xlabel("Frecuencia de encuentro ωe [rad/s]", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)

df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))