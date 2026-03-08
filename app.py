
"""
Phased Array Antenna Design Tool — CICAD 2025 Problem 1
=========================================================
Solves the design of a square-grid, square-aperture phased array
with microstrip patch elements at 14.5 GHz ± 1 GHz.

Equations follow:
  S. Rao & C. Ostroot, "Design Principles and Guidelines for
  Phased Array and Reflector Antennas," IEEE AP-Mag, Apr 2020.
"""

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── physical constants ──────────────────────────────────────────
c = 299792458.0  # speed of light, m/s

# ───────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phased Array Antenna Designer",
    page_icon="📡",
    layout="wide",
)

# ───────────────────────────────────────────────────────────────
# CUSTOM CSS for polished, light look
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f9fbff 0%, #e3edf9 100%);
        border: 1px solid #d0d8e8;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(15, 25, 35, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #48658c !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #132033 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f7fb 0%, #e6edf7 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #27436b;
        font-family: 'DM Sans', sans-serif;
    }

    /* Headers */
    h1, h2, h3, h4 {
        font-family: 'DM Sans', sans-serif !important;
        color: #132033;
    }

    /* Expander / section headers */
    .stExpander {
        border: 1px solid #d0d8e8;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# SIDEBAR — Input Parameters
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Input Parameters")
    st.markdown("---")

    f_center_ghz = st.number_input(
        "Center Frequency (GHz)", min_value=1.0, max_value=100.0,
        value=14.5, step=0.5, format="%.1f",
        help="Center frequency of the operating band."
    )
    bw_ghz = st.number_input(
        "Bandwidth ± (GHz)", min_value=0.0, max_value=20.0,
        value=1.0, step=0.1, format="%.1f",
        help="Half-bandwidth. Band = center ± this value."
    )
    eta_element = st.slider(
        "Element Efficiency (%)", 10, 100, 90, 1,
        help="Aperture efficiency of the individual patch element."
    )
    theta_max_deg = st.number_input(
        "Max Scan Angle (°)", min_value=0.0, max_value=89.0,
        value=45.0, step=1.0, format="%.1f",
        help="Maximum scan angle from boresight."
    )
    theta_g_deg = st.number_input(
        "Grating Lobe Margin Angle (°)", min_value=0.0, max_value=90.0,
        value=47.0, step=1.0, format="%.1f",
        help="Angle slightly beyond scan edge to avoid grating lobes."
    )
    G_min_dBi = st.number_input(
        "Min Gain over Scan Range (dBi)", min_value=0.0, max_value=60.0,
        value=20.0, step=0.5, format="%.1f",
        help="Minimum gain required at the edge of scan."
    )
    T_illumination_dB = st.number_input(
        "Edge Illumination Taper (dB)", min_value=0.0, max_value=20.0,
        value=0.0, step=0.5, format="%.1f",
        help="0 dB = uniform illumination."
    )
    antenna_loss_dB = st.number_input(
        "Front-end / Antenna Loss (dB)", min_value=0.0, max_value=10.0,
        value=0.5, step=0.1, format="%.1f",
        help="Includes mismatch, polarization, insertion losses."
    )
    pointing_error_loss_dB = st.number_input(
        "Pointing Error Loss (dB)", min_value=0.0, max_value=5.0,
        value=0.0, step=0.1, format="%.1f"
    )
    loss_beam_diameter_dB = st.number_input(
        "Loss over Beam Diameter (dB)", min_value=0.0, max_value=5.0,
        value=3.0, step=0.5, format="%.1f",
        help="Typically 3 dB if full beam, 0 dB if beam-peak only."
    )
    implementation_margin_dB = st.number_input(
        "Implementation Margin (dB)", min_value=0.0, max_value=5.0,
        value=0.5, step=0.1, format="%.1f",
        help="Margin for thermal, failures, amp/phase errors (typ. 0.5–1 dB)."
    )

    st.markdown("---")
    st.markdown("##### 🔬 Problem Defaults")
    st.caption("Freq 14.5 ± 1 GHz · Square grid · Patch 90% · θ_max 45° · G_min 20 dBi · Uniform illum.")

# ───────────────────────────────────────────────────────────────
# CORE CALCULATIONS
# ───────────────────────────────────────────────────────────────
f_center = f_center_ghz * 1e9          # Hz
f_low = (f_center_ghz - bw_ghz) * 1e9
f_high = (f_center_ghz + bw_ghz) * 1e9

lambda_nom = c / f_center              # wavelength at center freq
lambda_low = c / f_low                 # longest wavelength (lowest freq)
lambda_high = c / f_high               # shortest wavelength (highest freq) — for grating lobes

theta_max_rad = math.radians(theta_max_deg)
theta_g_rad = math.radians(theta_g_deg)
eta = eta_element / 100.0              # element efficiency as fraction

# ── Illumination taper efficiency (parabolic on pedestal, n=1) ──
T_lin = 10 ** (-T_illumination_dB / 20.0)  # pedestal voltage
if T_illumination_dB == 0.0:
    eta_taper = 1.0       # uniform
    taper_loss_dB = 0.0
else:
    eta_taper = 75.0 * ((1 + T_lin) ** 2) / (1 + T_lin + T_lin ** 2) / 100.0
    taper_loss_dB = -10.0 * math.log10(eta_taper)

# ── Part A: Required Peak Directivity ───────────────────────
# Scan loss for small elements (d ~ 0.5λ): SL = 10 log10(cos^n θ), n≈1.5
n_scan = 1.5
scan_loss_dB = -10.0 * math.log10(math.cos(theta_max_rad) ** n_scan)

# Peak directivity (Eq 5 from Rao & Ostroot)
# D_p = G_min + L_s + SL + GL_pe + T_L + X + I_m
D_peak_dBi = (G_min_dBi + antenna_loss_dB + scan_loss_dB +
              pointing_error_loss_dB + taper_loss_dB +
              loss_beam_diameter_dB + implementation_margin_dB)

D_peak_linear = 10.0 ** (D_peak_dBi / 10.0)

# ── Part B: Element Spacing & Element Gain ──────────────────
# Square lattice grating-lobe-free condition (Eq 1 with θ_G):
#   d / λ_h  ≤  1 / (sin θ_sm + sin θ_G)
d_over_lambda = 1.0 / (math.sin(theta_max_rad) + math.sin(theta_g_rad))
d_spacing = d_over_lambda * lambda_high          # physical spacing (m)
d_spacing_mm = d_spacing * 1000.0

# Element directivity (Eq 3 — unit cell area, lowest freq for directivity)
A_cell = d_spacing ** 2                            # square lattice unit cell
D_el_linear = eta * 4.0 * math.pi * A_cell / (lambda_low ** 2)
D_el_dBi = 10.0 * math.log10(D_el_linear)

G_el_dBi = D_el_dBi   # gain ≈ directivity × efficiency already included

# ── Part C: Number of Elements & Array Size ─────────────────
# N = 10^(0.1*D_p - 0.1*D_e)  (Eq 4)
N_required = 10.0 ** (0.1 * D_peak_dBi - 0.1 * D_el_dBi)
N_side = math.ceil(math.sqrt(N_required))
N_total = N_side * N_side

L_aperture = N_side * d_spacing                    # total aperture side (m)
L_aperture_mm = L_aperture * 1000.0

# Actual achieved peak directivity with N_total elements
D_actual_linear = N_total * D_el_linear * eta_taper
D_actual_dBi = 10.0 * math.log10(D_actual_linear)

# Achieved directivity at scan edge
D_scan_dBi = D_actual_dBi - scan_loss_dB

# Achieved gain at scan edge
G_scan_dBi = D_scan_dBi - antenna_loss_dB - pointing_error_loss_dB - implementation_margin_dB - loss_beam_diameter_dB

# Grating lobe locations
gl_ratio = lambda_high / d_spacing
if gl_ratio <= 1.0:
    gl_boresight_deg = math.degrees(math.asin(gl_ratio))
else:
    gl_boresight_deg = 90.0

gl_scan_arg = lambda_high / d_spacing - math.sin(theta_max_rad)
if -1.0 <= gl_scan_arg <= 1.0:
    gl_scan_deg = math.degrees(math.asin(gl_scan_arg))
else:
    gl_scan_deg = 90.0

# ── Half-power beamwidth (approximate for square aperture) ──
# θ_3 ≈ 0.886 λ / L  (radians) for uniform illumination
hpbw_rad = 0.886 * lambda_nom / L_aperture
hpbw_deg = math.degrees(hpbw_rad)

# First Sidelobe Level (Uniform square aperture)
first_sidelobe_dB = -13.26

# ───────────────────────────────────────────────────────────────
# TITLE
# ───────────────────────────────────────────────────────────────
st.markdown("# 📡 Phased Array Antenna Design Tool")
st.markdown("**CICAD 2025 — Assignment Problem 1** · Square Grid · Microstrip Patch · 14.5 GHz ± 1 GHz")
st.markdown("---")

# ───────────────────────────────────────────────────────────────
# PART A — Peak Directivity
# ───────────────────────────────────────────────────────────────
st.markdown("### Part (a): Required Peak Directivity")

col_a1, col_a2, col_a3, col_a4 = st.columns(4)
col_a1.metric("Min Gain Required", f"{G_min_dBi:.1f} dBi")
col_a2.metric("Scan Loss @ {:.0f}°".format(theta_max_deg), f"{scan_loss_dB:.2f} dB")
col_a3.metric("Taper Loss", f"{taper_loss_dB:.2f} dB")
col_a4.metric("**Peak Directivity Dₚ**", f"{D_peak_dBi:.2f} dBi")

st.info(
    f"**Calculation:** Dₚ = G_min + Lₛ + SL + GL_pe + T_L + X + Iₘ  \n"
    f"= {G_min_dBi} + {antenna_loss_dB} + {scan_loss_dB:.2f} + {pointing_error_loss_dB} "
    f"+ {taper_loss_dB:.2f} + {loss_beam_diameter_dB} + {implementation_margin_dB}  \n"
    f"= **{D_peak_dBi:.2f} dBi** \n\n"
    f"Scan loss uses SL = 10·log₁₀(cos^{n_scan:.1f}(θ)) with θ_max = {theta_max_deg}°."
)

st.markdown("---")

# ───────────────────────────────────────────────────────────────
# PART B — Element Spacing & Element Gain
# ───────────────────────────────────────────────────────────────
st.markdown("### Part (b): Element Spacing & Element Gain")

col_b1, col_b2, col_b3, col_b4 = st.columns(4)
col_b1.metric("d / λ_high", f"{d_over_lambda:.4f}")
col_b2.metric("Element Spacing d", f"{d_spacing_mm:.2f} mm")
col_b3.metric("Element Directivity Dₑ", f"{D_el_dBi:.2f} dBi")
col_b4.metric("λ_high (mm)", f"{lambda_high*1000:.2f}")

st.info(
    f"**Grating-lobe-free condition (square lattice):** \n"
    f"d / λ_h ≤ 1 / (sin θ_sm + sin θ_G) = 1 / (sin {theta_max_deg}° + sin {theta_g_deg}°) = **{d_over_lambda:.4f}** \n"
    f"d = {d_over_lambda:.4f} × {lambda_high*1000:.2f} mm = **{d_spacing_mm:.2f} mm** \n\n"
    f"**Element directivity:** Dₑ = 10·log₁₀(η_e · 4π · (d/λ_low)²) = "
    f"10·log₁₀({eta:.2f} × 4π × ({d_spacing_mm/1000:.4f}/{lambda_low:.4f})²) → **{D_el_dBi:.2f} dBi**"
)

st.markdown("---")

# ───────────────────────────────────────────────────────────────
# PART C — Number of Elements, Layout, Patterns
# ───────────────────────────────────────────────────────────────
st.markdown("### Part (c): Array Configuration & Radiation Patterns")

col_c1, col_c2, col_c3, col_c4 = st.columns(4)
col_c1.metric("Elements Required N", f"{N_required:.1f}")
col_c2.metric("Per Side N_side", f"{N_side}")
col_c3.metric("Total Elements", f"{N_total}")
col_c4.metric("Aperture Size", f"{L_aperture_mm:.1f} × {L_aperture_mm:.1f} mm")

col_c5, col_c6, col_c7, col_c8 = st.columns(4)
col_c5.metric("Achieved Dₚ (boresight)", f"{D_actual_dBi:.2f} dBi")
col_c6.metric("Directivity @ scan edge", f"{D_scan_dBi:.2f} dBi")
col_c7.metric("Gain @ scan edge", f"{G_scan_dBi:.2f} dBi")
col_c8.metric("First Sidelobe Level", f"{first_sidelobe_dB} dB")

st.info(
    f"**N** = 10^(0.1·Dₚ − 0.1·Dₑ) = 10^(0.1×{D_peak_dBi:.2f} − 0.1×{D_el_dBi:.2f}) "
    f"= **{N_required:.1f}** \n"
    f"Square array: N_side = ⌈√{N_required:.1f}⌉ = **{N_side}** → N_total = {N_side}² = **{N_total}** \n"
    f"Aperture = {N_side} × {d_spacing_mm:.2f} mm = **{L_aperture_mm:.1f} mm** per side  \n"
    f"First Sidelobe Level for uniform square aperture is **{first_sidelobe_dB} dB**."
)

# ───────────────────────────────────────────────────────────────
# PLOTS
# ───────────────────────────────────────────────────────────────

# Color palette (light theme)
C_BG = "#ffffff"
C_GRID = "#dde3ee"
C_LINE1 = "#0052cc"
C_LINE2 = "#ff6b6b"
C_LINE3 = "#2f9e44"
C_TEXT = "#334155"
C_ACCENT = "#f59e0b"

def style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply consistent dark styling to an axis."""
    ax.set_facecolor(C_BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=C_TEXT, fontsize=9)
    ax.set_ylabel(ylabel, color=C_TEXT, fontsize=9)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.grid(True, color=C_GRID, alpha=0.5, linewidth=0.5)

# ── Array Factor computation ────────────────────────────────
@st.cache_data
def compute_array_factor(N_side, d_m, wavelength, scan_deg, theta_range_deg):
    theta_rad = np.radians(theta_range_deg)
    scan_rad = np.radians(scan_deg)
    k = 2.0 * np.pi / wavelength

    psi = k * d_m * (np.sin(theta_rad) - np.sin(scan_rad))

    half_psi = psi / 2.0
    N = N_side
    numerator = np.sin(N * half_psi)
    denominator = N * np.sin(half_psi)

    with np.errstate(divide='ignore', invalid='ignore'):
        af = np.where(np.abs(denominator) < 1e-12, 1.0, numerator / denominator)

    af_power = np.abs(af) ** 2
    af_power_db = 10.0 * np.log10(np.maximum(af_power, 1e-15))

    return af_power_db

theta_range = np.linspace(-90, 90, 3601)

af_boresight = compute_array_factor(N_side, d_spacing, lambda_nom, 0.0, theta_range)
af_scanned = compute_array_factor(N_side, d_spacing, lambda_nom, theta_max_deg, theta_range)

element_pattern_dB = 10.0 * np.log10(np.maximum(np.cos(np.radians(theta_range)) ** n_scan, 1e-15))

total_boresight = af_boresight + element_pattern_dB
total_scanned = af_scanned + element_pattern_dB

total_boresight_dBi = total_boresight + D_actual_dBi
total_scanned_dBi = total_scanned + D_actual_dBi

y_floor = -40
total_boresight_clipped = np.clip(total_boresight, y_floor, 0)
total_scanned_clipped = np.clip(total_scanned, y_floor, 0)

# ── FIGURE: 2×2 plots ──────────────────────────────────────
fig = plt.figure(figsize=(14, 11), facecolor=C_BG)
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

# ── Plot 1: Radiation pattern — Boresight ────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Normalized Pattern — Boresight Beam", "θ (degrees)", "Normalized Gain (dB)")
ax1.plot(theta_range, total_boresight_clipped, color=C_LINE1, linewidth=1.0, label="Boresight")
ax1.set_xlim(-90, 90)
ax1.set_ylim(y_floor, 3)
ax1.axhline(first_sidelobe_dB, color=C_ACCENT, linestyle='--', linewidth=0.7, alpha=0.7, label=f"1st SLL ({first_sidelobe_dB} dB)")
ax1.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# ── Plot 2: Radiation pattern — Scanned beam ─────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, f"Normalized Pattern — Beam Scanned to {theta_max_deg:.0f}°", "θ (degrees)", "Normalized Gain (dB)")
ax2.plot(theta_range, total_scanned_clipped, color=C_LINE2, linewidth=1.0, label=f"Scanned {theta_max_deg:.0f}°")
ax2.set_xlim(-90, 90)
ax2.set_ylim(y_floor, 3)
ax2.axhline(-3, color=C_ACCENT, linestyle='--', linewidth=0.7, alpha=0.7, label="-3 dB")
ax2.axvline(theta_max_deg, color=C_LINE3, linestyle=':', linewidth=0.7, alpha=0.7, label=f"θ = {theta_max_deg:.0f}°")
ax2.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# ── Plot 3: Directivity patterns (dBi) ──────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Directivity Patterns (dBi)", "θ (degrees)", "Directivity (dBi)")
mask_boresight = total_boresight_dBi > (D_actual_dBi + y_floor)
mask_scanned = total_scanned_dBi > (D_actual_dBi + y_floor)
ax3.plot(theta_range[mask_boresight], total_boresight_dBi[mask_boresight],
         color=C_LINE1, linewidth=1.0, label="Boresight", alpha=0.9)
ax3.plot(theta_range[mask_scanned], total_scanned_dBi[mask_scanned],
         color=C_LINE2, linewidth=1.0, label=f"Scanned {theta_max_deg:.0f}°", alpha=0.9)
ax3.axhline(G_min_dBi, color=C_ACCENT, linestyle='--', linewidth=0.8, alpha=0.8, label=f"G_min = {G_min_dBi} dBi")
ax3.set_xlim(-90, 90)
ax3.set_ylim(D_actual_dBi - 45, D_actual_dBi + 3)
ax3.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

# ── Plot 4: Array Element Layout ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(C_BG)
ax4.set_title("Square Array Layout", color="white", fontsize=11, fontweight="bold", pad=10)
ax4.set_xlabel("x (mm)", color=C_TEXT, fontsize=9)
ax4.set_ylabel("y (mm)", color=C_TEXT, fontsize=9)
ax4.tick_params(colors=C_TEXT, labelsize=8)
for spine in ax4.spines.values():
    spine.set_color(C_GRID)
ax4.set_aspect('equal')

positions_1d = np.arange(N_side) * d_spacing_mm
positions_1d -= positions_1d.mean()  # center
xx, yy = np.meshgrid(positions_1d, positions_1d)

radius_mm = d_spacing_mm * 0.38
for xi, yi in zip(xx.ravel(), yy.ravel()):
    circle = plt.Circle((xi, yi), radius_mm, fill=True,
                         facecolor="#1a5276", edgecolor=C_LINE1,
                         linewidth=0.4, alpha=0.75)
    ax4.add_patch(circle)

ax4.scatter(xx, yy, s=2, color=C_LINE1, zorder=5)

margin = d_spacing_mm * 1.2
ax4.set_xlim(positions_1d[0] - margin, positions_1d[-1] + margin)
ax4.set_ylim(positions_1d[0] - margin, positions_1d[-1] + margin)
ax4.grid(True, color=C_GRID, alpha=0.3, linewidth=0.3)

ax4.text(0.02, 0.98, f"N = {N_total} ({N_side}×{N_side})\n"
         f"d = {d_spacing_mm:.2f} mm\n"
         f"L = {L_aperture_mm:.1f} mm",
         transform=ax4.transAxes, fontsize=8, color=C_ACCENT,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.4', facecolor=C_BG, edgecolor=C_GRID, alpha=0.9))

st.pyplot(fig)

# ───────────────────────────────────────────────────────────────
# POLAR PLOT
# ───────────────────────────────────────────────────────────────
st.markdown("### Polar Radiation Patterns")

fig_polar, (ax_p1, ax_p2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                                          figsize=(12, 5), facecolor=C_BG)

for ax_p, data, title, color in [
    (ax_p1, total_boresight_clipped, "Boresight", C_LINE1),
    (ax_p2, total_scanned_clipped, f"Scanned {theta_max_deg:.0f}°", C_LINE2),
]:
    ax_p.set_facecolor(C_BG)
    theta_plot = np.radians(theta_range)
    r = data - y_floor
    ax_p.plot(theta_plot, r, color=color, linewidth=0.8)
    ax_p.fill_between(theta_plot, 0, r, alpha=0.15, color=color)
    ax_p.set_thetamin(-90)
    ax_p.set_thetamax(90)
    ax_p.set_theta_zero_location('N')
    ax_p.set_title(title, color="white", fontsize=10, fontweight="bold", pad=15)
    ax_p.tick_params(colors=C_TEXT, labelsize=7)
    ax_p.set_rlabel_position(60)
    ax_p.grid(True, color=C_GRID, alpha=0.4)
    r_ticks = np.array([0, 10, 20, 30, 40])
    ax_p.set_rticks(r_ticks)
    ax_p.set_yticklabels([f"{int(v + y_floor)}" for v in r_ticks], fontsize=7, color=C_TEXT)

fig_polar.tight_layout()
st.pyplot(fig_polar)

# ───────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ───────────────────────────────────────────────────────────────
st.markdown("### 📋 Design Summary")

summary_data = {
    "Parameter": [
        "Operating Band",
        "Center Wavelength λ₀",
        "Highest Freq Wavelength λ_h",
        "Lowest Freq Wavelength λ_l",
        "Max Scan Angle θ_sm",
        "Scan Loss (cos^1.5 model)",
        "Element Efficiency",
        "Taper Efficiency",
        "Required Peak Directivity Dₚ",
        "Element Spacing d (d/λ_h)",
        "Element Spacing d (mm)",
        "Element Directivity Dₑ",
        "Required No. of Elements",
        "Array Grid",
        "Total Elements N",
        "Achieved Peak Directivity",
        "Directivity at Scan Edge",
        "Gain at Scan Edge",
        "HPBW (boresight)",
        "First Sidelobe (Square Aperture)",
        "Grating Lobe @ Boresight",
        "Grating Lobe @ Max Scan",
        f"Aperture Size",
    ],
    "Value": [
        f"{f_low/1e9:.1f} – {f_high/1e9:.1f} GHz",
        f"{lambda_nom*1000:.2f} mm",
        f"{lambda_high*1000:.2f} mm",
        f"{lambda_low*1000:.2f} mm",
        f"{theta_max_deg:.1f}°",
        f"{scan_loss_dB:.2f} dB",
        f"{eta_element}%",
        f"{eta_taper*100:.1f}%",
        f"{D_peak_dBi:.2f} dBi",
        f"{d_over_lambda:.4f} λ_h",
        f"{d_spacing_mm:.2f} mm",
        f"{D_el_dBi:.2f} dBi",
        f"{N_required:.1f}",
        f"{N_side} × {N_side} (square)",
        f"{N_total}",
        f"{D_actual_dBi:.2f} dBi",
        f"{D_scan_dBi:.2f} dBi",
        f"{G_scan_dBi:.2f} dBi",
        f"{hpbw_deg:.2f}°",
        f"{first_sidelobe_dB} dB",
        f"{gl_boresight_deg:.1f}°",
        f"{gl_scan_deg:.1f}°",
        f"{L_aperture_mm:.1f} × {L_aperture_mm:.1f} mm ({L_aperture*100:.2f} × {L_aperture*100:.2f} cm)",
    ]
}

import pandas as pd
df_summary = pd.DataFrame(summary_data)
st.table(df_summary)

# ───────────────────────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
   
    "**CICAD 2025 Internship Assignment** — Phased Array Problem 1"
)
