import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Mathematical Formulations ---
def calculate_phased_array(eval_freq_ghz, max_scan_deg, target_min_gain_dbi, element_efficiency_pct):
    # a. Calculate the required peak directivity
    # Evaluate at the highest frequency
    freq_hz = eval_freq_ghz * 1e9
    c = 299792458
    wavelength = c / freq_hz
    
    max_scan_rad = math.radians(max_scan_deg)
    
    # Uniform illumination implies taper loss = 0, efficiency = 100% (taper)
    # Scan loss using standard cos^1.5(theta) approximation
    scan_loss_db = -10 * math.log10((math.cos(max_scan_rad))**1.5)
    
    # Required Peak Directivity (Boresight) to meet minimum gain at max scan
    peak_directivity_req = target_min_gain_dbi + scan_loss_db
    
    # b. Calculate element spacing and element gain
    # To strictly avoid grating lobes in real space at max scan:
    # d/lambda <= 1 / (1 + sin(max_scan_angle))
    spacing_wl = 1 / (1 + math.sin(max_scan_rad))
    spacing_meters = spacing_wl * wavelength
    
    # Element Directivity (Gain)
    element_efficiency = element_efficiency_pct / 100.0
    element_directivity_linear = 4 * math.pi * element_efficiency * (spacing_wl**2)
    element_directivity_dbi = 10 * math.log10(element_directivity_linear)
    
    # c. Calculate required number of elements and physical size
    N_required_exact = 10**((peak_directivity_req - element_directivity_dbi) / 10)
    
    # Since it's a square aperture, N must be a perfect square
    elements_per_side = math.ceil(math.sqrt(N_required_exact))
    total_elements = elements_per_side**2
    
    achieved_boresight_dbi = 10 * math.log10(total_elements * element_directivity_linear)
    achieved_scan_dbi = achieved_boresight_dbi - scan_loss_db
    
    aperture_side_length_meters = elements_per_side * spacing_meters
    
    return {
        "wavelength_m": wavelength,
        "scan_loss_db": scan_loss_db,
        "peak_directivity_req": peak_directivity_req,
        "spacing_wl": spacing_wl,
        "spacing_mm": spacing_meters * 1000,
        "element_directivity_dbi": element_directivity_dbi,
        "n_exact": N_required_exact,
        "elements_per_side": elements_per_side,
        "total_elements": total_elements,
        "achieved_boresight_dbi": achieved_boresight_dbi,
        "achieved_scan_dbi": achieved_scan_dbi,
        "aperture_side_mm": aperture_side_length_meters * 1000
    }

def plot_array_layout(elements_per_side, spacing_mm):
    w = np.arange(elements_per_side) * spacing_mm
    x = np.repeat(w, elements_per_side)
    y = np.tile(w, elements_per_side)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.scatter(x, y, color='#1f77b4', s=50)
    
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')
    ax.set_title(f'Square Array Layout ({elements_per_side}x{elements_per_side} Elements)')
    return fig

def plot_radiation_pattern(elements_per_side, spacing_wl, achieved_boresight_dbi):
    theta_deg = np.linspace(-90, 90, 1000)
    theta_rad = np.radians(theta_deg)
    
    L_wl = elements_per_side * spacing_wl 
    
    argument = L_wl * np.sin(theta_rad)
    
    sinc_vals = np.sinc(argument)**2
    sinc_vals = np.where(sinc_vals < 1e-10, 1e-10, sinc_vals)
    
    pattern_dbi = achieved_boresight_dbi + 10 * np.log10(sinc_vals)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(theta_deg, pattern_dbi, color='red')
    ax.set_xlabel('Scan Angle $\\theta$ (Degrees)')
    ax.set_ylabel('Directivity (dBi)')
    ax.set_title('Array Radiation Pattern (Broadside)')
    ax.set_ylim([max(0, achieved_boresight_dbi - 50), achieved_boresight_dbi + 2])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

# --- Streamlit UI ---
st.set_page_config(page_title="Phased Array Designer", layout="wide")
st.title("Phased Array Antenna Design Application")

st.sidebar.header("Design Specifications")

# Fill-in number inputs for frequency, angles, and gain
center_freq_ghz = st.sidebar.number_input("Center Frequency (GHz)", value=14.5, step=0.1)
freq_threshold_ghz = st.sidebar.number_input("Frequency Threshold (+/- GHz)", value=1.0, step=0.1)
max_scan_deg = st.sidebar.number_input("Maximum Scan Angle (°)", value=45.0, step=1.0)
target_min_gain_dbi = st.sidebar.number_input("Minimum Gain over Scan (dBi)", value=20.0, step=1.0)

# Dropdowns for single-type categorical options
grid_shape = st.sidebar.selectbox("Grid", ["Square"])
aperture_shape = st.sidebar.selectbox("Aperture Shape", ["Square"])
feed_element = st.sidebar.selectbox("Feed Element", ["Microstrip Patch Antenna"])
element_efficiency_pct = st.sidebar.selectbox("Element Efficiency (%)", [90.0])
illumination = st.sidebar.selectbox("Illumination", ["Uniform"])

# Calculate worst-case evaluation frequency
eval_freq_ghz = center_freq_ghz + freq_threshold_ghz

if st.sidebar.button("Calculate Design"):
    results = calculate_phased_array(eval_freq_ghz, max_scan_deg, target_min_gain_dbi, element_efficiency_pct)
    
    st.header("A. Directivity & Gain Requirements")
    st.write(f"To maintain a minimum gain of **{target_min_gain_dbi} dBi** at a {max_scan_deg}° scan angle, we must account for scan loss.")
    st.write(f"**Calculated Scan Loss:** {results['scan_loss_db']:.2f} dB")
    st.success(f"**Required Peak Directivity (Boresight):** {results['peak_directivity_req']:.2f} dBi")
    
    st.header("B. Element Spacing & Gain")
    st.info(f"**Note:** To ensure grating lobes never enter real space across the entire frequency band, the element spacing is evaluated at the highest operational frequency: **{eval_freq_ghz} GHz** ({center_freq_ghz} GHz + {freq_threshold_ghz} GHz).")
    
    
    
    st.write("To completely avoid grating lobes entering real space during the maximum scan, the element spacing $d/\\lambda$ must satisfy:")
    st.latex(r"d/\lambda \le \frac{1}{1 + \sin(\theta_{max})}")
    
    col1, col2 = st.columns(2)
    col1.success(f"**Maximum Element Spacing ($d/\\lambda$):** {results['spacing_wl']:.3f} $\\lambda$ \n\n **Physical Spacing ($d$):** {results['spacing_mm']:.2f} mm")
    col2.success(f"**Element Directivity:** {results['element_directivity_dbi']:.2f} dBi")
    
    st.header("C. Array Layout & Physical Aperture")
    st.write(f"Based on the required peak directivity and element directivity, the exact continuous number of elements needed is {results['n_exact']:.1f}. Because the aperture is a **{aperture_shape}** grid, we round up to the nearest perfect square.")
    
    st.success(f"**Required Number of Elements:** {results['total_elements']} ({results['elements_per_side']} x {results['elements_per_side']} grid)")
    st.write(f"**Total Physical Aperture Size:** {results['aperture_side_mm']:.2f} mm x {results['aperture_side_mm']:.2f} mm")
    st.write(f"**Achieved Boresight Directivity:** {results['achieved_boresight_dbi']:.2f} dBi")
    st.write(f"**Achieved Directivity at {max_scan_deg}°:** {results['achieved_scan_dbi']:.2f} dBi (Meets > {target_min_gain_dbi} dBi target)")
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Array Layout")
        layout_fig = plot_array_layout(results['elements_per_side'], results['spacing_mm'])
        st.pyplot(layout_fig)
        
    with col4:
        st.subheader("Broadside Radiation Pattern")
        pattern_fig = plot_radiation_pattern(results['elements_per_side'], results['spacing_wl'], results['achieved_boresight_dbi'])
        st.pyplot(pattern_fig)
else:
    st.info("Adjust your parameters in the sidebar and click 'Calculate Design' to view the results.")