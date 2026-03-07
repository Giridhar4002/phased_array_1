---
title: PA Antenna Design — CICAD 2025 Problem 1
emoji: 📡
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.43.2
app_file: app.py
pinned: false
---

# 📡 Phased Array Antenna Design & Analysis Tool

**CICAD 2025 — Assignment Problem 1**

An interactive Streamlit web application that designs a square-grid, square-aperture phased array antenna with microstrip patch elements for wide-scan applications. The tool solves the complete design problem and provides interactive visualizations.

---

## Problem Statement

> Design a phased array antenna meeting the following specifications:
> - **Frequency** = 14.5 GHz ± 1 GHz (13.5 GHz to 15.5 GHz)
> - **Grid:** Square
> - **Aperture shape:** Square
> - **Feed Element:** Microstrip patch antenna (element efficiency = 90%)
> - **Maximum scan angle:** 45°
> - **Minimum gain over scan range:** 20 dBi
> - **Uniform illumination** across the array
>
> **(a)** Calculate the required peak directivity for the array.
> **(b)** Calculate the element spacing and element gain to avoid grating lobes.
> **(c)** Calculate the number of elements, array layout, and plot radiation patterns. What is the total physical aperture size?

---

## Design Methodology

The tool implements the phased array design equations from the following references:

1. **S. K. Rao and C. Ostroot**, *"Design Principles and Guidelines for Phased Array and Reflector Antennas,"* IEEE Antennas & Propagation Magazine, April 2020.
2. **S. Kotta and G. Gupta**, *"Phased Array Antenna Design and Analysis Tool,"* IEEE WAMS, 2023.

### Part A — Peak Directivity

The minimum gain of 20 dBi must be maintained at the worst-case scan angle (45°). The scan loss for small elements (d ≈ 0.5λ) is modelled as:

```
SL = 10·log₁₀(cosⁿ(θ_max))     where n = 1.0–1.5 for patch elements
```

The required peak directivity accounts for scan loss, element efficiency, array efficiency, and any additional system losses:

```
D_peak = G_min / (η_element · η_array · cosⁿ(θ_max))
```

### Part B — Element Spacing & Grating Lobes

For a square lattice, the maximum element spacing to avoid grating lobes at the highest operating frequency is:

```
d ≤ λ_h / (1 + sin(θ_max))
```

where λ_h is the wavelength at the highest frequency (15.5 GHz). This ensures no grating lobes appear in visible space even at maximum scan.

Element directivity uses the effective aperture model:

```
D_e = η_e · 4π·d² / λ_l²
```

where λ_l is the wavelength at the lowest frequency (worst-case for directivity).

### Part C — Number of Elements & Array Sizing

```
N = D_peak / D_element     (linear values)
N_side = ⌈√N⌉             (square array)
N_total = N_side²
Aperture = N_side × d      (each side)
```

### Array Factor Calculation

The normalized array factor for a uniform linear array is:

```
AF(θ) = sin(N·ψ/2) / [N·sin(ψ/2)]
```

where `ψ = k·d·(sinθ − sinθ_scan)` is the progressive phase difference.

---

## Features

| Feature | Description |
|---------|-------------|
| **Interactive Sidebar** | All design specs adjustable via sliders and number inputs |
| **Part A Results** | Peak directivity, peak gain, scan loss displayed as metric cards |
| **Part B Results** | Element spacing (mm, λ), element directivity, grating lobe locations |
| **Part C Results** | Element count, aperture size, achieved performance, spec compliance check |
| **Design Summary Table** | Complete tabulated design parameters |
| **Radiation Patterns** | Cartesian and polar plots for boresight and scanned beams |
| **Array Layout** | 2D visualization of physical element positions with zoom |
| **Parametric Curves** | Directivity vs N, Directivity & GL vs spacing, Efficiency vs taper |
| **Formula Reference** | Expandable section with all equations and references |

---

## App Structure

```
├── app.py              # Main Streamlit application (single file)
├── requirements.txt    # Python dependencies
├── .gitattributes      # Git LFS configuration (for HF Spaces)
└── README.md           # This file
```

### `app.py` — Code Organization

| Section | Lines (approx) | Description |
|---------|----------------|-------------|
| Constants & Config | 1–30 | Speed of light, page setup |
| Sidebar Inputs | 30–110 | All user-configurable parameters |
| Part A Calculations | 110–165 | Scan loss, required peak directivity |
| Part B Calculations | 165–210 | Element spacing, element directivity, grating lobes |
| Part C Calculations | 210–260 | Number of elements, aperture size, achieved performance |
| Results Display | 260–400 | Metric cards, summary table, margin check |
| Visualizations | 400–600 | Radiation patterns, array layout, parametric curves |
| Formulas Section | 600–650 | Expandable LaTeX equations reference |

---

## How to Run Locally

### Prerequisites

- Python 3.9 or later
- pip package manager

### Installation & Launch

```bash
# Clone the repository (or download the files)
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Deploy on Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with SDK = **Streamlit**.
2. Upload `app.py`, `requirements.txt`, `.gitattributes`, and `README.md`.
3. The Space will automatically build and deploy.

---

## Default Design Results (Problem Statement Values)

For the default inputs (14.5 GHz ± 1 GHz, 45° scan, 20 dBi min gain, 90% element efficiency, uniform illumination):

| Parameter | Value |
|-----------|-------|
| Scan Loss (cos¹·⁵ model) | ~4.52 dB |
| Required Peak Directivity | ~25.0 dBi |
| Element Spacing | ~11.3 mm (0.585 λ at 15.5 GHz) |
| Element Directivity | ~5.6 dBi |
| Number of Elements | ~85 (≈ 10×10 = 100) |
| Aperture Size | ~113 × 113 mm |

*Note: Exact values depend on the scan-loss exponent and loss budget settings.*

---

## Key Design Insights

1. **Frequency allocation matters:** The highest frequency controls grating lobes (element spacing), while the lowest frequency gives worst-case directivity.

2. **Scan loss dominates for wide-scan arrays:** At 45°, the cos¹·⁵(θ) model gives ~4.5 dB loss, requiring significantly more peak directivity than the minimum gain spec.

3. **Uniform illumination** maximizes directivity but results in first sidelobes at approximately −13.2 dB (sinc pattern for square aperture).

4. **Square lattice** is chosen for wide-scan applications because it provides more uniform grating-lobe margin across all azimuth planes compared to hexagonal lattice.

5. **Patch elements** (d ≈ 0.5–0.6λ) are preferred for ±45° scan due to their wide element patterns and compatibility with tight spacing requirements.

---

## References

1. S. K. Rao and C. Ostroot, "Design Principles and Guidelines for Phased Array and Reflector Antennas," *IEEE Antennas & Propagation Magazine*, vol. 62, no. 2, pp. 74–81, April 2020.
2. S. Kotta and G. Gupta, "Phased Array Antenna Design and Analysis Tool," *IEEE Wireless Antenna and Microwave Symposium (WAMS)*, 2023.
3. S. Kotta and G. Gupta, "Reflector Antennas Design and Analysis Software," *IEEE WAMS*, 2024.
4. R. J. Mailloux, *Phased Array Antenna Handbook*, Artech House, 1994.

---

## License

This tool is developed as part of the CICAD 2025 internship programme. Educational use only.
