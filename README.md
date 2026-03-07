---
title: Phased Array Antenna Design
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
pinned: false
---

# Phased Array Antenna Design Application

This Streamlit application calculates and visualizes the design parameters for a phased array antenna based on specific user requirements. 

## Default Specifications Addressed:
* **Frequency:** 14.5 GHz +/- 1 GHz (Evaluated at the worst-case highest frequency of 15.5 GHz to strictly avoid grating lobes)
* **Grid & Aperture:** Square
* **Feed Element:** Microstrip patch antenna (90% efficiency)
* **Maximum Scan Angle:** 45 degrees
* **Minimum Gain over scan range:** 20 dBi
* **Illumination:** Uniform across the array

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`