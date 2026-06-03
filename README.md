# Perinodular Radiomics Analysis

## Overview
This project investigates perinodular radiomics using the Japanese Society of Radiological Technology (JSRT) chest X-ray image database. The study aims to identify potential correlations between perinodular region characteristics and factors that may improve radiologist detection of pulmonary nodules.

## Objective
To analyze the relationship between perinodular brightness patterns and nodule visibility in chest radiographs, potentially improving diagnostic accuracy and detection rates.

## Dataset
- **Source:** JSRT Chest X-ray Database (154 images with annotated nodules)
- **Image Format:** JSRT standard (2048×2048 pixels, 0.175 mm/pixel resolution)
- **Clinical Data:** Nodule location, diameter measurements, and radiologist clarity assessments

## Project Structure
- `testingxray.py` – Main analysis script for nodule brightness calculations
- `circledrawer.py` – Visualization tool for marking nodule regions
- `test_parsing.py` – Clinical metadata parser and validation utility
- `testingcorrlation.py` – Statistical correlation analysis between perinodular brightness and radiologist clarity ratings
- `results/` – Output data and analysis results (not committed to version control)
