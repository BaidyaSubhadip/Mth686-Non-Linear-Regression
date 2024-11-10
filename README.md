# MTH 686 - Non-Linear Regression Project

This repository contains the project work for the course **MTH 686: Non-Linear Regression**.

## Project Overview

In this project, a dataset consisting of paired observations \((t, y(t))\) is analyzed using three proposed non-linear regression models. The primary objective is to determine which model best represents the observed data. To achieve this, unknown parameters for each model are estimated using the **Least Squares Estimation (LSE)** method.

## Methodology

1. **Model Fitting**:  
   - Three non-linear models are proposed for analysis.
   - Each model is fitted to the dataset using the Least Squares method, minimizing the sum of squared errors between observed and predicted values.

2. **Parameter Estimation**:  
   - The model parameters are estimated based on minimizing residuals, yielding the best fit of each model to the data.

3. **Statistical Analysis**:  
   - **Hypothesis Testing**: Conducted to evaluate the statistical significance of the model parameters.
   - **Confidence Interval Estimation**: Constructed for the model parameters to gauge the reliability of the estimates.

## Requirements

- Python packages: `numpy`, `scipy`, `matplotlib`

## Results

The model that best represents the data is determined based on goodness-of-fit metrics. Detailed statistical results and graphical representations of model fits are provided in the `results/` directory.
