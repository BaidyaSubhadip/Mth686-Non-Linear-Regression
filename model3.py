import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot
from scipy.linalg import inv

data = {
    "x": [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36,
        0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54,
        0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72,
        0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90,
        0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00
    ],
    "y": [
        6.7464, 6.8873, 6.8963, 6.9970, 6.9236, 7.0945, 7.0462, 7.1714, 7.2678, 7.2464, 7.3870, 7.4598, 7.4832, 7.5905,
        7.6564, 7.7677, 7.8405, 7.7970, 7.8770, 7.9905, 8.0313, 8.0453, 8.1306, 8.2005, 8.4381, 8.3738, 8.4746, 8.4907,
        8.6134, 8.6825, 8.8369, 8.8204, 8.9991, 9.0070, 9.1691, 9.1154, 9.2680, 9.2978, 9.3789, 9.5074, 9.6803, 9.6629,
        9.7271, 9.8358, 9.9258, 10.0706, 10.1321, 10.2287, 10.3611, 10.3989, 10.5320, 10.7185, 10.6583, 10.8535, 10.9607,
        10.9682, 11.0898, 11.3732, 11.4509, 11.5536, 11.6829, 11.6107, 11.7849, 11.9809, 12.0438, 12.2362, 12.2693,
        12.4132, 12.6209, 12.6355, 12.7991, 12.8955, 13.1936, 13.2388, 13.3425, 13.4170, 13.6859, 13.7734, 13.9247,
        13.9711, 14.1495, 14.3493, 14.4159, 14.7500, 14.7771, 14.8974, 15.0384, 15.2206, 15.5036, 15.6202, 15.7084,
        15.8786, 16.1730, 16.2470, 16.4231, 16.6360, 16.8172, 16.9160, 17.1021, 17.2713
    ]
}
df = pd.DataFrame(data)
x = df['x'].values
y = df['y'].values

# Defining the model 3
def model(params, x):
    b0, b1, b2, b3, b4 = params
    return b0 + b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4

# Residue calculation
def residue(y, x, params):
    return y - model(params, x)

# Jacobian calculation
def jacobian(x, params):
    b0, b1, b2, b3, b4 = params
    J = np.zeros((len(x), 5))
    J[:, 0] = 1
    J[:, 1] = x
    J[:, 2] = x**2
    J[:, 3] = x**3
    J[:, 4] = x**4
    return J

# Gauss-Newton method
def gauss_newton(y, x, initial_params, max_iter=1000, lamda_reg=1e-6, tol=1e-8):
    params = np.array(initial_params, dtype=np.float64)
    for i in range(max_iter):
        r = residue(y, x, params) # Calculating the residue for the current parameters
        J = jacobian(x, params)  # Calculating the Jacobian for the current parameters
        JTJ = J.T @ J + lamda_reg * np.eye(J.shape[1]) # Regularization to keep JTJ non-singular
        delta = np.linalg.inv(JTJ) @ J.T @ r
        temp_params = params + delta
        new_residue = residue(y, x, temp_params)
        
        earlier_val = np.sum(r**2)
        new_val = np.sum(new_residue**2)
        
        # Update if error improves
        if new_val < earlier_val:
            params = temp_params
        else:
            params += 0.05 * delta  # Step-factor modification

        if np.linalg.norm(delta) < tol:
            break

    return params

# Grid Search
best_params = None
best_error = float('inf')
for b1 in np.linspace(5, 6, 10):
    for b2 in np.linspace(1, 2, 10):
        for b3 in np.linspace(-5, 5, 10):
            b0 = 6.7
            b4 = 17.2713 - b0 - b1 - b2 - b3
            params = [b0, b1, b2, b3, b4]
            error = np.sum(residue(y, x, params)**2)
            if error < best_error:
                best_error = error
                best_params = params

print("Best initial guess for model 3:", best_params)

initial_guess = best_params
LSE = gauss_newton(y, x, initial_guess)
print("The Least Squares Estimator for model 3:", LSE)

final_residue = residue(y,x,LSE)
error = np.sum(final_residue**2)
print("The error for the model 3:")
print(error)

variance_ans = np.var(final_residue) #Caluclating variance for the epsilon values
print("The caluclated variance according to the model 3:")
print(variance_ans)

y_pred = model(LSE, x)


# Plotting the residue
plt.plot(x, final_residue, marker='o', linestyle='-', color='g')
plt.title("Model 3: Residue Plot")
plt.xlabel("x")
plt.ylabel("Residue (y - y_pred)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the data and the model curve
plt.plot(x, y, 'bo', label="Original Data")
plt.plot(x, y_pred, color = 'orange', label="Fitted Curve (Model 3)", linewidth = 2.5)  
plt.title("Model 3: Data and Fitted Model Curve")
plt.xlabel("x")  
plt.ylabel("y")  
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Shapiro-Wilk Test for normality
shapiro_test = shapiro(final_residue)
print("Shapiro-Wilk Test:")
print(f"Statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}")

if shapiro_test.pvalue > 0.05:
    print("The normality assuption is satisfied.")
else:
    print("The normality assumption is not satisfied.")



J = jacobian(x, LSE)  # Jacobian matrix at the LSE parameters

# Step 2: Calculate Fisher Information Matrix as J^T @ J assuming the normality assumption
Fisher_info_matrix = J.T @ J
if np.linalg.det(Fisher_info_matrix) < 1e-10:
    print("Warning: Fisher information matrix is near singular. Applying regularization.")
    Fisher_info_matrix += np.eye(Fisher_info_matrix.shape[0]) * 1e-8
# Step 3: Covariance Matrix of the parameter estimates
cov_matrix = variance_ans * inv(Fisher_info_matrix)

# Step 4: Calculating standard errors from the diagonal of the covariance matrix
standard_errors = np.sqrt(np.diag(cov_matrix))

# Step 5: Calculate confidence intervals for each parameter
confidence_intervals = []
z_score = 1.96  # For a 95% confidence interval

for i, param in enumerate(LSE):
    lower_bound = param - z_score * standard_errors[i]
    upper_bound = param + z_score * standard_errors[i]
    confidence_intervals.append((lower_bound, upper_bound))

# Display the confidence intervals
print("Confidence Intervals for Model 3 Parameters:")
for i, (param, ci) in enumerate(zip(LSE, confidence_intervals)):
    print(f"Parameter {i+1}: Estimate = {param:.4f}, 95% CI = ({ci[0]:.4f}, {ci[1]:.4f})")
