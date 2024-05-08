import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import f, t


# Calculate linear regression model
def calculate_linear_regression(df):
    model = LinearRegression()
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]
    model.fit(X, Y)
    return model, X, Y

# Extract regression statistics
def extract_regression_stats(model, X, Y):
    r_squared = model.score(X, Y)
    adj_r_squared = 1 - (1 - r_squared) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)
    std_error = np.sqrt(np.sum((Y - model.predict(X))**2) / (len(Y) - X.shape[1] - 1))

    return r_squared, adj_r_squared, std_error

# Calculate ANOVA related values
def calculate_anova(model, X, Y):
    n = len(Y)
    k = X.shape[1]
    Y_pred = model.predict(X)
    SS_regression = np.sum((Y_pred - np.mean(Y))**2)
    SS_residual = np.sum((Y - Y_pred)**2)
    SS_total = np.sum((Y - np.mean(Y))**2)
    MS_regression = SS_regression / (k - 1)
    MS_residual = SS_residual / (n - k)
    F_value = MS_regression / MS_residual
    significance = 1 - f.cdf(F_value, k - 1, n - k)

    return n, k, SS_regression, SS_residual, SS_total, MS_regression, MS_residual, F_value, significance

# Calculate t-statistics and P-value for model coefficients, as well as prediction intervals
def calculate_coefficient_stats(model, X, Y):
    n = len(Y)
    k = X.shape[1]
    dof_residual = n - k
    t_values = []
    p_values = []
    confidence_intervals = []
    predict_intervals = []  # New list to store prediction intervals

    for i, coef in enumerate(model.coef_):
        std_err = np.sqrt(np.sum((Y - model.predict(X))**2) / dof_residual / np.sum((X.iloc[:, i] - X.iloc[:, i].mean())**2))
        t_value = coef / std_err
        p_value = 2 * (1 - t.cdf(abs(t_value), dof_residual))
        confidence_interval_upper = coef + t.ppf(0.975, dof_residual) * std_err
        confidence_interval_lower = coef - t.ppf(0.975, dof_residual) * std_err

        t_values.append(t_value)
        p_values.append(p_value)
        confidence_intervals.append((confidence_interval_upper, confidence_interval_lower))

        # Calculate prediction intervals
        predict_interval = t.ppf(0.975, dof_residual) * std_err * np.sqrt(1 + np.dot(np.dot(X.iloc[i, :], np.linalg.inv(np.dot(X.T, X))), X.iloc[i, :].T))
        predict_interval_upper = model.predict(X.iloc[i, :].values.reshape(1, -1))[0] + predict_interval
        predict_interval_lower = model.predict(X.iloc[i, :].values.reshape(1, -1))[0] - predict_interval
        predict_intervals.append((predict_interval_upper, predict_interval_lower))

    return t_values, p_values, confidence_intervals, predict_intervals

# Calculate residuals
def calculate_residuals(model, X, Y, num_samples):
    residuals = Y - model.predict(X)
    return residuals[:num_samples]

# Main program
def main():
    st.header("Summary Result")

    # Get model dataset from session state
    model_dataset = st.session_state.model_dataset

    if model_dataset is not None:
        # Calculate linear regression model using the model dataset
        model, X, Y = calculate_linear_regression(model_dataset)

        # Extract regression statistics
        r_squared, adj_r_squared, std_error = extract_regression_stats(model, X, Y)

        # Calculate ANOVA related values
        n, k, SS_regression, SS_residual, SS_total, MS_regression, MS_residual, F_value, significance = calculate_anova(model, X, Y)

        # Calculate t-statistics and P-value for model coefficients, and prediction intervals
        t_values, p_values, confidence_intervals, predict_intervals = calculate_coefficient_stats(model, X, Y)

        # Calculate residuals
        residuals = calculate_residuals(model, X, Y, num_samples=5)

        # Display regression statistics table
        st.write("Regression Statistics:")
        regression_stats = {
            "R-squared": r_squared,
            "Adjusted R-squared": adj_r_squared,
            "Standard Error": std_error,
            "Number of Observations": len(Y)
        }
        st.table(pd.DataFrame.from_dict(regression_stats, orient="index", columns=["Value"]))

        # Display ANOVA related table
        st.write("ANOVA Related:")
        anova_stats = {
            "Degrees of Freedom": [k - 1, n - k, n - 1],
            "SS": [SS_regression, SS_residual, SS_total],
            "MS": [MS_regression, MS_residual, "-"],
            "F Value": [F_value, "-", "-"],
            "Significance": [significance, "-", "-"]
        }
        st.table(pd.DataFrame(anova_stats, index=["Regression", "Residual", "Total"]))

        # Display model coefficient information
        st.write("Coefficient Statistics:")
        coefficient_stats = {
            "Coefficients": np.append(model.intercept_, model.coef_),
            "Standard Error": np.append(np.nan, np.sqrt(np.diag(np.linalg.inv(X.T @ X) * MS_residual))),  # Variance inflation factor
            "t Statistics": np.append(np.nan, t_values),
            "P-value": np.append(np.nan, p_values),
            "Confidence Interval Upper": [np.nan] + [ci[0] for ci in confidence_intervals],
            "Confidence Interval Lower": [np.nan] + [ci[1] for ci in confidence_intervals],
            "Prediction Interval Upper": [np.nan] + [pi[0] for pi in predict_intervals],  # Add prediction interval upper
            "Prediction Interval Lower": [np.nan] + [pi[1] for pi in predict_intervals]   # Add prediction interval lower
        }
        st.table(pd.DataFrame(coefficient_stats, index=["Intercept"] + list(X.columns)))

       # Determine if individual coefficients are significant
        significant_coefficients_status = ["Significant" if p_value < 0.05 else "Not Significant" for p_value in p_values]

        # Get the indices of significant coefficients
        significant_coefficient_indices = np.where(np.array(p_values) < 0.05)[0]

        # Get the names of significant coefficients using indices
        significant_coefficient_names = X.columns[significant_coefficient_indices]

        # Format the result using column names
        significant_coefficients_result = [f"{col} (Significant)" if col in significant_coefficient_names else f"{col} (Not Significant)" for col in X.columns]

        # Model significance and interpretation
        model_significance = "Significant" if significance < 0.05 else "Not Significant"
        model_interpretation = "As the model's R-squared value approaches 1, the model's explanatory power becomes stronger."
        st.write("Conclusion:")
        st.write(f"Model Significance: {model_significance}")
        st.write(f"Significant Coefficients: {', '.join(significant_coefficients_result) if significant_coefficients_result else 'None'}")
        st.write("Model Interpretation:", model_interpretation)
    else:
        st.error("Please go back to the model fitting page and select a model.")
        if st.button("Back to Model Fitting Page"):
            st.experimental_rerun()


if __name__ == "__main__":
    main()
pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/4_residual.py")
