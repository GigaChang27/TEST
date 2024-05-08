import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

# Transformation function
def apply_transformation(e, transformation):
    if transformation == "Log Transformation":
        return np.log(e)
    elif transformation == "Square Transformation":
        return e ** 2
    elif transformation == "Box-cox Transformation":
        # Take the absolute value of residuals first, then perform Box-Cox transformation
        abs_e = np.abs(e)
        _, lmbda = stats.boxcox(abs_e)
        return stats.boxcox(e, lmbda=lmbda)
    else:
        return e  # Return the original value if no transformation is selected

# Residual analysis
def residual_analysis(model_dataset):
    st.header("Residual Analysis")

    if model_dataset is not None:
        model_firstorder = LinearRegression()
        X = model_dataset.iloc[:, 1:]
        Y = model_dataset.iloc[:, 0]
        model_firstorder.fit(X, Y)
        Y_pred = model_firstorder.predict(X)  # Use model_firstorder for prediction
        e = Y - Y_pred

        # Display residual plot
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.scatter(Y_pred, e)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")

        # Display absolute residual plot
        plt.subplot(2, 2, 2)
        plt.scatter(Y_pred, np.abs(e))
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Fitted Values")
        plt.ylabel("Absolute Residuals")
        plt.title("Absolute Residuals Plot")

        # Display QQ plot
        plt.subplot(2, 2, 3)
        probplot(e, plot=plt)
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Ordered Residuals")
        plt.title("QQ Plot")

        # Display residuals time series plot
        plt.subplot(2, 2, 4)
        plt.plot(e)
        plt.xlabel("Observations")
        plt.ylabel("Residuals")
        plt.title("Residuals Time Series")

        plt.subplots_adjust(hspace=0.5, wspace=0.3)              
        st.pyplot(plt)  # Pass the plt object to st.pyplot()

        # Ask if violate assumptions
        violation_option = st.radio("Do you violate assumptions?", ["Yes", "No"])

        if violation_option == "Yes":
            # Ask user to fill in violated assumption
            violation = st.text_input("Please specify the violated assumption (Normality/Independence/Homoscedasticity)", "")

            # Display the filled result
            st.write("Violated Assumption:", violation)

            # Ask for transformation method
            transformation = st.selectbox("Please select a transformation method", ["Log Transformation", "Square Transformation", "Box-cox Transformation"])

            # Display selected transformation method
            st.write("Transformation Method:", transformation)

            # Execute transformation
            transformed_e = apply_transformation(e, transformation)

            # Display transformed residuals plot
            plt.figure()
            plt.scatter(Y_pred, transformed_e)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel("Fitted Values")
            plt.ylabel("Transformed Residuals")
            plt.title("Transformed Residuals Plot")
            st.pyplot(plt)
        else:
            st.write("No violation of assumptions, no need for transformation.")

    else:
        st.error("Please go back to the model fitting page and select a model.")
        if st.button("Back to Model Fitting Page"):
            st.experimental_rerun()

# Main program
def main():
    # Get model dataset from session state
    model_dataset = st.session_state.model_dataset

    # Perform residual analysis
    residual_analysis(model_dataset)

# Run main program
if __name__ == "__main__":
    main()

pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/3_model_fitting.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/5_result.py")