import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, adfuller, q_stat
from statsmodels.tools import add_constant
from statsmodels.regression import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data(uploaded_file, delimiter_option):
    try:
        df = pd.read_csv(uploaded_file, delimiter=delimiter_option)
        df['Discrepancy'] = df.iloc[:, 1] - df.iloc[:, 2]
        return df
    except pd.errors.EmptyDataError:
        st.error("File is empty.")
    except pd.errors.ParserError:
        st.error("Invalid CSV format. Please check the delimiter.")
    except Exception as e:
        st.error(f"An error occurred: GANTI DELIMITER CSV")
    return None

def train_linear_regression(X_train, y_train):
    try:
        X_train_with_constant = add_constant(X_train)
        model = linear_model.OLS(y_train, X_train_with_constant).fit()
        return model
    except Exception as e:
        st.error(f"Error training linear regression model: {e}")
        return None

def calculate_vif(X):
    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    except Exception as e:
        st.error(f"Error calculating VIF: {e}")
        return None

def plot_actual_vs_predicted(y_test, y_pred):
    try:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted Values')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting actual vs predicted values: {e}")

def plot_residuals(residuals, ucl, lcl):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(residuals.values, label="Residuals")
        plt.axhline(y=ucl, color='r', linestyle='--', label="Upper Control Limit (UCL)")
        plt.axhline(y=lcl, color='g', linestyle='--', label="Lower Control Limit (LCL)")
        plt.xlabel("Index")
        plt.ylabel("Residuals")
        plt.title("Residuals with UCL and LCL")
        plt.legend()
        st.pyplot()
    except Exception as e:
        st.error(f"Error plotting residuals: {e}")

def run_analysis(df, feature_columns, target_column):
    try:
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = train_linear_regression(X_train, y_train)

        if model is not None:
            y_pred = model.predict(add_constant(X_test))

            # Regression Metrics
            st.write("## Regression Metrics")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")
            st.write(model.summary())
            st.write()

            # VIF (Variance Inflation Factor)
            st.write("### Variance Inflation Factor (VIF)")
            vif_data = calculate_vif(X)
            st.write(vif_data)

            # Actual vs Predicted Plot
            plot_actual_vs_predicted(y_test, y_pred)

            # Residuals Analysis
            st.write("### Residuals")
            residuals = y_test - y_pred
            table_df_residuals = pd.DataFrame({"Residuals": residuals})
            st.table(table_df_residuals.T)

            # Accuracy and Precision Calculation
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            total_data = len(y_test)
            accuracy = 1 - (rmse / total_data)
            average_residual = residuals.mean()
            ucl = average_residual + (2 * rmse)
            lcl = average_residual - (2 * rmse)
            data_within_limits = residuals[(residuals >= lcl) & (residuals <= ucl)].shape[0]
            precision = (data_within_limits / len(residuals)) * 100

            # Display Results in Sidebar
            st.sidebar.title(f"Results")
            st.sidebar.write(f"Root Mean Square Error (RMSE): {rmse}")
            st.sidebar.write(f"Accuracy: {accuracy}")
            st.sidebar.write(f"Accuracy (%): {float(accuracy * 100)}")
            st.sidebar.write(f"Upper Control Limit (UCL): {ucl}")
            st.sidebar.write(f"Lower Control Limit (LCL): {lcl}")
            st.sidebar.write(f"Precision: {precision:.2f}%")

            # Display Results
            st.title(f"Results")
            st.write(f"Root Mean Square Error (RMSE): {rmse}")
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Accuracy (%): {float(accuracy * 100)}")
            st.write(f"Upper Control Limit (UCL): {ucl}")
            st.write(f"Lower Control Limit (LCL): {lcl}")
            st.write(f"Precision: {precision:.2f}%")

            # Augmented Dickey-Fuller Test for Stationarity
            st.write("## Augmented Dickey-Fuller Test for Stationarity on Residuals")
            adf_result = adfuller(residuals)
            table_df_adf_result = pd.DataFrame({
                "Statistic": ["Test Statistic", "P-value", "Number of Lags", "Number of Observations",
                               "Critical Values", "Z(t)"],
                "Value": adf_result
            })
            table_df_critical_val = pd.DataFrame({"Critical Values": adf_result[4]})
            st.table(table_df_adf_result)
            st.write("### Critical Values")
            st.table(table_df_critical_val.T)

            if adf_result[1] < 0.05:
                st.write("The residuals exhibit stationarity, suggesting no random walk behavior.")
            else:
                st.write("The residuals do not exhibit stationarity, indicating potential random walk behavior.")

            # Portmanteau Test for White Noise
            st.write("## Portmanteau Test for White Noise on Residuals")
            lags = min(10, len(residuals) // 5)  # Select a reasonable number of lags
            autocorrelation = acf(residuals, nlags=lags)
            q_stat_result = q_stat(autocorrelation, len(residuals))
            st.write(f"Portmanteau Q-statistic: {q_stat_result[0][-1]:.4f}")
            st.write(f"P-value: {q_stat_result[1][-1]:.4f}")

            # Autocorrelation Function (ACF) Values
            acf_values, q_statistic, p_values = acf(residuals, nlags=min(40, len(residuals) - 1), qstat=True)
            num_lags_acf = len(acf_values)
            acf_values = acf_values[:num_lags_acf]
            q_statistic = q_statistic[:num_lags_acf]
            p_values = p_values[:num_lags_acf]

            # Create DataFrames for the Portmanteau Test results
            portmanteau_data = {
                "Q Statistic": q_statistic,
                "P-value": p_values
            }
            acf_data = {
                "ACF Value": acf_values
            }
            table_df_portmanteau = pd.DataFrame(portmanteau_data)
            table_df_acf = pd.DataFrame(acf_data)

            st.table(table_df_portmanteau.T)
            st.table(table_df_acf.T)

            # Interpretation of the Ljung-Box Test result
            if any(p < 0.05 for p in p_values):
                st.write("The residuals exhibit serial correlation, suggesting non-white noise behavior.")
            else:
                st.write("The residuals do not exhibit serial correlation, indicating white noise behavior.")

            # Plot Residuals with UCL and LCL
            plot_residuals(residuals, ucl, lcl)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    st.header("Multiple Linear Regression and Time Series Analysis App")

    # Sidebar for file upload and delimiter selection
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    delimiter_option = st.sidebar.selectbox("Select Delimiter", [',', ';'], index=0)

    if uploaded_file is not None:
        # Load and display the input data
        df = load_data(uploaded_file, delimiter_option)
        if df is not None:
            st.write("## Input Data")
            st.write(df)

            # Choose independent variables (features) and dependent variable (target)
            feature_columns = st.sidebar.multiselect("Select X", df.columns)
            target_column = st.sidebar.selectbox("Select Y", df.columns)

            try:
                run_analysis(df, feature_columns, target_column)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.error('Error : Pilih X dan Y')