import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.tools import add_constant
from statsmodels.regression import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)
if __name__ == "__main__" :

    st.header("")

    # Sidebar for file upload
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    delimiter_option = st.sidebar.selectbox("Select Delimiter", [',', ';'], index=0)

    if uploaded_file is not None:
        # Read CSV file into DataFrame
        df = pd.read_csv(uploaded_file, delimiter=delimiter_option)
        # Add a new column "Discrepancy" based on the difference between columns 1 and 2
        df['Discrepancy'] = df.iloc[:, 1] - df.iloc[:, 2]

        st.write("## Input Data")
        st.write(df)

        # Choose independent variables (features) and dependent variable (target)
        feature_columns = st.sidebar.multiselect("Select X", df.columns)
        target_column = st.sidebar.selectbox("Select Y", df.columns)

        if target_column is not None and feature_columns is not None:
            # Split the data into training and testing sets
            # Split the data into training and testing sets
            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the linear regression model
            X_train_with_constant = add_constant(X_train)
            X_test_with_constant = add_constant(X_test)
            model = linear_model.OLS(y_train, X_train_with_constant).fit()

            # Make predictions on the test set
            y_pred = model.predict(X_test_with_constant)
            # Display regression metrics
            st.write("## Regression Metrics")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")
            st.write(model.summary())
            st.write()

            # Display VIF (Variance Inflation Factor) for each variable
            st.write("### Variance Inflation Factor (VIF)")
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            st.write(vif_data)

            # Plot the actual vs predicted values
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)

            # Perform Augmented Dickey-Fuller test for stationarity on residuals


            st.write("### Residuals")
            residuals = y_test - y_pred
            # Create a DataFrame for Residuals
            residuals_data = {
                "Residuals": residuals
            }
            table_df_residuals = pd.DataFrame(residuals_data)
            st.table(table_df_residuals.T)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            # Calculate Accuracy
            total_data = len(y_test)
            accuracy = 1 - (rmse / total_data)


            # Menampilkan hasil
            # Calculate UCL and LCL
            average_residual = residuals.mean()
            ucl = average_residual + (2 * rmse)
            lcl = average_residual - (2 * rmse)
            # Hitung jumlah data yang berada di antara UCL dan LCL
            data_within_limits = residuals[(residuals >= lcl) & (residuals <= ucl)].shape[0]

            # Hitung presisi
            precision = (data_within_limits / len(residuals)) * 100

            # Tampilkan presisi

            st.sidebar.title(f"Hasil")
            st.sidebar.write(f"Root Mean Square Error (RMSE): {rmse}")
            st.sidebar.write(f"Accuracy: {accuracy}")
            st.sidebar.write(f"Accuracy (%): {float(accuracy * 100)}")
            st.sidebar.write(f"Upper Control Limit (UCL): {ucl}")
            st.sidebar.write(f"Lower Control Limit (LCL): {lcl}")
            st.sidebar.write(f"Precision: {precision:.2f}%")

            st.title(f"Hasil")
            st.write(f"Root Mean Square Error (RMSE): {rmse}")
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Accuracy (%): {float(accuracy * 100)}")
            st.write(f"Upper Control Limit (UCL): {ucl}")
            st.write(f"Lower Control Limit (LCL): {lcl}")
            st.write(f"Precision: {precision:.2f}%")

            adf_result = adfuller(residuals)
            residu = pd.Series(residuals)
            # Plot Residuals with UCL and LCL
            plt.figure(figsize=(10, 6))
            plt.plot(residu.values, label="Residuals")
            plt.axhline(y=ucl, color='r', linestyle='--', label="Upper Control Limit (UCL)")
            plt.axhline(y=lcl, color='g', linestyle='--', label="Lower Control Limit (LCL)")
            plt.xlabel("Index")
            plt.ylabel("Residuals")
            plt.title("Residuals with UCL and LCL")
            plt.legend()
            st.pyplot()

            st.write("## Augmented Dickey-Fuller Test for Stationarity on Residuals")
            # Extracting details from the ADF result
            test_statistic, p_value, num_lags, num_obs, critical_values, icbest = adf_result

            # Create a table to display the detailed analysis
            table_data = {
                "Statistic": ["Test Statistic", "P-value", "Number of Lags", "Number of Observations",
                              "Critical Values",
                              "Z(t)"],
                "Value": [test_statistic, p_value, num_lags, num_obs, critical_values, icbest]
            }
            table_df = pd.DataFrame(table_data)
            # Create a DataFrame for Critical Values
            critical_values_data = {
                "Critical Values": critical_values
            }
            table_df_critical_val = pd.DataFrame(critical_values_data)

            st.table(table_df)
            # Display the table for Critical Values
            st.write("### Critical Values")
            st.table(table_df_critical_val.T)

            # Interpretation of the test result
            if p_value < 0.05:
                st.write("The residuals exhibit stationarity, suggesting no random walk behavior.")
            else:
                st.write("The residuals do not exhibit stationarity, indicating potential random walk behavior.")

            # Perform Portmanteau test for white noise on residuals
            st.write("## Portmanteau Test for White Noise on Residuals")
            lags = min(10, len(residuals) // 5)  # Select a reasonable number of lags
            autocorrelation = acf(residuals, nlags=lags)
            q_stat_result = q_stat(autocorrelation, len(residuals))
            st.write(f"Portmanteau Q-statistic: {q_stat_result[0][-1]:.4f}")
            st.write(f"P-value: {q_stat_result[1][-1]:.4f}")

            acf_values, q_statistic, p_values = acf(residuals, nlags=min(40, len(residuals) - 1), qstat=True)
            num_lags_acf = len(acf_values)

            # Set the length of acf_values, q_statistic, and p_values
            acf_values = acf_values[:num_lags_acf]
            q_statistic = q_statistic[:num_lags_acf]
            p_values = p_values[:num_lags_acf]

            # Create a DataFrame for the Portmanteau Test results
            portmanteau_data = {
                # "Lag": list(range(1, num_lags_acf)),
                # "ACF Value": acf_values,
                "Q Statistic": q_statistic,
                "P-value": p_values
            }
            portmanteau_acf_data = {
                # "Lag": list(range(1, num_lags_acf)),
                "ACF Value": acf_values,
                # "Q Statistic": q_statistic,
                # "P-value": p_values
            }

            table_df_portmanteau = pd.DataFrame(portmanteau_data)
            table_df_portmanteau_acf = pd.DataFrame(portmanteau_acf_data)

            st.table(table_df_portmanteau.T)
            st.table(table_df_portmanteau_acf.T)
            # Interpretation of the Ljung-Box Test result
            if any(p < 0.05 for p in p_values):
                st.write("The residuals exhibit serial correlation, suggesting non-white noise behavior.")
            else:
                st.write("The residuals do not exhibit serial correlation, indicating white noise behavior.")