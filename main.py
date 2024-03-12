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
import seaborn as sns
from scipy import stats



##
## Portmentau P Value : last index (V)
##
## Critical value (V)
##
## Plot Residuals UCL LCL plot all data
##
st.set_option('deprecation.showPyplotGlobalUse', False)
def analyze_data_validity(df, feature_columns, target_column):
    st.subheader("Data Validity Analysis")
    col = feature_columns
    col.append(target_column)

    # Deskripsi statistik
    st.write("### Descriptive Statistics")
    st.write(df[col].describe())

    # Distribusi Variabel
    st.write("### Distribution of Variables")
    for column in df[col].columns:
        st.write(f"#### {column}")
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        st.pyplot()

    # # Deteksi Outliers
    # st.write("### Outliers Detection")
    # for column in df[col].columns:
    #     st.write(f"#### {column}")
    #     z_scores = np.abs(stats.zscore(df[column]))
    #     outliers = np.where(z_scores > 3)[0]
    #     st.write(f"Number of outliers in {column}: {len(outliers)}")

    # Korelasi antar variabel
    st.write("### Variable Correlation")
    st.write("Korelasi antar variabel")
    correlation_matrix = df[col].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()

def load_data(uploaded_file, delimiter_option):
    try:
        df = pd.read_csv(uploaded_file, delimiter=delimiter_option)

        df['Discrepancy'] = df.iloc[:, 2] - df.iloc[:, 1]
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

        model = train_linear_regression(X, y)

        if model is not None:
            y_pred = model.predict(add_constant(X))

            # Regression Metrics
            st.write("## Regression Metrics")
            # st.write(f"Mean Squared Error: {mean_squared_error(y, y_pred):.2f}")
            # st.write(f"R-squared: {r2_score(y, y_pred):.2f}")
            st.write(model.summary())
            st.write()

            # Actual vs Predicted Plot
            plot_actual_vs_predicted(y, y_pred)

            # Residuals Analysis
            st.write("### Residuals")
            residuals = y - y_pred
            table_df_residuals = pd.DataFrame({"r": residuals})
            st.table(table_df_residuals.T.style.format('{:7,.20f}'))
            st.write("### Residuals ^ 2")

            table_df_residuals2 = pd.DataFrame({"R^2": residuals ** 2 })
            st.table(table_df_residuals2.T.style.format('{:7,.30f}'))


            # st.write("Y")
            # st.table(y)
            # st.write("Y PRed")
            # st.table(y_pred.T)
            # st.write("Y Test")
            # st.table(y.T)


            # Accuracy and Precision Calculation



            average_residuals = np.mean(residuals)
            rmse = np.sqrt(np.sum(residuals ** 2) / len(residuals))
            total_data = len(y)
            accuracy = 1 - (np.abs(np.sqrt(np.sum(residuals ** 2) / len(residuals)) / np.average(y)))

            ucl = average_residuals + (2 * rmse)
            lcl = average_residuals - (2 * rmse)



            def check_LCL(r, LCL):
                if r < LCL:
                    return 'Yes'
                else:
                    return 'No'

            def check_UCL(r, UCL):
                if r > UCL:
                    return 'Yes'
                else:
                    return 'No'

            df_hasil = df.copy()
            df_hasil['r'] = residuals
            df_hasil['R^2'] = residuals ** 2
            df_hasil['average'] = average_residuals
            df_hasil['ucl'] = ucl
            df_hasil['lcl'] = lcl
            df_hasil['Checklist UCL'] = df_hasil.apply(lambda row: check_UCL(row['r'], row['ucl']), axis=1)
            df_hasil['Checklist LCL'] = df_hasil.apply(lambda row: check_LCL(row['r'], row['lcl']), axis=1)

            count_yes_UCL = df_hasil['Checklist UCL'].value_counts().get('Yes', 0)

            # Menghitung jumlah 'Yes' dalam kolom 'Checklist LCL'
            count_yes_LCL = df_hasil['Checklist LCL'].value_counts().get('Yes', 0)


            data_within_limits = count_yes_UCL + count_yes_LCL

            precision = ((total_data - data_within_limits) / total_data) * 100


            # Display Results in Sidebar
            st.sidebar.title(f"Results")
            st.sidebar.write(f"Total Data: {total_data}")
            st.sidebar.write(f"Average: {average_residuals}")
            st.sidebar.write(f"Root Mean Square Error (RMSE): {rmse}")
            st.sidebar.write(f"Accuracy: {accuracy}")
            st.sidebar.write(f"Accuracy (%): {float(accuracy * 100)}")
            st.sidebar.write(f"Upper Control Limit (UCL): {ucl}")
            st.sidebar.write(f"Lower Control Limit (LCL): {lcl}")
            st.sidebar.write(f"Count YES UCL & LCL: {data_within_limits}")
            st.sidebar.write(f"Precision: {precision:.2f}%")

            # Display Results
            st.title(f"Results")
            # st.table(df_hasil.T.style.format('{:7,.30f}'))
            pd.set_option('display.float_format',
                          lambda x: '%.100f' % x)  # Ubah angka sesuai dengan jumlah digit presisi yang diinginkan

            st.table(df_hasil.style.format({'r': '{:.15f}', 'R^2': '{:.30f}', 'average': '{:.16f}', 'ucl': '{:.16f}', 'lcl': '{:.16f}'}))

            st.write(f"Total Data: {total_data}")
            st.write(f"Average: {average_residuals}")
            st.write(f"Root Mean Square Error (RMSE): {rmse}")
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Accuracy (%): {float(accuracy * 100)}")
            st.write(f"Upper Control Limit (UCL): {ucl}")
            st.write(f"Lower Control Limit (LCL): {lcl}")
            st.write(f"Count YES UCL & LCL: {data_within_limits}")
            st.write(f"Precision: {precision:.2f}%")

            # Augmented Dickey-Fuller Test for Stationarity
            st.write("## Augmented Dickey-Fuller Test for Stationarity on Residuals")
            adf_result = adfuller(residuals, autolag='BIC',regression='c', maxlag=0)
            table_df_adf_result = pd.DataFrame({
                "Statistic": ["Test Statistic", "P-value", "Number of Lags", "Number of Observations",
                               "Critical Values", "Z(t)"],
                "Value": adf_result
            })
            st.write(adf_result)
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
            from scipy.stats import chi2
            def portmanteau_test(residuals, lag=10, significance_level=0.05):
                """
                Menghitung statistik uji Q untuk white noise pada residual.

                Parameters:
                residuals : array_like
                    Array berisi nilai residual.
                lag : int, optional
                    Jumlah lag yang ingin digunakan dalam perhitungan.
                significance_level : float, optional
                    Tingkat signifikansi yang diinginkan untuk tes.

                Returns:
                Q_statistic : float
                    Nilai statistik uji Q.
                p_value : float
                    Nilai p untuk tes.
                """
                n = len(residuals)
                if lag is None:
                    # Jika lag tidak ditentukan, gunakan jumlah lag yang direkomendasikan oleh Box-Pierce
                    lag = int(np.log(n) ** 2)

                # Hitung korelasi sampel untuk setiap lag
                autocorrelation = np.correlate(residuals, residuals, mode='full')
                autocorrelation /= np.max(autocorrelation)  # Normalisasi

                # Hitung nilai statistik uji Q
                Q_statistic = n * (n + 2) * np.sum(
                    (autocorrelation[n - 1:n + lag - 1]) ** 2 / np.arange(n, n - lag, -1))

                # Hitung nilai p terkait
                p_value = 1 - chi2.cdf(Q_statistic, lag)

                return Q_statistic, p_value

            # Q_statistic, p_value = portmanteau_test(residuals, lag=40, significance_level=0.05)
            # st.write(portmanteau_test(residuals, lag=1))


            lags = min(40, (len(residuals) /2) - 2)  # Select a reasonable number of lags
            autocorrelation = acf(residuals, nlags=lags)
            q_stat_result = q_stat(autocorrelation, len(residuals))
            # st.write( len(residuals))
            # Autocorrelation Function (ACF) Values
            acf_values, q_statistic, p_values = acf(residuals, nlags=(len(residuals) // 2) + 40, qstat=True)

            num_lags_acf = len(acf_values)
            acf_values = acf_values[:num_lags_acf]
            q_statistic = q_statistic[:num_lags_acf]
            p_values = p_values[:num_lags_acf]

            # st.write(q_stat_result)
            st.write(f"Portmanteau Q-statistic: {q_statistic[-1]:.4f}")
            st.write(f"P-value: {p_values[-1]:.4f}")

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
            if (p_values[-1] < 0.005 ):
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
                analyze_data_validity(df, feature_columns, target_column )
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.error('Error : Pilih X dan Y')