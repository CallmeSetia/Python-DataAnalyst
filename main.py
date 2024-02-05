import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, q_stat, adfuller

import matplotlib.pyplot as plt
import seaborn as sns

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

        # Split the data into training and testing sets
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Display regression metrics
        st.write("## Multi Regression Metrics")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

        # Plot the actual vs predicted values
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted Values')
        st.pyplot(fig)

        # Perform Augmented Dickey-Fuller test for stationarity on residuals
        residuals = y_test - y_pred
        st.write("## Augmented Dickey-Fuller Test for Stationarity on Residuals")
        adf_result = adfuller(residuals)
        st.write(f"ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"P-value: {adf_result[1]:.4f}")
        st.write(f"Critical Values: {adf_result[4]}")


        # Perform Ljung-Box test for white noise on residuals
        st.write("## Portmanteau Test for White Noise on Residuals")
        lags = min(10, len(residuals) // 5)  # Select a reasonable number of lags
        autocorrelation = acf(residuals, nlags=lags)
        q_stat_result = q_stat(autocorrelation, len(residuals))
        st.write(f"Portmanteau Q-statistic: {q_stat_result[0][-1]:.4f}")
        st.write(f"P-value: {q_stat_result[1][-1]:.4f}")