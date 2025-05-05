import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="OFFSHORE TEMPERATURE FORECAST", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #FFFAFA;'>🧰 OFFSHORE TEMPERATURE FORECAST</h1>",
    unsafe_allow_html=True
)

mode = st.sidebar.radio("Choose your desired mode:", [
    "Prediction and Comparison with Given Actual Value",
    "Future Forecasting Only",
    "Compare Predicted with Actual",
    "Visualize Actual vs Predicted",
    "Visualize Accuracy",
    "Compute SE and MAPE for Each Row",
    "Calculate Overall Metrics",
    "Visualize Error Metrics"
])

uploaded_file = st.file_uploader("Upload Excel result file", type=["xlsx"])

def round_df(df):
    return df.round(2)

def plot_styled_chart(df, y1, y2, title, y_label, legend1, legend2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df[y1],
        mode='lines', name=legend1,
        line=dict(color='blue', width=4),
        hovertemplate=f'{legend1}: %{{y:.2f}}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df[y2],
        mode='lines', name=legend2,
        line=dict(color='red', width=4),
        hovertemplate=f'{legend2}: %{{y:.2f}}<extra></extra>'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        font=dict(family="Times New Roman", size=24, color="black"),
        hoverlabel=dict(font=dict(color='red', size=20, family="Times New Roman")),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=60, b=60),
        xaxis=dict(showgrid=True, tickangle=-45, tickfont=dict(size=18, color='black')),
        yaxis=dict(showgrid=True, tickfont=dict(size=18, color='black')),
        legend=dict(font=dict(size=18)),
        shapes=[{
            "type": "rect",
            "xref": "paper", "yref": "paper",
            "x0": 0, "y0": 0, "x1": 1, "y1": 1,
            "line": {"color": "black", "width": 2}
        }]
    )
    return fig

def download_chart(fig, filename):
    img_bytes = fig.to_image(format="png", engine="kaleido", width=1200, height=600)
    st.download_button(
        label=f"📥 Download {filename}.png",
        data=img_bytes,
        file_name=f"{filename}.png",
        mime="image/png"
    )

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = round_df(df)
    st.subheader("📋 Preview of Uploaded Data")
    st.dataframe(df.head())

    if mode == "Visualize Actual vs Predicted":
        for depth in ['Te03m', 'Te30m', 'Te50m']:
            a, p = f'Actual_{depth}', f'Predicted_{depth}'
            if a in df.columns and p in df.columns:
                fig = plot_styled_chart(df, a, p,
                    f"Actual vs Predicted for {depth} Temperature", "Temperature (°C)", a, p)
                st.plotly_chart(fig, use_container_width=True)
                download_chart(fig, f"actual_vs_predicted_{depth}")

    elif mode == "Visualize Accuracy":
        for depth in ['Te03m', 'Te30m', 'Te50m']:
            a, p = f'Actual_{depth}', f'Predicted_{depth}'
            acc_col = f'Accuracy_{depth}'
            if a in df.columns and p in df.columns:
                df[acc_col] = 100 - (np.abs(df[a] - df[p]) / df[a] * 100)
                fig = plot_styled_chart(df, acc_col, acc_col,
                    f"Accuracy for {depth} Temperature", f"Accuracy (%)", acc_col, acc_col)
                st.plotly_chart(fig, use_container_width=True)
                download_chart(fig, f"accuracy_{depth}")

    elif mode == "Visualize Error Metrics":
        for depth in ['Te03m', 'Te30m', 'Te50m']:
            a, p = f'Actual_{depth}', f'Predicted_{depth}'
            se_col, mape_col = f'SE_{depth}', f'MAPE_{depth}'
            if a in df.columns and p in df.columns:
                df[se_col] = (df[a] - df[p]) ** 2
                df[mape_col] = (np.abs(df[a] - df[p]) / df[a]) * 100
                fig1 = plot_styled_chart(df, se_col, se_col,
                    f"Squared Error for {depth}", "Squared Error", se_col, se_col)
                st.plotly_chart(fig1, use_container_width=True)
                download_chart(fig1, f"squared_error_{depth}")
                fig2 = plot_styled_chart(df, mape_col, mape_col,
                    f"MAPE for {depth}", "MAPE (%)", mape_col, mape_col)
                st.plotly_chart(fig2, use_container_width=True)
                download_chart(fig2, f"mape_{depth}")

    elif mode == "Calculate Overall Metrics":
        st.subheader("📈 Overall Error Metrics")
        for depth in ['Te03m', 'Te30m', 'Te50m']:
            a, p = f'Actual_{depth}', f'Predicted_{depth}'
            if a in df.columns and p in df.columns:
                rmse = mean_squared_error(df[a], df[p], squared=False)
                mae = mean_absolute_error(df[a], df[p])
                r2 = r2_score(df[a], df[p])
                st.markdown(f"### {depth}")
                st.markdown(f"- RMSE: **{rmse:.4f}**")
                st.markdown(f"- MAE: **{mae:.4f}**")
                st.markdown(f"- R² Score: **{r2:.4f}**")

    else:
        st.info(f"Mode '{mode}' is under development or unchanged. Please upload data to proceed.")
