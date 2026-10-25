import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Skincare Product Analytics", layout="wide")
st.title("Skincare Product Analytics")
st.caption("Explore Sephora skincare product prices and ratings.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "primary_category" in df.columns:
        df = df[df["primary_category"] == "Skincare"].copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    st.subheader("Data Preview")
    st.write(f"Number of products: {len(df):,}")
    st.dataframe(df.head(10), hide_index=True, width="stretch")

    st.subheader("Explore the Data")
    variable = st.selectbox("Column to plot", numeric_cols)
    chart_type = st.selectbox(
        "Chart type",
        ["Line chart", "Scatter plot", "Histogram"]
    )

    if chart_type == "Histogram":
        fig = px.histogram(df, x=variable, title=f"Distribution of {variable}")
    elif chart_type == "Line chart":
        fig = px.line(df, y=variable, title=f"{variable} across products")
    else:
        x_options = [col for col in numeric_cols if col != variable]
        x_col = st.selectbox("Choose the x-axis", x_options)
        fig = px.scatter(df, x=x_col, y=variable, title=f"{variable} versus {x_col}")

    st.plotly_chart(fig, width="stretch")

    with st.expander("See full dataset"):
        st.dataframe(df, hide_index=True, width="stretch")

    st.subheader("Linear Regression Model")

    feature_options = numeric_cols.copy()
    x1_col = st.selectbox(
        "Input 1",
        feature_options,
        index=feature_options.index("price_usd")
        if "price_usd" in feature_options else 0
    )

    x2_options = [col for col in feature_options if col != x1_col]
    x2_col = st.selectbox(
        "Input 2",
        x2_options,
        index=x2_options.index("loves_count")
        if "loves_count" in x2_options else 0
    )

    y_options = [col for col in feature_options if col not in [x1_col, x2_col]]
    default_target = "rating" if "rating" in y_options else y_options[0]
    y_col = st.selectbox(
        "Target to predict",
        y_options,
        index=y_options.index(default_target)
    )

    data = df[[x1_col, x2_col, y_col]].dropna()

    if len(data) < 5:
        st.error("Need at least 5 rows with values in all selected columns.")
        st.stop()

    x = data[[x1_col, x2_col]]
    y = data[y_col]

    model = LinearRegression()
    model.fit(x, y)

    st.metric("R² on this data", f"{model.score(x, y):.3f}")
    st.write(f"Rows used for training: {len(data):,}")
  
st.subheader("Try a Prediction")

input1 = st.number_input(
    f"Enter {x1_col}",
    value=float(data[x1_col].mean())
)

input2 = st.number_input(
    f"Enter {x2_col}",
    value=float(data[x2_col].mean())
)

new_data = pd.DataFrame({
    x1_col: [input1],
    x2_col: [input2]
})

prediction = model.predict(new_data)[0]

st.metric(f"Predicted {y_col}", f"{prediction:.2f}")

outside_range = (
    input1 < data[x1_col].min()
    or input1 > data[x1_col].max()
    or input2 < data[x2_col].min()
    or input2 > data[x2_col].max()
)

if outside_range:
    st.warning(
        "One or more inputs are outside the range of the training data. "
        "Treat this prediction with caution."
    )

st.caption(
    "This model estimates a numeric product outcome from two selected variables. "
    "Predictions may be inaccurate because other factors may affect the outcome, "
    "and linear regression assumes a linear relationship."
)
else:
    st.info("Upload the Sephora product CSV file to get started.")
