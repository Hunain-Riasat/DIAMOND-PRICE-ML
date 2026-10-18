"""
Diamond Price Predictor - Streamlit web app
--------------------------------------------
Web front end for Assignment 3 (AIC354 Machine Learning Fundamentals).

It loads diamond_price_model.pkl, the pipeline saved by the notebook
(imputer + scaler + one-hot encoder + Hist Gradient Boosting), so the
raw diamond properties can be passed straight to the model.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")

MODEL_PATH = Path(__file__).parent / "diamond_price_model.pkl"

# Ordered from lowest to highest quality (this is only the order shown in the dropdowns)
CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLORS = ["J", "I", "H", "G", "F", "E", "D"]
CLARITIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

NUMERIC = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAL = ["cut", "color", "clarity"]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("💎 Diamond Price Predictor")
st.markdown(
    """
    Enter the properties of a diamond and the trained model will estimate its price in US dollars.

    The model is a **Hist Gradient Boosting** regressor, chosen from 5 algorithms using
    **10-fold cross validation** on 53,794 diamonds. Its typical error (cross-validation RMSE)
    is about **\\$551**.
    """
)

st.divider()

st.subheader("Size and weight")
col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat (weight)", min_value=0.2, max_value=5.1, value=1.00, step=0.01, format="%.2f")
    depth = st.number_input("Depth % (total depth)", min_value=43.0, max_value=79.0, value=61.8, step=0.1, format="%.1f")
    table = st.number_input("Table % (width of top)", min_value=43.0, max_value=95.0, value=57.0, step=0.1, format="%.1f")
with col2:
    x = st.number_input("Length x (mm)", min_value=3.0, max_value=11.0, value=6.40, step=0.01, format="%.2f")
    y = st.number_input("Width y (mm)", min_value=3.0, max_value=59.0, value=6.43, step=0.01, format="%.2f")
    z = st.number_input("Depth z (mm)", min_value=1.0, max_value=32.0, value=3.97, step=0.01, format="%.2f")

st.subheader("Quality")
col3, col4, col5 = st.columns(3)
with col3:
    cut = st.selectbox("Cut", CUTS, index=CUTS.index("Ideal"))
with col4:
    color = st.selectbox("Colour (D = best, J = worst)", COLORS, index=COLORS.index("G"))
with col5:
    clarity = st.selectbox("Clarity (IF = best)", CLARITIES, index=CLARITIES.index("VS2"))

if st.button("Predict Price", type="primary", use_container_width=True):
    diamond = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table, "x": x, "y": y, "z": z,
        "cut": cut, "color": color, "clarity": clarity,
    }])[NUMERIC + CATEGORICAL]

    predicted_price = float(model.predict(diamond)[0])
    predicted_price = max(predicted_price, 0.0)

    st.divider()
    st.success(f"### Estimated price: \\${predicted_price:,.0f}")
    st.caption(
        f"Typical error of this model is about \\$551, so the real price would usually fall "
        f"between roughly \\${max(predicted_price - 551, 0):,.0f} and \\${predicted_price + 551:,.0f}."
    )

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(diamond, hide_index=True)

st.divider()
st.caption(
    "Assignment 3 - Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus. "
    "Trained on the public 'diamonds' dataset; estimates reflect that data, not current market prices."
)
