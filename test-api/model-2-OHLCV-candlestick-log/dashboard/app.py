from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stock Direction AI", layout="wide")
st.title("Stock Direction AI")
st.caption("Educational probability estimates and historical evaluation; not financial advice or live trading.")

path = st.text_input("Predictions CSV", "outputs/daily_fnn/test_predictions.csv")
if not Path(path).exists():
    st.warning("Train a model first or enter the path to a saved test_predictions.csv file.")
    st.stop()

predictions = pd.read_csv(path, parse_dates=["target_date", "feature_end"])
dates = sorted(predictions["target_date"].unique(), reverse=True)
selected_date = st.selectbox("Prediction date", dates)
day = predictions[predictions["target_date"] == selected_date].sort_values("probability_up", ascending=False)

st.subheader("Ranked next-period probabilities")
st.dataframe(
    day[["symbol", "probability_up", "direction", "target_return"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Probability distribution")
st.bar_chart(day.set_index("symbol")["probability_up"])
