import streamlit as st
from datetime import date, datetime

today = datetime.today().strftime("%H:%M:%S")



st.write("hello")

st.title(today)

st.write([1, 2, 3, 4])

st.write({"x":1})

st.selectbox("Choose your model", ("GPT-3", "GPT-4"))
