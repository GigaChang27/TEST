import streamlit as st
import pandas as pd



st.header("Welcome to RegressifyXpert")
st.write("""
    We're dedicated to empowering your data-driven decisions through advanced regression analysis. Whether you're a seasoned analyst or just beginning your journey into data science, RegressifyXpert is here to support you every step of the way.
    """)

# Always show the image
# st.image("E:/python_venv/stat/try/project.png", use_column_width=True)




pages = st.container(border=False )
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col5:
        if st.button("next page ▶️"):
            st.switch_page("pages/0_example.py")

