import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
import seaborn as sns

if 'df_dropNA' in st.session_state:
    df = st.session_state.df_dropNA
else:
    if 'df_changeNA' in st.session_state:
        df = st.session_state.df_changeNA
    else:
        if 'df_deleted' in st.session_state:
            df = st.session_state.df_deleted
        else:
            if 'df_raw' in st.session_state:
                df = st.session_state.df_raw
            else:
                df = None


st.header("Data Visualizing")

if df is not None:
    # Create a copy of the original DataFrame to preserve the original data
    df_copy = df.copy()

    # Show data types of each variable
    st.subheader("Data Types of Variables:")
    data_types = df_copy.dtypes.to_frame().transpose()  # Transpose the DataFrame
    st.dataframe(data_types)

    if 'categorical_vars' not in st.session_state:
        st.session_state.categorical_vars = data_types.columns[data_types.loc[0] == 'object'].tolist()

    if 'numerical_vars' not in st.session_state:
        st.session_state.numerical_vars = data_types.columns[(data_types.loc[0] == 'int64') | (data_types.loc[0] == 'float64')].tolist()

    # Convert categorical variables to numerical variables for visualization
    categorical_vars_to_convert = st.multiselect("Select categorical variables in dataset:", options=df.select_dtypes(include=['object','int','bool']).columns.tolist())
    if st.button("Convert Categorical Variables"):
        data_types_trans =''
        for var in categorical_vars_to_convert:
            df_copy[var] = df_copy[var].astype(str)

        data_types_trans = df_copy.dtypes.to_frame().transpose()
        st.write("Categorical variables converted to numerical variables for visualization successfully!")
        st.dataframe(data_types_trans)

        categorical_vars = data_types_trans.columns[data_types_trans.loc[0] == 'object'].tolist()
        numerical_vars = data_types_trans.columns[(data_types_trans.loc[0] == 'int') | (data_types_trans.loc[0] == 'float')].tolist()

        st.session_state.categorical_vars = categorical_vars
        st.session_state.numerical_vars = numerical_vars

        

    # Create two columns for visualization
    col1, col2 = st.columns(2)

    # Visualize categorical variables
    with col1:
        st.subheader("Categorical Variables:")
        categorical_var_plot = st.selectbox("Select categorical variable:", options=st.session_state.categorical_vars, key="categorical_var_selectbox")
        with st.expander(f"Show Pie Chart of {categorical_var_plot}"):
            categorical_var_counts = df_copy[categorical_var_plot].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))  # Create figure and axis objects
            ax.pie(categorical_var_counts, labels=categorical_var_counts.index, autopct='%1.1f%%')
            ax.set_title(f"Pie Chart of {categorical_var_plot}")
            st.pyplot(fig)  # Pass the figure object to st.pyplot()

    # Visualize numerical variables
    with col2:
        st.subheader("Numerical Variables:")
        numerical_var_plot= st.selectbox("Select numerical variable:", options=st.session_state.numerical_vars, key="numerical_var_selectbox")
        with st.expander(f"Show Histogram of {numerical_var_plot}"):
            fig, ax = plt.subplots(figsize=(6, 6))  # Create figure and axis objects
            ax.hist(df_copy[numerical_var_plot], bins=20)
            ax.set_title(f"Histogram of {numerical_var_plot}")
            ax.set_xlabel(numerical_var_plot)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # Create a form for Scatter Plot between Two Numerical Variables
    st.subheader("Scatter Plot between Two Numerical Variables:")
    with st.form(key='scatter_plot_form'):
        col1, col2 = st.columns(2)
        with col1:
            scatter_x = st.selectbox('Select x variable:', st.session_state.numerical_vars, index=0)
        with col2:
            scatter_y = st.selectbox('Select y variable:', st.session_state.numerical_vars, index=len(st.session_state.numerical_vars)-1)
        submitted = st.form_submit_button("Show Scatter Plot")

        # If form is submitted, show the scatter plot
        if submitted:
            fig_scatter = px.scatter(df_copy, x=scatter_x, y=scatter_y, color=None)  # Use scatter_x as color column
            st.plotly_chart(fig_scatter)
            # Calculate and display correlation coefficient
            correlation_coef = pearsonr(df_copy[scatter_x], df_copy[scatter_y])[0]
            st.write(f"Correlation coefficient between {scatter_x} and {scatter_y}: {correlation_coef}")

    # Visualize scatter matrix
    st.subheader("Scatter Matrix:")
    st.write("Visualizing the relationship between numerical variables")
    if st.button("Show Scatter Matrix"):
        with st.spinner('Wait for it...'):
            fig, ax = plt.subplots(figsize=(12, 12))
            pd.plotting.scatter_matrix(df_copy[st.session_state.numerical_vars], ax=ax)
            st.pyplot(fig)

    # Visualize correlation matrix
    st.subheader("Correlation Matrix:")
    st.write("Visualizing the correlation between numerical variables")
    if st.button("Show Correlation Matrix"):
        with st.spinner('Wait for it...'):
            corr_matrix = df_copy[st.session_state.numerical_vars].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
            st.pyplot(fig)

    st.subheader("",divider='gray')         
    # Add data filtering section
    st.header("Data Filtering")
    #df_copy
    st.subheader('Numeric Filters')
    data_filter_num = st.container(border=True)
    with data_filter_num:
        for var in st.session_state.numerical_vars:
            min_val = df_copy[var].min()
            max_val = df_copy[var].max()
            min_val_select, max_val_select = st.slider(f'Select {var} range', min_val, max_val, (min_val, max_val), key=var)
            filtered_data = df_copy[(df_copy[var] >= min_val_select) & (df_copy[var] <= max_val_select)]
    
    st.subheader('Categorical Filters')
    data_filter_cat = st.container(border=True)
    with data_filter_cat:
        for var in st.session_state.categorical_vars:
            selected_values = st.multiselect(f'Select {var}', df_copy[var].unique(), key=var)
            filtered_data = df_copy[df_copy[var].isin(selected_values)]

    if filtered_data is not None:
        st.subheader('Filtered Results')
        st.write(filtered_data)

    

    

else:
    st.error("Please upload a CSV file on 1 data preprocessing page.")


pages = st.container(border=False) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/1_data_preprocessing.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/3_model_fitting.py")


