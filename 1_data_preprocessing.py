import streamlit as st
import pandas as pd
import numpy as np

text_deleted_successful=''

st.header("Data Preprocessing")
st.subheader("Upload a CSV file:")
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'nrows_raw' not in st.session_state:
    st.session_state.nrows_raw = None
if 'df_deleted' not in st.session_state:
    st.session_state.df_deleted = None
if 'df_changeNA' not in st.session_state:
    st.session_state.df_changeNA = None
if 'df_dropNA' not in st.session_state:
    st.session_state.df_dropNA = None

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df_raw = df
    st.session_state.nrows_raw = df.shape[0]

df = st.session_state.df_raw

# Retrieve the DataFrame from session state

if df is not None:
    # Show dataframe
    st.write(f"Preview of the uploaded dataset: `{df.shape[0]}`the numbers of rows ")
    st.dataframe(df)

    # Allow user to delete rows and columns
    st.subheader("Delete rows and columns:")

    # Create two columns for options
    col_delete_rows, col_delete_columns = st.columns(2)

    # Left column: Delete rows
    with col_delete_rows:
        st.write("Delete rows:")
        rows_to_delete = st.multiselect("Select rows to delete:", options=df.index.tolist(), default=None)

    # Right column: Delete columns
    with col_delete_columns:
        st.write("Delete columns:")
        columns_to_delete = st.multiselect("Select columns to delete:", options=df.columns.tolist(), default=None)

    left1, middle , right1 = st.columns([0.3,0.4,0.3])
    with middle:
        if st.button("Delete Selected Rows and Columns"):
            if rows_to_delete or columns_to_delete:
                df = df.drop(index=rows_to_delete, columns=columns_to_delete)
                st.session_state.df_deleted = df
                text_deleted_successful = True
            else:
                st.error("Please select rows or columns to delete.")

    if text_deleted_successful:
        st.write(":green[deleted successfully!] There is filtered data: the sample size is now", st.session_state.df_deleted.shape[0])
        st.dataframe(st.session_state.df_deleted)

    if st.session_state.df_deleted is not None:
        df = st.session_state.df_deleted
    else:
        df = st.session_state.df_raw

    # Show missing values information
    missing_values = df.isnull().sum()
    missing_values_transposed = missing_values.to_frame().T  # Transpose the DataFrame
    st.subheader("Missing values information:")
    st.dataframe(missing_values_transposed)
    if missing_values_transposed.all().all() == 0:
        st.write(":green[No missing values found in the dataset.]")

    # Additional functionality for handling missing values
    st.subheader("Handle potential missing values:")
    st.write("若資料遺失值不為NA，請輸入以下資訊 : 在以下變數中")

    # Allow user to select variables with potential missing values
    variables_with_potential_missing_values = st.multiselect("Select variables with potential missing values:", options=df.columns.tolist())

    # Display text input for missing value representation
    missing_value_representation = st.text_input("Enter missing value representation:", value="...")

    # Show missing value representation
    st.write(f"遺失值以 '{missing_value_representation}' 表示")

    # Update missing value representation in DataFrame
    if st.button("Update Missing Value Representation"):
        if variables_with_potential_missing_values and missing_value_representation:
            try:
                missing_value_representation = int(missing_value_representation)
            except ValueError:
                # 如果無法轉換為整數，則保留為字串型態
                pass
            for variable in variables_with_potential_missing_values:
                if df[variable].dtype == "object":  # 如果是字串型態，直接用replace替換
                    df[variable].replace(missing_value_representation, np.nan, inplace=True)
                else:  # 如果是數字型態，則轉換為浮點數再進行替換
                    missing_value = float(missing_value_representation)
                    df[variable].replace(missing_value, np.nan, inplace=True)
            st.write("Missing value representation updated successfully!")
            # Show missing values information again after updates
            missing_values = df.isnull().sum()
            missing_values_transposed = missing_values.to_frame().T
            st.dataframe(missing_values_transposed)
            st.dataframe(df)
            st.session_state.df_changeNA = df
        else:
            st.warning("Please select variables with potential missing values.")

    st.subheader("Delete rows with missing values")

    if st.session_state.df_changeNA is not None:
        df = st.session_state.df_changeNA
    else:
        if st.session_state.df_deleted is not None:
            df = st.session_state.df_deleted
        else:
            df = st.session_state.df_raw


    if st.button("Delete missing values"):
        df.dropna(inplace=True)
        st.write(f"Successfully deleted all rows with missing values！ The number of rows in the dataset is now: :red{df.shape[0]}")
        st.dataframe(df)
        st.session_state.df_dropNA = df


else:
    st.error("Please upload a CSV file.")






pages = st.container(border=False) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/0_example.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/2_data_visualization.py")







