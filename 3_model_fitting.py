import streamlit as st
import pandas as pd
from collections import defaultdict
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

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


# 創建一個空的字典來存放 column name 和對應的值的型態
column_types = {}
model_data = pd.DataFrame()


# title
st.header("Model Fitting")

# data purposing
var_dict = defaultdict(list)

if "user_choose_y" not in st.session_state:
    st.session_state.user_choose_y = None

if "model_dataset" not in st.session_state:
    st.session_state.model_dataset = None

if df is not None:
    if  'categorical_vars' in st.session_state:
        categorical_vars = st.session_state.categorical_vars
        numerical_vars = st.session_state.numerical_vars
    else:
        data_types_trans = df.dtypes.to_frame().transpose()
        categorical_vars = data_types_trans.columns[data_types_trans.loc[0] == 'object'].tolist()
        numerical_vars = data_types_trans.columns[(data_types_trans.loc[0] == 'int') | (data_types_trans.loc[0] == 'float')].tolist()
    
    container_ModelFitting11 = st.container(border=True) 
    with container_ModelFitting11:
        st.write("<div style='padding-bottom: 0.5rem;'>數據分析要求：</div>", unsafe_allow_html=True)
        container_ModelFitting1_2, container_ModelFitting1_3, \
        container_ModelFitting1_4, container_ModelFitting1_5 = st.columns([1,0.19,1.7,0.6])
            
        with container_ModelFitting1_2:
            #加on_change 
            model_y = st.selectbox(options=numerical_vars, label="輸入y變數", index=None, placeholder="選取單一應變數(Y)",label_visibility="collapsed")
            if model_y:
                st.session_state.user_choose_y = model_y
        with container_ModelFitting1_3:
            st.write("和")
            #st.write("<div style='display: flex; align-items: center;'>和</div>", unsafe_allow_html=True)
        with container_ModelFitting1_4:
            varX = list(set(categorical_vars + numerical_vars) - {model_y})
            #加on_change
            model_x = st.multiselect(options=varX, placeholder="選取自變數(X)",label="輸入x變數",label_visibility="collapsed")
        with container_ModelFitting1_5:
            st.write("之間的關係")

    # show the categorical variables, numerical variables
    numeric_x = list(set(model_x) & set(numerical_vars))
    category_x = list(set(model_x) & set(categorical_vars))

    container_ModelFitting12 = st.container(border=True)
    with container_ModelFitting12:
        st.write("<div style='padding-bottom: 0.5rem;'>區分所選取的變數類別：</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text_area( label="Y-連續型變數",value= model_y)
        with col2:
            st.text_area(label="X-連續型變數",value= ", ".join(numeric_x))
        with col3:
            st.text_area(label="X-類別型變數",value= ", ".join(category_x))

    # 確認變數間的關係
    # drawing scatter matrix plot for all selected numeric variables \
        st.write("<div style='padding-bottom: 0.5rem;'>將模型選用的連續形變數繪製散佈矩陣圖：</div>", unsafe_allow_html=True)
        if len(numeric_x) > 0:
            df_numeric = df[[model_y]+ numeric_x]
            with st.spinner('Wait for it...'):
                scatter_matrix = sns.pairplot(df_numeric, diag_kind='kde')
                st.pyplot(scatter_matrix)
        # 解讀散步矩陣圖 描述是否有關係以集關係的強弱
            all_x = ", ".join(numeric_x)
            text = ""
            for var in numeric_x:
                text += f"觀察 `{var}` 與 `{df_numeric.columns[0]}` 之間是否存在線性關係，"
            text = text[:-1]
            st.markdown("解讀散佈矩陣圖：")
            st.markdown(f"- 請注意自變數與應變數之間的線性關係：{text}。這些關係可能暗示著 `{all_x}` 對於預測 `{df_numeric.columns[0]}` 有顯著影響。")
            if len(numeric_x) > 1:
                st.markdown(f"- 請注意自變數之間的相關性：觀察 `{all_x}` 變數兩兩之間是否存在某種關聯。")
        # for categorical variables, draw boxplot for each variable and scatter plot for x is order and group by the categorical variables
            # 繪製箱形圖和散點圖

    # dealing with categorical variables  
    #if len(category_x) > 0:
    if category_x:
        need_to_dummy = []
        container_ModelFitting13 = st.container(border=True)   
        with container_ModelFitting13:
            st.write("<div style='padding-bottom: 0.5rem;'>類別變數的處理：</div>", unsafe_allow_html=True)
            for var in category_x:
                grouped_data = df.groupby(var)
                categories_level = list(grouped_data.groups.keys())
                if len(categories_level) > 1:
                    need_to_dummy.append(var)
                    dummyvar = []
                    for levels in categories_level[1:]:
                        new_var = f"{var}_{{{levels}}}"
                        dummyvar.append(new_var)
                    dummy_text = f"將其轉換為虛擬變數：`{dummyvar}`"
                else:
                    dummy_text = f"只有一個類別，該變數不適合納入模型。"
                
                st.write(f"- 類別變數 `{var}` 的類別數量為 `{len(categories_level)}`，值為 `{categories_level}`，{dummy_text}")
                if len(categories_level) > 1:
                    for dummy in dummyvar :
                        parts = dummy.split("_")
                        st.write(f"which ${dummy}$ is :")
                        st.markdown(rf" $ {dummy} = \begin{{cases}} 1 & ,\;\; \text{{if }} {var} = {parts[1]} \\ 0 & ,\;\; \text{{otherwise}} \end{{cases}} $")

        # create dummy variables in dataframe
        df_dummy = pd.get_dummies(df, columns=need_to_dummy, drop_first=True)
        new_dummy_variables = df_dummy.loc[:, df_dummy.columns.difference(df.columns)]
    # build the model
        # output a general model
        #根據上方的關係 考慮add polynomial terms 、 transformation terms
            # add polynomial terms and example
            # add transformation terms and example
        # output the final model
    if numeric_x:
        selected_columns = numeric_x.copy() 
        selected_columns.insert(0, model_y)
        model_data = df[selected_columns]
    if category_x:
        if not new_dummy_variables.empty:
            model_data = model_data.merge(new_dummy_variables, left_index=True, right_index=True)

    st.session_state.model_dataset = model_data
    # model selection
    st.subheader("Model Selection")
    tab1, tab2, tab3, tab4, tab5= st.tabs(["📈 first-order", "🗃 second-order","🗃 first-order with interaction ", "🗃 second-order with interaction", "🗃 others"])
    with tab1:
        st.markdown("The multiple regression equation with an intercept term can be written as:")
        if not model_data.empty:
            y_var = model_data.columns[0]
            x_vars = model_data.columns[1:]
            equation = f"$$ {y_var} = β₀ + "
            for idx, var in enumerate(x_vars, start=1):
                equation += f"β_{idx} {var} + "
            equation += f"ε $$"
            st.markdown(equation)
            st.markdown(r"-  use least square method or MLE to fit the model : $\beta = (X^T X)^{-1}X^T Y$")
            # 創建一個線性回歸模型的實例
            model_firstorder = LinearRegression()
            # 使用X_matrix和Y_vector來擬合模型
            X = model_data.iloc[:, 1:]
            Y = model_data.iloc[:, 0]
            model_firstorder.fit(X, Y)
            # 取得估計的係數
            beta_sklearn = np.insert(model_firstorder.coef_, 0, model_firstorder.intercept_)
            equation_est = f"${y_var}$ = `{round(beta_sklearn[0], 2)}`"
            equation_est_mean = f"$E({y_var})$ = `{round(beta_sklearn[0], 2)}`"
            func = ""
            interpretation = f"- This estimated regression function indicates that ：\n"
            for i, beta in enumerate(beta_sklearn[1:], start=1):
                func += f" + `{round(beta, 2)}`${x_vars[i-1]}$"
                interpretation += f"   - :red[ the mean of {y_var}] are expected to change by {beta:.2f} units when the {x_vars[i-1]} increases by 1 unit, holding  other constant\n"
            #st.markdown( equation_est_mean+func)
            #st.markdown(interpretation)
            # 解讀係數
            #st.markdown("- 解讀係數：")
            #st.markdown("   - This estimated regression function indicates that mean y are expected to increase/decrease by beta1 單位")
            # when the x1 increases by 1 單位, holding  x2 constant, and that mean y are expected to increase/decrease 
            # by beta2 單位 when per x2 increases by 1 單位, holding the x1 constant
            func += " + $residuals$"
            st.markdown(equation_est+func)


            
        else:
            markdown_text = """

    $$
    Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\ldots + \\beta_n X_n + \\varepsilon
    $$

    Where:
    - $Y$ is the dependent variable,
    - $\\beta_0 $ is the intercept term,
    - ${\\beta_1, \\beta_2, \\ldots, \\beta_n }$ are the regression coefficients corresponding to the independent variables ${ X_1, X_2, \ldots, X_n }$ respectively,
    - ${X_1, X_2, \ldots, X_n }$ are the independent variables,
    - $\\varepsilon $ is the error term.

    **Assumptions of the error term $\\varepsilon $:**
    1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
    2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
    3. The error term $ \\varepsilon $ is normally distributed.
    4. The error terms are independent of each other.

    These assumptions are important for making statistical inferences using regression analysis.
    """  
            st.markdown(markdown_text)


    with tab2:
        st.markdown("The multiple regression equation with an intercept term can be written as:")
        if not model_data.empty:
            model_data
        else:
            markdown_text = """

    $$
    Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\ldots + \\beta_n X_n + \\beta_{n+1} {X_1}^2 + \\beta_{n+2} {X_2}^2 + \\ldots + \\beta_{2n} {X_n}^2 + \\varepsilon
    $$

    Where:
    - $Y$ is the dependent variable,
    - $\\beta_0 $ is the intercept term,
    - ${\\beta_1, \\beta_2, \\ldots, \\beta_{2n} }$ are the regression coefficients corresponding to the independent variables $X_1, X_2, \ldots, X_n ,{X_1}^2, {X_2}^2, \ldots, {X_n}^2$ respectively,
    - $X_1, X_2, \ldots, X_n ,{X_1}^2, {X_2}^2, \ldots, {X_n}^2 $ are the independent variables,
    - $\\varepsilon $ is the error term.

    **Assumptions of the error term $\\varepsilon $:**
    1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
    2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
    3. The error term $ \\varepsilon $ is normally distributed.
    4. The error terms are independent of each other.

    These assumptions are important for making statistical inferences using regression analysis.
    """  
            st.markdown(markdown_text)


    with tab3:
        st.markdown("The multiple regression equation with an intercept term can be written as:")
        if not model_data.empty:
            model_data
        else:
            markdown_text = """

    $$
    Y = \\beta_0 + \\beta_1 X_1 + \\ldots + \\beta_n X_n + \\beta_{n+1} X_1 X_2 + \\beta_{n+2} X_1 X_3 + \\ldots  + \\varepsilon
    $$

    Where:
    - $Y$ is the dependent variable,
    - $\\beta_0 $ is the intercept term,
    - ${\\beta_1, \\beta_2, \\ldots, \\beta_n, \\beta_{n+1}, \\beta_{n+2} , \\ldots}$ are the regression coefficients corresponding to the independent variables ${ X_1, X_2, \ldots, X_n, X_1 X_2 , X_1 X_3 , \\ldots}$ respectively,
    - ${X_1, X_2, \ldots, X_n }$ are the independent variables,
    - $ X_1 X_2 , X_1 X_3 , \\ldots $ are the interaction terms,
    - $\\varepsilon $ is the error term.

    **Assumptions of the error term $\\varepsilon $:**
    1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
    2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
    3. The error term $ \\varepsilon $ is normally distributed.
    4. The error terms are independent of each other.

    These assumptions are important for making statistical inferences using regression analysis.
    """  
            st.markdown(markdown_text)
            

    with tab4:
        st.markdown("The multiple regression equation with an intercept term can be written as:")
        if not model_data.empty:
            model_data
        else:
            markdown_text = """

    $$
    Y = \\beta_0 + \\beta_1 X_1 + \\ldots + \\beta_n X_n + \\beta_{n+1} {X_1}^2 + \\ldots + \\beta_{2n} {X_n}^2 + \\beta_{2n+1} X_1 X_2 + \\beta_{2n+2} X_1 X_3 + \\ldots  +\\varepsilon
    $$

    Where:
    - $Y$ is the dependent variable,
    - $\\beta_0 $ is the intercept term,
    - ${\\beta_1, \\beta_2, \\ldots, \\beta_{2n+2} \\ldots }$ are the regression coefficients corresponding to the independent variables $X_1 \ldots X_n ,{X_1}^2 \ldots {X_n}^2, X_1 X_2 , X_1 X_3 \\ldots$ respectively,
    - $X_1 \ldots X_n ,{X_1}^2 \ldots {X_n}^2 $ are the independent variables,
    - $ X_1 X_2 , X_1 X_3 \\ldots $ are the interaction terms,
    - $\\varepsilon $ is the error term.

    **Assumptions of the error term $\\varepsilon $:**
    1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
    2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
    3. The error term $ \\varepsilon $ is normally distributed.
    4. The error terms are independent of each other.

    These assumptions are important for making statistical inferences using regression analysis.
    """  
            st.markdown(markdown_text)


    with tab5:
        st.markdown("The multiple regression equation with an intercept term can be written as:")
        st.text_area( label="請依照格式輸入模型",value= "$Y = beta_0 + beta_1*X_1 + beta_2*X_2 + \\ldots + beta_n*X_n + \\varepsilon$")

else:
    st.error("Please upload a CSV file on data preprocessing page.")

# model fitting 
    #say: use least square method or MLE to fit the model
    #model result show係數 解讀:
        # This estimated regression function indicates that mean y are expected to increase/decrease by beta1 單位 
        # when the x1 increases by 1 單位, holding  x2 constant, and that mean y are expected to increase/decrease 
        # by beta2 單位 when per x2 increases by 1 單位, holding the x1 constant


pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/2_data_visualization.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/4_residual.py")