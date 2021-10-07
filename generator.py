from os import write
from google.protobuf.symbol_database import Default
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from streamlit.state.session_state import Value
import plotly as pl
import plotly.express as pex
import plotly.graph_objects as go
import io
from sdv.tabular import CTGAN # GAN for Implementation
import warnings
warnings.filterwarnings("ignore")

## Function to create CTGAN Model
#def gansamples(df, n):
    #model = CTGAN()
    #model.fit(df)
    #newdata = model.sample(n)
    #return(newdata)

#Asthetics of the Page###
res = f"Welcome to Data Generator ✍️"
st.title(res)
st.subheader("Synthetic Data Powered by GANs")
st.sidebar.header("Synthetic Data Generator 🛠️")

miss_func = st.container()

with miss_func:
    def na_vals(df, col, miss):
        n = int(np.round(df.shape[0]*miss))
        samp = df.sample(n, random_state = 100)
        index = samp.index
        for i in range(0, n):
            df.loc[index[i], col] = np.nan
        return(df)

outlier = st.container()
plot_box = st.container()

#with plot_box:
    #def makeboxplots(df):
        #return(df.plot(kind = "box"))


#### File Uploader #######
uploaded_file = st.sidebar.file_uploader("Choose a .csv file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding = "utf-8")
    st.caption("Preview of the Data Uploaded")
    ##### List of Numerical & Categorical Columns ##########
    num_cols = df.select_dtypes(include = np.number).columns
    cat_cols = df.select_dtypes(exclude = np.number).columns
    ##### Printing the DataFrame Imported
    st.write(df.head())
   
    #################### S Y N T H E T I C D A T A  G E N E R A T O R  #########################
    
    # Synthetic Data SelecBox
    select_yes = st.sidebar.selectbox(label="Do you Want to Generate Synthetic Data", 
                         options = ["Yes", "No"])
    gan_input = st.sidebar.number_input("Enter the Sample Count for GAN", min_value = 0, max_value=1000)
        
      
  
    ###################### M I S S I N G V A L U E S L I D E R############################
    st.sidebar.header("Set the Noise in the Data ✒️")
    missingval_slider = st.sidebar.slider(label = "Drag Slider to Introduce the Missing Values",
                                    min_value =0, max_value = 25)/100
    msg = f"{missingval_slider} missing values will be introduced in the Data"
    st.sidebar.caption(msg)
    
    ################### C O L U M N S N A M E S I D E B A R #############################
    # Missing Values
    miss_table = df.isnull().sum()[df.isnull().sum()!=0].reset_index(name = "Miss_Count")
    missingitem_list = miss_table["index"].tolist()
    ####### If there are missing values, those column names should not show in the Dropdown #####
    cols = [] # Blank List
    for i in df.columns:
        if i not in missingitem_list:
            cols.append(i)  
    
    ################ M U L T I S E L E C T S L I D E R #################################
    #### cols is passing the Conditional List of the Feature Names...####
    colname = list(st.sidebar.multiselect("Select the Column(s) for Missing Values", cols))
    if len(colname)==0:
        st.sidebar.error("⛔ No Col is Selected")
    else:
        pass
    #if colname in num_cols:
        #st.sidebar.caption("Message: Above Column is Numerical Col")
    #else:
        #st.sidebar.caption("Message: Above Columns is Categorical Col")
    st.sidebar.caption("Tip: You can Select More than One Column.")
    st.sidebar.caption("Note:- Exclude Target Variable in Noise.")
    
    ########## O u t l i e r S l i d e r ##################################
    #st.sidebar.header("Introducing Outliers in the Data")
    #outliers = st.sidebar.selectbox("Select the Outlier Option", ["Yes", "No"], index = 0)
    #msg = f" Note - The Outliers are Introduced using IQR Method"
    #st.sidebar.caption(msg)
    
    ############## D A T A U P L O A D S T A T U S ###########
    st.subheader("🔰 About the Data")
    len_num = len(num_cols)
    len_cat = len(cat_cols)
    num_cat = f"The Data Contains **{len_num} NUM columns** & **{len_cat} CAT columns**"
    st.write(num_cat)

    rows_cols = f"The Data Contains **{df.shape[0]}** rows and **{df.shape[1]}** columns"
    st.write(rows_cols)

    list_num = f"Numerical Columns: \
        **{df.select_dtypes(include = np.number).columns.tolist()}**"
    st.markdown(list_num)

    list_cat = f"Categorical Columns: \
        **{df.select_dtypes(exclude = np.number).columns.tolist()}**"
    st.markdown(list_cat)
    
    
    st.subheader("Status of Missing Values")
    if miss_table["Miss_Count"].sum()!=0:
        st.write(miss_table)
        st.write(np.round(100*(miss_table["Miss_Count"].sum()/df.shape[0]),2), \
            "% Values are Missing from Data.")
        #st.error("No Missing Values Found in Dataset")
    else:
        st.info("No Missing Values Found in the Data")
    #################### Introducing Missing Values In the Dataset#################
    # colname stores the value in the MissingValues Dropdown
    # Function to Introduce Missing Values
    
    
    #st.write(sampledata.shape)
    
     # Missing Values Inserted in DataFrame
    pressed = st.sidebar.button("Prep the Data ⛏️", True)
    if pressed:
        #callable(newdf) # Function returns true/false
        if (df.shape[0]<1000 and select_yes=="Yes"):
            mymodel = CTGAN() 
            mymodel.fit(df)
            sampledata = mymodel.sample(gan_input)
            newdf = na_vals(sampledata, colname, missingval_slider)
            
        else:
            st.info("Cannot Generate More Samples as the Shape is Greater than 1000 Obs")
            newdf = na_vals(df, colname, missingval_slider)
        
        #newdf = na_vals(sampledata, colname, missingval_slider)
        
        if missingval_slider==0:
            st.subheader("Missing Value Status")
            st.error("Missing Value Percentage is 0. Drag the Slider to Introduce Missing Values")
        else:
            st.subheader("New DataFrame with Noise")
            msg = f"Note: Check if the Selected Columns are showing in the table below. "
            st.info(msg)
            st.write("Missing Value(s) Table")
            st.write(newdf.isnull().sum()[newdf.isnull().sum()!=0].reset_index(name = "MissCount"))
            st.caption("New DataFrame with Missing Values")
            st.dataframe(newdf.head()) # Preview of the Dataset with missing columns
            st.write("New Data Frame Shape:", newdf.shape)
            st.success("Success: The Missing Values have been Created")
        
        ############### Outlier Analysis #######################
        st.markdown("---")
        st.subheader("Boxplot showing Outliers in Data")
        # Creating Boxplot of the DataFrame
        
        fig = pex.box(df.loc[:, num_cols])
        st.write(fig)
        #st.write(df.describe())
        #st.write("Price has the Highest No of Values and therefore it has Outliers..")
        
        ################################ DOWNLOAD THE FILE CODE #######################################
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine = "xlsxwriter") as writer:
            newdf.to_excel(writer, index = False)
            writer.save()
        pressed = st.download_button(label = " 📝 Download the File (.xlsx)", data = buffer, 
                                     file_name="data.csv", mime="application/vnd.ms-excel")