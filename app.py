import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from pickle import load

df=pd.read_csv('merged_dataset1.csv')

selected = option_menu(
    menu_title = None,
    options=["Home", "Prediction"],
    icons = ['house', 'emoji-dizzy'],
    menu_icon = 'cast',
    default_index=0,
    orientation='horizontal'
)

if selected == 'Home':
    st.title('Amazing Mart Dataset')
    st.image('mart.jpeg')
    st.markdown("""---""")
    st.markdown("""---""")


    

    st.title('Mart Dataset💎')

    df1 = df.drop('Sales',axis = 1)
    st.dataframe(df1)

    st.subheader('Shape of Datasets')
    st.dataframe(df.shape)
    

if selected == 'Prediction':
      # Loading pretrained models from pickle file
    enc1=load(open('models/encoder.pkl','rb'))
    scaler1 = load(open('models/scaler.pkl', 'rb'))
    xgb1=load(open('models/xgboost.pkl','rb'))

    with st.form('my_form'):

        city = st.selectbox(label='City', options=df.City.unique())
        country = st.selectbox(label='Country', options=df.Country.unique())
        region = st.selectbox(label='Region', options=df.Region.unique())
        segment = st.selectbox(label='Segment', options=df.Segment.unique())
        ship_mode = st.selectbox(label='Ship Mode', options=df['Ship Mode'].unique())
        state = st.selectbox(label='State', options=df.State.unique())
        product_name = st.selectbox(label='Product Name', options=df['Product Name'].unique())
        sub_category = st.selectbox(label='Sub Category', options=df['Sub-Category'].unique())
        discount = st.number_input('Enter Discount : ')
        actual_discount = st.number_input('Enter Actual Discount : ')
        profit = st.slider('Enter Profit : ',min_value = 0,max_value=1000)
        quantity = st.slider('Enter Quantity : ',min_value = 1,max_value=20)
        year = st.slider('Enter Year : ',min_value = 2013,max_value=2025)
        
        btn = st.form_submit_button(label='Predict')

        if btn:
            if city and country and region and segment and ship_mode and state and product_name and sub_category and quantity and year :
                query_cat = pd.DataFrame({'City':[city], 'Country':[country],'Region':[region],'Segment':[segment],'Ship Mode':[ship_mode],'State':[state],'Product Name':[product_name],'Sub-Category':[sub_category]})
                query_num = pd.DataFrame({'Discount':[discount], 'Actual Discount':[actual_discount], 'Profit':[profit], 'Quantity':[quantity], 'Year':[year]}) 
            
                X_t_cat = pd.DataFrame(enc1.transform(query_cat),columns = query_cat.columns,index = query_cat.index)
                X_t_num = pd.DataFrame(scaler1.transform(query_num),columns = query_num.columns,index = query_num.index)

                query_point = pd.concat([pd.DataFrame(X_t_cat), pd.DataFrame(X_t_num)], axis=1)
                sales = xgb1.predict(query_point)
                st.success(f"The Sales is {round(sales[0],0)}")

            else:
                 st.error('Please enter all values')



