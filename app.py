import streamlit as st
import pickle
import numpy as np

df = pickle.load(open("data.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))
df_new = pickle.load(open("df_new.pkl","rb"))

area = st.slider("House area in square fit",00,int(df_new["total_sqft"].max()))
# st.write(area)

bathroom = st.slider("number of bathroom in house",0,int(df_new["bath"].max()))
# st.write(bathroom)

balcony = st.slider("number of balcony",0,int(df_new["balcony"].max()))
# st.write(balcony)

size = st.slider("size in BHK",1,int(df_new["size"].max()))
# st.write(size)

area_type = st.selectbox("select area type",df["area_type"].unique())
# st.write(area_type)

# encode the area type data
transformation_dict = {'Built-up  Area':1,'Carpet  Area':2,'Plot  Area':3,'Super built-up  Area':4}
transformed_area_type = transformation_dict[area_type]

# convert into numpy array
data = np.array([transformed_area_type,size,area,bathroom,balcony],ndmin=2)

# scale the data
scaled_data = scaler.transform(data)

ans = model.predict(data)
st.write("Total price of house is ",ans*100000)
