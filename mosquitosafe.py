import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


st.title('Mosquito-borne disease risk assessor')
st.write("User-friendly web application to assess seasonal mosquito-borne disease risk across Massachusetts")

user_input = st.text_input("Search town, e.g., 'Somerville, MA'", "Somerville, MA")
geolocator = Nominatim(user_agent="my-application")

def do_geocode(user_input):
    try:
        return geopy.geocode(user_input)
    except GeocoderTimedOut:
        return do_geocode(user_input)

#try:
location = geolocator.geocode(user_input)
#print(loc.raw)
print('Coordinates: ', location.latitude, location.longitude)
st.write("Found: ", location)
st.write('Coordinates: ', location.latitude, location.longitude)
    
#except:
#   st.write("Couldn't find this location. Try a different town name?") #   location = geolocator.geocode('Burlington, Vermont')
x = location.longitude
y = location.latitude

disease = ['West Nile virus','Eastern Equine Encephalitis']

moslist = ['January','February','March','April','May','June','July','August','September','October','November','December']

datelist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

#df

#user_input2 = st.text_input("Enter date of interest as month-day")
#st.write(user_input2)

option1 = st.selectbox(
    'Search Disease',
     disease)

option2 = st.selectbox(
    'Search Month',
     moslist)

option3 = st.selectbox(
    'Search Day of Month',
     datelist)

##################################################################
# Run ML model
df3 = pd.read_csv("/Users/lbrown01/Dropbox/DataScienceStuff/Data/WNV_EE/MLscaffolddf.csv")

from sklearn.model_selection import train_test_split
X = df3.drop(['ResVal'], axis = 1) #axis = 1 for columns; need to drop response because I don't want to predict on response or get perfect model
y = df3['ResVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

prediction = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))  #very bad because too few 0s

#day 193 is July 12
d = {'POP2010': [81360], 'DOY': [193]}
dfpredtest = pd.DataFrame(data=d)

dog = logmodel.predict(dfpredtest)

st.write("Your WNV risk score is ", print(dog))