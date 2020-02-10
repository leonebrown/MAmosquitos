import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import datetime
from sklearn.linear_model import LogisticRegression
import pickle

st.title('Mosquito-borne disease risk assessor')
st.write("User-friendly web application to assess seasonal mosquito-borne disease risk across Massachusetts")

user_input = st.text_input("Search town, e.g., 'Somerville'", "Somerville")
#geolocator = Nominatim(user_agent="my-application")

#def do_geocode(user_input):
#    try:
#        return geopy.geocode(user_input)
#    except GeocoderTimedOut:
#        return do_geocode(user_input)

#try:
#location = geolocator.geocode(user_input)
#print(loc.raw)
#print('Coordinates: ', location.latitude, location.longitude)
#st.write("Found: ", location)
#st.write('Coordinates: ', location.latitude, location.longitude)
    
#x = location.longitude
#y = location.latitude

disease = ['West Nile virus','Eastern Equine Encephalitis']

moslist = ['January','February','March','April','May','June','July','August','September','October','November','December']

datelist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

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
#read in landcover data to get land cover for each town
#user will input a town, and for the prediction, pull out landcover values that match that town
LCdat2011 = pd.read_csv("townLC2011fin_2.csv")

#read in climate data x town x day of year (averaged over the past 5 years)
climdat = pd.read_csv("predDFclim2.csv")

# load the model from disk
if option1 == 'Eastern Equine Encephalitis' :
    loaded_model = pickle.load(open('logreg_EEEmod.sav','rb'))
else:
    loaded_model = pickle.load(open('logreg_WNVmod.sav','rb'))

inputdate = option2 + str(" ") + str(option3) + str(" ") + str(2020)

#get date and convert to day of year
#date_object = datetime.strptime(str, '%m/%d/%y')
dateTime = datetime.datetime.strptime(inputdate, "%B %d %Y")
DOY = dateTime.strftime('%j')

#filter landcover dataframe to only have town = userinput
LCtemp = LCdat2011[LCdat2011['town']==user_input.upper()]

cc = LCtemp['RPcultcrop'].iloc[0]
ss = LCtemp['RPshrubscrub'].iloc[0]
do = LCtemp['RPdevopen'].iloc[0]
dl = LCtemp['RPdevlow'].iloc[0]
dm = LCtemp['RPdevmed'].iloc[0]
dh = LCtemp['RPdevhigh'].iloc[0]
gh = LCtemp['RPgrassherb'].iloc[0]
ph = LCtemp['RPpasturehay'].iloc[0]
ow = LCtemp['RPopenwater'].iloc[0]
ehw = LCtemp['RPemergherbwet'].iloc[0]
ww = LCtemp['RPwoodywet'].iloc[0]
dfor = LCtemp['RPdecidforest'].iloc[0]
efor = LCtemp['RPevergrnfor'].iloc[0]
mfor = LCtemp['RPmixedforest'].iloc[0]

#filter climate dataframe to only have town = userinput
cltemp = climdat[climdat['town']==user_input.upper()]

cltemp['DOY'] = cltemp['DOY'].apply(str)
cltemp['DOY'] = cltemp['DOY'].str.zfill(3)

ind = cltemp.loc[cltemp['DOY']== DOY]

T7 = ind.iat[0,2]
T14 = ind.iat[0,3]
T21 = ind.iat[0,4]
p7 = ind.iat[0,5]
p14 = ind.iat[0,6]
p21 = ind.iat[0,7]

d = {'DOY':[DOY],'RPcultcrop':[cc],'RPshrubscrub':[ss],'RPdevopen':[do],
       'RPdevlow':[dl],'RPdevmed':[dm],'RPdevhigh':[dh],'RPgrassherb':[gh],'RPpasturehay':[ph],
       'RPopenwater':[ow],'RPemergherbwet':[ehw],'RPwoodywet':[ww],'RPdecidforest':[dfor],
       'RPevergrnfor':[efor],'RPmixedforest':[mfor],'ppt7':[p7],'ppt14':[p14],'ppt21':[p21],'avgT7':[T7],'avgT14':[T14],'avgT21':[T21]}

dfpredtest = pd.DataFrame(data=d)
dog = loaded_model.predict(dfpredtest)

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

st.write("Your local", print(option1), "risk per mosquito in", print(user_input), "on", print(inputdate), "is", print(dog))

st.write("Your local risk is calculated based on the number of mosquitos that have tested positive in your area over the past 16 years, as well as climate and land cover variables that affect infection rate in mosquitos. This estimate is the probability that any given mosquito biting you has West Nile virus (WNv) Eastern Equine Encephalitis (EEE). WNv and EEE are rare; for West Nile virus in particular, approximately 80% of infected people never show symptoms. Those that do show symptomes are typically those over the age of 40-50. If you are over the age of 60 you are at greatest risk. Please see https://www.cdc.gov/features/westnilevirus/index.html for more information.")

st.write("Disclaimer: This tool is not meant to provide a medical evaluation or replace the advice of a medical professional. This tool was implemented using a logistic regression machine learning algorithm as part of the developer's participation in the Insight Health Data Science program in Boston, Massachusetts, USA (https://www.insighthealthdata.com). Please use this tool at your own discretion, and always protect yourself from biting insects and other animals that may be vectors of zoonotic diseases.")

st.write("Accompanying slides found here: https://drive.google.com/open?id=1EBigwWTrTQufbD6xVlndKLScqe_h1GxcsvLa4H1DXhM")
