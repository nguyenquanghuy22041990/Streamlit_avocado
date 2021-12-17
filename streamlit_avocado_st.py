# import libraries
# pip install pyqt5
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot

# from pmdarima import auto_arima

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


from sklearn.preprocessing import LabelEncoder

import pickle

st.markdown(f'''
    <style>
    section[data-testid="stSidebar"] .css-ng1t4o {{width: 30rem;}}
    </style>
''',unsafe_allow_html=True)

# Source Code
data = pd.read_csv("avocado.csv")

#--------------
# GUI
st.title("Data Science Project")
st.write("### Hass Avocados Price Prediction")
# Upload file

# Filter Avocados - California
# Make new dataframe from original dataframe: data

df_ca = data[data['region'] == 'California']
df_ca['Date'] = pd.to_datetime(df_ca['Date'])
df_ca = df_ca[df_ca['type'] == 'organic']

agg = {'AveragePrice': 'mean'}
df_ca_gr = df_ca.groupby(df_ca['Date']).aggregate(agg).reset_index()
df_ca_gr.head()

df_ts = pd.DataFrame() 
df_ts['ds'] = pd.to_datetime(df_ca_gr['Date']) 
df_ts['y'] = df_ca_gr['AveragePrice'] 
df_ts.head()

# Train/Test Prophet
# create test dataset, remove last 10 months
train = df_ts.drop(df_ts.index[-10:])
test = df_ts.drop(df_ts.index[0:-10])

# Build model
model = Prophet(yearly_seasonality=True, \
            daily_seasonality=False, weekly_seasonality=False)

# # Serialize with Pickle
# with open('facebook_prophet.pkl', 'wb') as pkl:
#     pickle.dump(model, pkl)

# with open('facebook_prophet.pkl', 'rb') as pkl:
#     model = pickle.load(pkl)

model.fit(train)

# 10 month in test and 12 month to predict new values
months = pd.date_range('2017-01-01','2019-12-01', 
              freq='MS').strftime("%Y-%m-%d").tolist()    
future = pd.DataFrame(months)
future.columns = ['ds']
future['ds'] = pd.to_datetime(future['ds'])

# Use the model to make a forecast
forecast = model.predict(future)

# calculate MAE/RMSE between expected and predicted values
y_test = test['y'].values
y_pred = forecast['yhat'].values[:10]
mae_p = mean_absolute_error(y_test, y_pred)
# print('MAE: %.3f' % mae_p)
rmse_p = sqrt(mean_squared_error(y_test, y_pred))

# visualization
y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']),columns=['Actual'])
y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']),columns=['Prediction'])


m = Prophet(yearly_seasonality=True, \
            daily_seasonality=False, weekly_seasonality=False)

# # Long-term prediction for the next 5 years => Consider whether to expand cultivation/production, and trading
# with open('facebook_prophet.pkl', 'rb') as pkl:
#     m = pickle.load(pkl)

m.fit(df_ts)
future_new = m.make_future_dataframe(periods=12*5, freq='M') # next 5 years
forecast_new = m.predict(future_new)

region_df = data.copy(deep=True)
region_df['Date'] = pd.to_datetime(region_df['Date'])
sub_df = region_df[['Date', 'region', 'AveragePrice', 'type']]

# Arima
# arima_organic_df = data.copy(deep=True)
# arima_organic_df = arima_organic_df.loc[(arima_organic_df['type'] == 'organic') & (arima_organic_df['region'] == 'California')]
# arima_organic_df['Date'] = pd.to_datetime(arima_organic_df['Date'])
# arima_organic_df['value'] = arima_organic_df['AveragePrice']
# arima_organic_df = arima_organic_df[['Date', 'value']]
# arima_organic_df.columns = ['date', 'value']
# arima_organic_df = arima_organic_df.sort_values(by=['date'])
# arima_organic_df.reset_index(drop=True, inplace=True)
# arima_organic_df.set_index('date', inplace=True)
# arima_organic_df.info()

# arima_model = auto_arima(arima_organic_df, start_p=2, d=1, 
#                         start_q=2, max_p=5,
#                         max_q=5, start_P=0, D=1, 
#                         start_Q=0, max_P=5,
#                         m=52, seasonal=True, 
#                         error_action='ignore', trace = True, 
#                         supress_warnings=True, 
#                         stepwise = True)

# # Serialize with Pickle
# with open('arima.pkl', 'wb') as pkl:
#     pickle.dump(arima_model, pkl)


# with open('arima.pkl', 'rb') as pkl:
#     arima_model = pickle.load(pkl)


# arima_train = arima_organic_df[arima_organic_df.index.year < int(2017)]
# arima_test = arima_organic_df[arima_organic_df.index.year >= int(2017)]

# arima_model.fit(arima_train)

# For serialization:


# # Serialize with Pickle
# with open('arima.pkl', 'wb') as pkl:
#     pickle.dump(arima_model, pkl)

#1. USA's Avocados Average Prediction

regression_df = data.copy(deep=True)

def convert_month(month):
    if month == 3 or month == 4 or month == 5:
        return 0
    elif month == 6 or month == 7 or month == 8:
        return 1
    elif month == 9 or month == 10 or month == 11:
        return 2
    else:
        return 3

regression_df['Date'] = pd.to_datetime(regression_df['Date'])
regression_df['Month'] = pd.DatetimeIndex(regression_df['Date']).month
regression_df['Season'] = regression_df['Month'].apply(lambda x: convert_month(x))

le = LabelEncoder()
regression_df['type_new'] = le.fit_transform(regression_df['type'])

df_ohe = pd.get_dummies(data=regression_df, columns=['region'])

X = df_ohe.drop(['Date', 'AveragePrice', 'type', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags'], axis=1)
y = regression_df['AveragePrice']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.3,
                                                   random_state=0)


#=====================================================================================================================================================

# 3. Conventional Avocados in California


# -----------------------------------------------------

# GUI

menu = ["Business Objective", "Build Project", 
"1. USA's Avocados Average Prediction", 
"2. Organic Avocados in California - Time Series", 
"3. California's Conventional Avocados - Average Prediction",
"4. California's Conventional Avocados - Time Series",
"5. Boise's Avocados trend",
"6. Find the trend of regions in the future"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == "Business Objective":
    st.subheader("Business Objective")
    st.write("""
    ###### At present, the company which sell avocado in lots of region in USA with 2 type including conventional and organic one, packed into 3 sizes (Small/Large/XLarge Bags with 3 PLU ( Product Look Up) – 4046, 4225,4770. However, they don’t have average price prediction for business expansion.
    """)
    st.write("""
    #### => Objective:
    Build average price prediction model of “Hass” Avocado in USA and then consider to expand the business
    """)
    st.image("Hass_avocado_2.jpg")

    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data.to_csv("avocado_new.csv", index = False)

elif choice == "Build Project":
    import seaborn as sns

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 32px;">Build Project</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.write("""
    ##### Some data:
    """)

    st.dataframe(df_ts.head(3))
    st.dataframe(df_ts.tail(3))
    # st.text("Mean of Organic Avocados AveragePrice in California: " + str(round(df_ts['y'].mean(), 2)) + " USD")
    text = "Mean of Organic Avocados AveragePrice in California: " + str(round(df_ts['y'].mean(), 2)) + " USD"
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">' + text + '</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    text = "Calculate MAE/RMSE between expected and predicted values"
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">' + text + '</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.code("MAE: " + str(round(mae_p, 2)))
    st.code("RMSE: " + str(round(rmse_p, 2)))
    
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">This result shows that Prophet\'s RMSE and MAE are good enough to predict the organic avocado AveragePrice in California</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">Visualization: AveragePrice v AveragePrice Prediction</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    # Visualize the result
    fig, ax = plt.subplots()
    ax.plot(y_test_value, label='AveragePrice')
    ax.plot(y_pred_value, label="AveragePrice Prediction")
    ax.set_xticklabels(y_test_value.index.date, rotation=60)
    ax.legend()
    st.pyplot(fig)

    st.image("horizontal_grayline.png")

    # Organic
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">Organic:</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data = regression_df[regression_df['type'] == 'organic'],
            x="Season", y="AveragePrice", ax=ax)
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

    st.image("horizontal_grayline.png")

    # Conventional
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">Conventional:</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data = regression_df[regression_df['type'] == 'conventional'],
            x="Season", y="AveragePrice", ax=ax)
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">\'AveragePrice\' was affected by \'Season\' (both in \'organic\' type and \'conventional\' type)</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    # Organic
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">Organic:</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(data=regression_df[regression_df["type"] == "organic"],
            x="region", y="AveragePrice", ax=ax)
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

    st.image("horizontal_grayline.png")

    # Conventional
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">Conventional:</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(data=regression_df[regression_df["type"] == "conventional"],
            x="region", y="AveragePrice", ax=ax)
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">* Some regions have high prices: Sanfrancisco, Chicago...</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">* Some regions have low prices: Houston, PhoenixTucson...</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">=> \'AveragePrice\' was affected by \'region\' (both in \'organic\' type and \'conventional\' type)</p>'
    st.markdown(new_title, unsafe_allow_html=True)

elif choice == "1. USA's Avocados Average Prediction":

    # LinearRegression
    # ln_model = LinearRegression()
    # ln_model.fit(X_train, y_train)

    # with open('1_LR_model.pkl', 'wb') as pkl:
    #     pickle.dump(ln_model, pkl)

    with open('1_LR_model.pkl', 'rb') as pkl:
        ln_model = pickle.load(pkl)

    y_pred_LR = ln_model.predict(X_test)
    
    mae_LR = mean_absolute_error(y_test, y_pred_LR)


    # Random Forest
    # rf_model = RandomForestRegressor(n_estimators=500, 
    #                                 min_samples_split=2, 
    #                                 min_samples_leaf=1, 
    #                                 max_features='sqrt', 
    #                                 max_depth=None, 
    #                                 bootstrap=False)
    # rf_model.fit(X_train, y_train)

    # with open('1_RF_model.pkl', 'wb') as pkl:
    #     pickle.dump(rf_model, pkl)

    # with open('1_RF_model.pkl', 'rb') as pkl:
    #     rf_model = pickle.load(pkl)

    # y_pred_RF = rf_model.predict(X_test)
    
    # mae_RF = mean_absolute_error(y_test, y_pred_LR)

    # XGB
    # xgb_model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    # xgb_model.fit(X_train, y_train)

    # with open('1_XGB_model.pkl', 'wb') as pkl:
    #     pickle.dump(xgb_model, pkl)

    with open('1_XGB_model.pkl', 'rb') as pkl:
        xgb_model = pickle.load(pkl)

    y_pred_XGB = xgb_model.predict(X_test)

    mae_XGB = mean_absolute_error(y_test, y_pred_XGB)

    # DecisionTreeRegressor
    # dtr_model = DecisionTreeRegressor(random_state = 0)
    # dtr_model.fit(X_train, y_train)

    # with open('1_DTR_model.pkl', 'wb') as pkl:
    #     pickle.dump(dtr_model, pkl)

    with open('1_DTR_model.pkl', 'rb') as pkl:
        dtr_model = pickle.load(pkl)

    y_pred_DTR = dtr_model.predict(X_test)

    mae_DTR = mean_absolute_error(y_test, y_pred_DTR)

    # BayesianRidge

    with open('1_BR_model.pkl', 'rb') as pkl:
        br_model = pickle.load(pkl)

    y_pred_BR = br_model.predict(X_test)

    mae_BR = mean_absolute_error(y_test, y_pred_BR)

    lst = [
    ['Linear Regression', 
        r2_score(y_test, y_pred_LR), 
        ln_model.score(X, y), 
        ln_model.score(X_train, y_train), 
        ln_model.score(X_test, y_test), 
        mae_LR],
        
        ['Random Forest', 
        0.9170, 
        0.9752, 
        1.0000, 
        0.9170, 
        0.1964],
        
        ['XGB', 
        r2_score(y_test, y_pred_XGB), 
        xgb_model.score(X, y), 
        xgb_model.score(X_train, y_train), 
        xgb_model.score(X_test, y_test), 
        mae_XGB],
        
        ['Decision Tree Regressor', 
        r2_score(y_test, y_pred_DTR), 
        dtr_model.score(X, y),
        dtr_model.score(X_train, y_train), 
        dtr_model.score(X_test, y_test), 
        mae_DTR],
        
        ['Bayesian Ridge', 
        r2_score(y_test, y_pred_BR), 
        br_model.score(X, y), 
        br_model.score(X_train, y_train), 
        br_model.score(X_test, y_test), 
        mae_BR]
    ]

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">1. USA\'s Avocados Average Prediction</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">The table above shows the result of models:</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    result_table = pd.DataFrame(lst, columns =['Model', 'R2 score', 'full R2', 'train R2', 'test R2', 'Mean absolute error'])

    st.dataframe(result_table)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">We will choose XGB since this model has the best result</p>'
    st.markdown(new_title, unsafe_allow_html=True)

elif choice == '2. Organic Avocados in California - Time Series':

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">2. Organic Avocados in California - Time Series</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.subheader("Make new prediction for the future in California")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">Next 12 months</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("horizontal_grayline.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">FACEBOOK PROPHET</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    fig1 = model.plot(forecast)
    fig1.show()
    a = add_changepoints_to_plot(fig1.gca(), model, forecast)
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Next 12 months
    df_new = forecast[['ds', 'yhat']].tail(12)
    st.table(df_new)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">Long-term prediction for the next 5 years => Consider whether to expand cultivation/production, and trading.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    fig3 = m.plot(forecast_new)
    a = add_changepoints_to_plot(fig3.gca(), m, forecast_new)
    st.pyplot(fig3)

    fig4, ax = plt.subplots()
    ax.plot(df_ts['y'], label='AveragePrice')
    ax.plot(forecast_new['yhat'], label='AveragePrice with next 60 months prediction',
    color='red')
    ax.legend()
    st.pyplot(fig4)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">Based on the above results, we can see that It\'s possible to expand the cultivation/production and trading of organic avocados.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("horizontal_grayline.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">ARIMA</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("1_Arima/1_Arima_Prediction.png")
    st.image("1_Arima/2_Arima_Prediction.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">In the next 5 years:</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("1_Arima/1_Arima_next_5_years.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">According to Arima\'s prediction, Average Price of conventional avocados in Carlifornia will fluctuate in the future.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    rmse_tables = [
        ['Facebook Prophet', 
        0.203
        ],
        
        ['ARIMA', 
        0.357]
    ]


    rmse_tables_df = pd.DataFrame(rmse_tables, columns =['Model', 'RMSE'])
    st.dataframe(rmse_tables_df)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">We will choose Facebook Prophet since this model has the best result</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    # arima_model.fit(arima_train)

    # future_forecast = arima_model.predict(n_periods=len(arima_test))

    # future_forecast = pd.DataFrame(future_forecast, index = arima_test.index, columns=['Prediction'])

    # plt.figure(figsize=(15, 8))
    # plt.plot(arima_test, label='Average Price')
    # plt.plot(future_forecast, label='Prediction')
    # plt.xticks(rotation='vertical')
    # plt.legend()
    # plt.show()
    # st.pyplot(plt)

    # plt.figure(figsize=(15, 8))
    # plt.plot(arima_organic_df, label='Average Price')
    # plt.plot(future_forecast, label='Prediction', color="Red")
    # plt.xticks(rotation='vertical')
    # plt.legend()
    # plt.show()
    # st.pyplot(plt)

    # st.write("R2 score of Arima:")
    # st.write(r2_score(arima_test, future_forecast))
    # st.write("=> R2 score is not good!!!")

    # future_3_years_forecast = arima_model.predict(n_periods=(len(arima_test) + 52*3))

    # st.markdown("In the next 5 years:")
    # future_3_years_forecast = arima_model.predict(n_periods=(len(arima_test) + 52*3))
    # datime_index = pd.date_range("2017-01-01", periods=(len(arima_test) + 52*3), freq="W")
    # future_3_years_forecast = pd.DataFrame(future_3_years_forecast, index = datime_index, columns=['Prediction'])
    # plt.figure(figsize=(15, 8))
    # plt.plot(arima_organic_df, label='Average Price')
    # plt.plot(future_3_years_forecast, label='Prediction', color="Red")
    # plt.xticks(rotation='vertical')
    # plt.legend()
    # plt.show()
    # st.pyplot(plt)
    
    # st.markdown("According to Arima's prediction, Average Price of organic avocados in Carlifornia will fluctuate in the future")


elif choice == "3. California's Conventional Avocados - Average Prediction":

    #=================================

    conventional_ca_df = regression_df.loc[(regression_df['type'] == 'conventional') & (regression_df['region'] == 'California')]

    # conventional_df_ohe = pd.get_dummies(data=conventional_ca_df, columns=['region'])

    conventional_ca_X = conventional_ca_df.drop(['Date', 'AveragePrice', 'type', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags', 'region'], axis=1)
    conventional_ca_y = conventional_ca_df['AveragePrice']

    scaler = StandardScaler()
    conventional_ca_X = scaler.fit_transform(conventional_ca_X)

    conventional_ca_X_train, conventional_ca_X_test, conventional_ca_y_train, conventional_ca_y_test = train_test_split(conventional_ca_X, conventional_ca_y,
                                                    test_size=0.3,
                                                    random_state=0)


    # Linear Regression
    with open('3_LR_model.pkl', 'rb') as pkl:
        ln_model_3 = pickle.load(pkl)

    conventional_ca_y_pred_LR = ln_model_3.predict(conventional_ca_X_test)

    conventional_ca_mae_LR = mean_absolute_error(conventional_ca_y_test, conventional_ca_y_pred_LR)

    # Random Forest
    # conventional_ca_pipe_RF = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=500, 
    #                                 min_samples_split=2, 
    #                                 min_samples_leaf=1, 
    #                                 max_features='sqrt', 
    #                                 max_depth=None, 
    #                                 bootstrap=False))])
    # conventional_ca_pipe_RF.fit(conventional_ca_X_train, conventional_ca_y_train)
    # conventional_ca_y_pred_RF = conventional_ca_pipe_RF.predict(conventional_ca_X_test)

    # conventional_ca_mae_RF = mean_absolute_error(conventional_ca_y_test, conventional_ca_y_pred_RF)


    # XGB
    with open('3_XGB_model.pkl', 'rb') as pkl:
        xgb_model_3 = pickle.load(pkl)

    conventional_ca_y_pred_XGB = xgb_model_3.predict(conventional_ca_X_test)

    conventional_ca_mae_XGB = mean_absolute_error(conventional_ca_y_test, conventional_ca_y_pred_XGB)


    # DecisionTreeRegressor
    with open('3_DTR_model.pkl', 'rb') as pkl:
        dtr_model_3 = pickle.load(pkl)

    conventional_ca_y_pred_DTR = dtr_model_3.predict(conventional_ca_X_test)

    conventional_ca_mae_DTR = mean_absolute_error(conventional_ca_y_test, conventional_ca_y_pred_DTR)


    # BayesianRidge
    with open('3_BR_model.pkl', 'rb') as pkl:
        br_model_3 = pickle.load(pkl)

    conventional_ca_y_pred_BR = br_model_3.predict(conventional_ca_X_test)

    conventional_ca_mae_BR = mean_absolute_error(conventional_ca_y_test, conventional_ca_y_pred_BR)

    #=================================

    conventional_ca_lst = [
    ['Linear Regression', 
        r2_score(conventional_ca_y_test, conventional_ca_y_pred_LR), 
        ln_model_3.score(conventional_ca_X, conventional_ca_y), 
        ln_model_3.score(conventional_ca_X_train, conventional_ca_y_train), 
        ln_model_3.score(conventional_ca_X_test, conventional_ca_y_test), 
        conventional_ca_mae_LR],
        
        ['Random Forest', 
        0.9116, 
        0.9730, 
        1.0000, 
        0.9116, 
        0.0535],
        
        ['XGB', 
        r2_score(conventional_ca_y_test, conventional_ca_y_pred_XGB), 
        xgb_model_3.score(conventional_ca_X, conventional_ca_y), 
        xgb_model_3.score(conventional_ca_X_train, conventional_ca_y_train), 
        xgb_model_3.score(conventional_ca_X_test, conventional_ca_y_test), 
        conventional_ca_mae_XGB],
        
        ['Decision Tree Regressor', 
        r2_score(conventional_ca_y_test, conventional_ca_y_pred_DTR), 
        dtr_model_3.score(conventional_ca_X, conventional_ca_y),
        dtr_model_3.score(conventional_ca_X_train, conventional_ca_y_train), 
        dtr_model_3.score(conventional_ca_X_test, conventional_ca_y_test), 
        conventional_ca_mae_DTR],
        
        ['Bayesian Ridge', 
        r2_score(conventional_ca_y_test, conventional_ca_y_pred_BR), 
        br_model_3.score(conventional_ca_X, conventional_ca_y), 
        br_model_3.score(conventional_ca_X_train, conventional_ca_y_train), 
        br_model_3.score(conventional_ca_X_test, conventional_ca_y_test), 
        conventional_ca_mae_BR]
    ]

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">3. California\'s Conventional Avocados - Average Prediction</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">The table above shows the result of models:</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    conventional_ca_result_table = pd.DataFrame(conventional_ca_lst, columns =['Model', 'R2 score', 'full R2', 'train R2', 'test R2', 'Mean absolute error'])

    st.dataframe(conventional_ca_result_table)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">We will choose XGB since this model has the best result</p>'
    st.markdown(new_title, unsafe_allow_html=True)


elif choice == "4. California's Conventional Avocados - Time Series":
    # Filter Avocados - California
    # Make new dataframe from original dataframe: data

    conventional_df_ca = data[data['region'] == 'California']
    conventional_df_ca['Date'] = pd.to_datetime(conventional_df_ca['Date'])
    conventional_df_ca = conventional_df_ca[conventional_df_ca['type'] == 'conventional']

    conventional_agg = {'AveragePrice': 'mean'}
    conventional_df_ca_gr = conventional_df_ca.groupby(conventional_df_ca['Date']).aggregate(conventional_agg).reset_index()
    conventional_df_ca_gr.head()

    conventional_df_ts = pd.DataFrame() 
    conventional_df_ts['ds'] = pd.to_datetime(conventional_df_ca_gr['Date']) 
    conventional_df_ts['y'] = conventional_df_ca_gr['AveragePrice'] 
    conventional_df_ts.head()

    # Train/Test Prophet
    # create test dataset, remove last 10 months
    conventional_train = conventional_df_ts.drop(conventional_df_ts.index[-10:])
    conventional_test = conventional_df_ts.drop(conventional_df_ts.index[0:-10])

    # Build model
    conventional_model = Prophet(yearly_seasonality=True, \
                daily_seasonality=False, weekly_seasonality=False) 
    conventional_model.fit(conventional_train)

    # 10 month in test and 12 month to predict new values
    conventional_months = pd.date_range('2017-01-01','2019-12-01', 
                freq='MS').strftime("%Y-%m-%d").tolist()    
    conventional_future = pd.DataFrame(conventional_months)
    conventional_future.columns = ['ds']
    conventional_future['ds'] = pd.to_datetime(conventional_future['ds'])

    # Use the model to make a forecast
    conventional_forecast = conventional_model.predict(conventional_future)

    # calculate MAE/RMSE between expected and predicted values
    conventional_y_test = conventional_test['y'].values
    conventional_y_pred = conventional_forecast['yhat'].values[:10]
    conventional_mae_p = mean_absolute_error(conventional_y_test, conventional_y_pred)
    # print('MAE: %.3f' % mae_p)
    conventional_rmse_p = sqrt(mean_squared_error(conventional_y_test, conventional_y_pred))

    # visualization
    conventional_y_test_value = pd.DataFrame(conventional_y_test, index = pd.to_datetime(conventional_test['ds']),columns=['Actual'])
    conventional_y_pred_value = pd.DataFrame(conventional_y_pred, index = pd.to_datetime(conventional_test['ds']),columns=['Prediction'])

    # Long-term prediction for the next 5 years => Consider whether to expand cultivation/production, and trading
    conventional_m = Prophet(yearly_seasonality=True, \
                daily_seasonality=False, weekly_seasonality=False) 
    conventional_m.fit(conventional_df_ts)
    conventional_future_new = conventional_m.make_future_dataframe(periods=12*5, freq='M') # next 5 years
    conventional_forecast_new = conventional_m.predict(conventional_future_new)

    # Arima
    # conventional_arima_organic_df = data.copy(deep=True)
    # conventional_arima_organic_df = conventional_arima_organic_df.loc[(conventional_arima_organic_df['type'] == 'conventional') 
    # & (conventional_arima_organic_df['region'] == 'California')]
    # conventional_arima_organic_df['Date'] = pd.to_datetime(conventional_arima_organic_df['Date'])
    # conventional_arima_organic_df['value'] = conventional_arima_organic_df['AveragePrice']
    # conventional_arima_organic_df = conventional_arima_organic_df[['Date', 'value']]
    # conventional_arima_organic_df.columns = ['date', 'value']
    # conventional_arima_organic_df = conventional_arima_organic_df.sort_values(by=['date'])
    # conventional_arima_organic_df.reset_index(drop=True, inplace=True)
    # conventional_arima_organic_df.set_index('date', inplace=True)
    # conventional_arima_organic_df.info()

    # conventional_arima_model = auto_arima(conventional_arima_organic_df, start_p=2, d=1, 
    #                         start_q=2, max_p=5,
    #                         max_q=5, start_P=0, D=1, 
    #                         start_Q=0, max_P=5,
    #                         m=52, seasonal=True, 
    #                         error_action='ignore', trace = True, 
    #                         supress_warnings=True, 
    #                         stepwise = True)

    # # Serialize with Pickle
    # with open('conventional_arima.pkl', 'wb') as pkl:
    #     pickle.dump(conventional_arima_model, pkl)


    # with open('conventional_arima.pkl', 'rb') as pkl:
    #     conventional_arima_model = pickle.load(pkl)


    # conventional_arima_train = conventional_arima_organic_df[conventional_arima_organic_df.index.year < int(2017)]
    # conventional_arima_test = conventional_arima_organic_df[conventional_arima_organic_df.index.year >= int(2017)]


    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">4. California\'s Conventional Avocados - Time Series</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.subheader("Make new prediction for the future in California")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">Next 12 months</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("horizontal_grayline.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">FACEBOOK PROPHET</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    fig1 = conventional_model.plot(forecast)
    fig1.show()
    a = add_changepoints_to_plot(fig1.gca(), conventional_model, conventional_forecast)
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Next 12 months
    conventional_df_new = conventional_forecast[['ds', 'yhat']].tail(12)
    st.table(conventional_df_new)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">Long-term prediction for the next 5 years => Consider whether to expand cultivation/production, and trading.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    fig3 = conventional_m.plot(forecast_new)
    a = add_changepoints_to_plot(fig3.gca(), conventional_m, forecast_new)
    st.pyplot(fig3)

    fig4, ax = plt.subplots()
    ax.plot(conventional_df_ts['y'], label='AveragePrice')
    ax.plot(conventional_forecast_new['yhat'], label='AveragePrice with next 60 months prediction',
    color='red')
    ax.legend()
    st.pyplot(fig4)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">Based on the above results, we can see that It\'s possible to expand the cultivation/production and trading of conventional avocados.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("horizontal_grayline.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">ARIMA</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("3_Arima/1_Arima_Prediction.png")
    st.image("3_Arima/2_Arima_Prediction.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">In the next 5 years:</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("3_Arima/1_Arima_next_5_years.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">According to Arima\'s prediction, Average Price of conventional avocados in Carlifornia will fluctuate in the future.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    rmse_tables = [
        ['Facebook Prophet', 
        0.319
        ],
        
        ['0.4752', 
        0.35722]
    ]

    rmse_tables_df = pd.DataFrame(rmse_tables, columns =['Model', 'RMSE'])
    st.dataframe(rmse_tables_df)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">We will choose Facebook Prophet since this model has the best result</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    # conventional_arima_model.fit(conventional_arima_train)

    # conventional_future_forecast = conventional_arima_model.predict(n_periods=len(conventional_arima_test))

    # conventional_future_forecast = pd.DataFrame(conventional_future_forecast, index = conventional_arima_test.index, columns=['Prediction'])

    # plt.figure(figsize=(15, 8))
    # plt.plot(conventional_arima_test, label='Average Price')
    # plt.plot(conventional_future_forecast, label='Prediction')
    # plt.xticks(rotation='vertical')
    # plt.legend()
    # plt.show()
    # st.pyplot(plt)

    # plt.figure(figsize=(15, 8))
    # plt.plot(conventional_arima_organic_df, label='Average Price')
    # plt.plot(conventional_future_forecast, label='Prediction', color="Red")
    # plt.xticks(rotation='vertical')
    # plt.legend()
    # plt.show()
    # st.pyplot(plt)

    # st.write("R2 score of Arima:")
    # st.write(r2_score(conventional_arima_test, conventional_future_forecast))
    # st.write("=> R2 score is not good!!!")

    # conventional_future_3_years_forecast = conventional_arima_model.predict(n_periods=(len(conventional_arima_test) + 52*3))

    # st.markdown("In the next 5 years:")
    # conventional_future_3_years_forecast = arima_model.predict(n_periods=(len(conventional_arima_test) + 52*3))
    # conventional_datime_index = pd.date_range("2017-01-01", periods=(len(conventional_arima_test) + 52*3), freq="W")
    # conventional_future_3_years_forecast = pd.DataFrame(conventional_future_3_years_forecast, index = conventional_datime_index, columns=['Prediction'])
    # plt.figure(figsize=(15, 8))
    # plt.plot(conventional_arima_organic_df, label='Average Price')
    # plt.plot(conventional_future_3_years_forecast, label='Prediction', color="Red")
    # plt.xticks(rotation='vertical')
    # plt.legend()
    # plt.show()
    # st.pyplot(plt)
    
    # st.markdown("According to Arima's prediction, Average Price of conventional avocados in Carlifornia will increase in the future")
elif choice == "5. Boise's Avocados trend":

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">5. Boise\'s Avocados trend</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    def find_trend_of_region(region, avocado_type):
        st.write("REGION: ", region)
        st.write("type: ", avocado_type)
        data_df = sub_df.loc[(sub_df['type'] == avocado_type) & (sub_df['region'] == region)]
        data_df = data_df.sort_values(by=['Date'])
        data_df.reset_index(drop=True, inplace=True)
        
        data_df = data_df[['Date', 'AveragePrice']]
        data_df.columns = ['ds', 'y']
        
        # Build model
        m = Prophet(yearly_seasonality=True,
            daily_seasonality=False, weekly_seasonality=False)
        
        m.fit(data_df)
        future = m.make_future_dataframe(periods=12*5, freq='M') # next 5 years
        forecast = m.predict(future)
        
        
        fig = m.plot(forecast, figsize=(10, 12))
        fig.show()
        a = add_changepoints_to_plot(fig.gca(), m, forecast)
        st.pyplot(fig)

        fig1 = model.plot_components(forecast)
        fig1.show()
        st.pyplot(fig1)

        st.image("horizontal_grayline.png")

    find_trend_of_region('Boise', 'organic')
    find_trend_of_region('Boise', 'conventional')


    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">ARIMA</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("Boise_Arima/1_Arima_Prediction.png")
    st.image("Boise_Arima/2_Arima_Prediction.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">In the next 5 years:</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.image("Boise_Arima/Boise_Arima_next_5_years.png")

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">According to Arima\'s prediction, Average Price of conventional avocados in Carlifornia will fluctuate in the future.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    rmse_tables = [
        ['Facebook Prophet', 
        0.246
        ],
        
        ['ARIMA', 
        0.588]
    ]


    rmse_tables_df = pd.DataFrame(rmse_tables, columns =['Model', 'RMSE'])
    st.dataframe(rmse_tables_df)

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 25px;">We will choose Facebook Prophet since this model has the best result</p>'
    st.markdown(new_title, unsafe_allow_html=True)

elif choice == "6. Find the trend of regions in the future":

    new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">6. Find the trend of regions in the future</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    def find_trend_of_region(region, avocado_type):
        st.write("REGION: ", region)
        st.write("type: ", avocado_type)
        data_df = sub_df.loc[(sub_df['type'] == avocado_type) & (sub_df['region'] == region)]
        data_df = data_df.sort_values(by=['Date'])
        data_df.reset_index(drop=True, inplace=True)
        
        data_df = data_df[['Date', 'AveragePrice']]
        data_df.columns = ['ds', 'y']
        
        # Build model
        m = Prophet(yearly_seasonality=True,
            daily_seasonality=False, weekly_seasonality=False)
        
        m.fit(data_df)
        future = m.make_future_dataframe(periods=12*5, freq='M') # next 5 years
        forecast = m.predict(future)
        
        
        fig = m.plot(forecast, figsize=(10, 12))
        fig.show()
        a = add_changepoints_to_plot(fig.gca(), m, forecast)
        st.pyplot(fig)

        fig1 = model.plot_components(forecast)
        fig1.show()
        st.pyplot(fig1)

        st.image("horizontal_grayline.png")

    
    regions_list = sub_df.region.unique()
    regions_list = regions_list[regions_list != 'TotalUS']

    selected_region = st.selectbox('Please select a region', regions_list)

    types_list = ['organic', 'conventional', 'both']

    selected_type = st.selectbox('Please select a type', types_list)

    if st.button('Trend of this region in the future'):
        if selected_type == 'both':
            find_trend_of_region(selected_region, 'organic')
            find_trend_of_region(selected_region, 'conventional')
        else:
            find_trend_of_region(selected_region, selected_type)
        