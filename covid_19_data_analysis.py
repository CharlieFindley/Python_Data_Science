#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 13:31:25 2025

@author: charliefindley
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 




covid19_df = pd.read_csv('worldometer_coronavirus_daily_data.csv')
covid19_df.head(10)
covid19_df.isnull().sum()
# we are only interested in the uk covid-19 data for the momemnt 
ukcovid19_data = covid19_df.copy()
covid19_df['country'].unique()
ukcovid19_data = ukcovid19_data[ukcovid19_data['country'] == 'UK']
# in general if i want to check whether a particular value is in a column we use an in clause
"UK" in covid19_df["country"].values
"Somalia" in covid19_df['country'].values
'Turjikistan' in covid19_df['country'].values
print(ukcovid19_data)
ukcovid19_data.head(10)
ukcovid19_data.isnull().sum()
# any data cleaning we will simplify by setting any null values straight to 0
ukcovid19_data['daily_new_cases'] = ukcovid19_data['daily_new_cases'].fillna(0)
ukcovid19_data['daily_new_deaths'] = ukcovid19_data['daily_new_deaths'].fillna(0)
ukcovid19_data.isnull().sum()
ukcovid19_data.columns
ukcovid19_data['date'] = pd.to_datetime(ukcovid19_data['date'])
ukcovid19_data = ukcovid19_data.sort_values('date')
plt.figure(figsize = (12,6))
plt.plot(ukcovid19_data['date'], ukcovid19_data['daily_new_cases'])
plt.xlabel('Date')
plt.ylabel('Daily New Cases')
plt.title('Daily New Covid-19 Cases in the UK')

plt.grid(True, linestyle = '--', alpha = 0.4)
# highly recommended is a seven day rolling average because Covid 19 data is noisy 
df = ukcovid19_data.copy()
ukcovid19_data['new_cases_7day_avg'] = ukcovid19_data['daily_new_cases'].rolling(window = 7).mean()
# a rolling average is useful because covid-19 data is very noisy because we 
# have weekend dips, weekday rises, reporting delays and random spikes
# a 7 day rolling average makes the cue much smoother 
# so a seven day rolling average shows the true trend 
plt.figure(figsize = (12,6))
plt.plot(ukcovid19_data['date'], ukcovid19_data['new_cases_7day_avg'], color = 'blue', label = 'Daily Rolling Average')
plt.plot(df['date'], df['daily_new_cases'], color = 'red', label = 'raw daily new cases')
plt.xlabel('Date')
plt.ylabel('Daily New Cases')
plt.title('UK COVID-19 CASES WITH 7-DAY MOVING AVERAGE')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show()
# we also use a 7-day rolling average when considering daily new deaths 
df['new_deaths_7day_avg'] = df['daily_new_deaths'].rolling(window = 7).mean()
df['new_deaths_7day_avg'].head(100)
# now we are going to do a lineplot of daily new cases and daily new deaths using a 7 day rolling average to make the curves smoother and less nosiy
df['new_cases_7day_avg'] = df['daily_new_cases'].rolling(window = 7).mean()
plt.figure(figsize = (12,6))
plt.plot(df['date'],df['new_cases_7day_avg'], color = 'blue', label = 'daily new cases')
plt.plot(df['date'], df['new_deaths_7day_avg'], color = 'red', label = 'daily new deaths')
plt.xlabel('Date')
plt.ylabel('Daily new cases / deaths')
plt.title('UK daily new deaths and cases using a 7day rolling average')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show() # from this we can see that deaths were only a tiny fraction of total cases 
# perhaps the lockdown measures were too extreme, # lets zoom in on only the deaths to see if the measures were justified
plt.figure(figsize = (12,6))
plt.plot(df['date'], df['new_deaths_7day_avg'], color = 'red', label = 'daily new deaths')
plt.xlabel('Date')
plt.ylabel('Daily new cases / deaths')
plt.title('UK daily new deaths and cases using a 7day rolling average')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show() 
df['new_deaths_7day_avg'].max() # 1310 was the maximum number of daily deaths when using a 7 day rolling average
# lets compare this to the maximal daily deaths for the raw data
df['daily_new_deaths'].max() # 1387, so there is not a huge difference between the values 
# in depth analysis of new cases vs deaths 
# when we plotted deaths vs new cases we saw that deaths was much lower than cases
# so we could scale the rolling deaths to compare them and see the lag effect 
# high deaths tended to lag slightly after periods of high cases, so called waves 
scale = 20
plt.figure(figsize = (10,6))
plt.plot(df["date"], df["new_cases_7day_avg"], label="Cases (7-day avg)")
plt.plot(df["date"], df["new_deaths_7day_avg"] * scale, 
         label=f"Deaths (7-day avg) ×{scale}", color="red")
plt.xlabel('Date')
plt.ylabel("Scaled Counts")
plt.title('Cases vs Deaths (Scaled) - Visual Lag Comparison')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show()
# from this we can see that the peak death rate slightly trailed the peak infection rate 
# we now want to find the lag with the highest correlation 
# we could isolate the first wave which is between november 2019 and june 2020 
df_first_wave = df.copy()
start = "2019-11-01"
end = "2020-06-01"
df_first_wave = df[(df["date"] >= start) & (df["date"] <= end)]
df_first_wave["new_cases_7day_avg"].head(10)
df_first_wave["new_cases_7day_avg"] = df_first_wave["new_cases_7day_avg"].fillna(0)
df_first_wave["new_cases_7day_avg"].head(10)
df_first_wave["new_deaths_7day_avg"].head(100)
df_first_wave["new_deaths_7day_avg"] = df_first_wave["new_deaths_7day_avg"].fillna(0)
df_first_wave["new_deaths_7day_avg"].head(30)
# this plot should only show the first wave and we are going to further scale deaths so that we can see the lag more easily 
scale = 4
plt.figure(figsize = (10,6))
plt.plot(df_first_wave["date"], df_first_wave["new_cases_7day_avg"], label="Cases (7-day avg)")
plt.plot(df_first_wave["date"], df_first_wave["new_deaths_7day_avg"] * scale, 
         label=f"Deaths (7-day avg) ×{scale}", color="red")
plt.xlabel('Date')
plt.ylabel("Scaled Counts")
plt.title('Cases vs Deaths (Scaled) - Visual Lag Comparison')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show()
# ignore this we will just use the initial plot to find the lag 
scale = 20
plt.figure(figsize = (10,6))
plt.plot(df["date"], df["new_cases_7day_avg"], label="Cases (7-day avg)")
plt.plot(df["date"], df["new_deaths_7day_avg"] * scale, 
         label=f"Deaths (7-day avg) ×{scale}", color="red")
plt.xlabel('Date')
plt.ylabel("Scaled Counts")
plt.title('Cases vs Deaths (Scaled) - Visual Lag Comparison')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show()
# the following loop tries each possible lag and measures how similar the curves are
import numpy as np 
max_lag = 30
correlations = []
for lag in range(0,max_lag+1):
    shifted_deaths = df["new_deaths_7day_avg"].shift(-lag) # this moves the death data backwards by lag days
    corr = df["new_cases_7day_avg"].corr(shifted_deaths)
    correlations.append((lag,corr))
# we want to find the lag with the maximal correlation 
best_lag, best_corr = max(correlations, key = lambda x:x[1])
print(best_lag) # from this we get that the best lag is 15 days, so deaths trail cases by 15 days
# now we want to plt cases vs shifted_deaths to visualise the alignment 
scale_factor = 40
shifted_deaths = df["new_deaths_7day_avg"].shift(-best_lag)
plt.figure(figsize=(12,6))
plt.plot(df["date"], df["new_cases_7day_avg"],              label="Cases (7-day avg)")
plt.plot(df["date"], shifted_deaths * scale_factor,
         label=f"Deaths (7-day avg, shifted {best_lag} days, ×{scale_factor})",
         color="red")
plt.xlabel("Date")
plt.ylabel("Scaled Counts")
plt.title(f"Cases vs Deaths (Deaths shifted by {best_lag} days)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
df = df.rename(columns = {"new_cases_7day_avg":"cases_7d"})
df= df.rename(columns = {"new_deaths_7day_avg":"deaths_7d"})
# now we want to investigate the case to fatality ratio (CFR) 
df["cfr"] = df["cumulative_total_deaths"] / df["cumulative_total_cases"]
df["cfr"].head(100)
# first we will do a simple line plot to see how the cfr changes over time 
plt.figure(figsize = (10,6))
plt.plot(df["date"], df["cfr"])
plt.xlabel("Date")
plt.ylabel("CFR")
plt.title("CFR over Time- UK COVID-19")
plt.grid(True, linestyle ="--", alpha = 0.4)
plt.show() # early on cfr is very unstgable(low cases leads to noise)
# then it stabilises and trends downwards as testing increases 
# we can create a smooth cfr to reduce noisy fluctuations, particularly early on 
df["cfr_7d"] = (
    df["daily_new_deaths"].rolling(7).sum()/
    df["daily_new_cases"].rolling(7).sum()
)
df["cfr_7d"].head(100)
plt.figure(figsize = (10,6))
plt.plot(df["date"], df["cfr"], label = "cumulative_cfr")
plt.plot(df["date"],df["cfr_7d"], label = "cfr 7day avg")
plt.xlabel("Date")
plt.ylabel("CFR")
plt.title("CFR over Time- UK COVID-19")
plt.legend() # need to use plt.legend() to call the labels
plt.grid(True, linestyle ="--", alpha = 0.4)
plt.show()
# lets map the daily cfr which is extremely noisy, smoothing is obviously necessary
df["daily_cfr"] = df["daily_new_deaths"] / df["daily_new_cases"]

plt.figure(figsize = (8,6))
plt.plot(df["date"],df["daily_cfr"],alpha = 0.3, label = "Daily CFR")
plt.plot(df["date"], df["cfr_7d"], alpha = 0.3, label = "7day cfr", linewidth =2)
plt.legend()
plt.title("Daily CFR vs Smoothed CFR")
plt.grid(True, linestyle ='--')
plt.show()
# next we are going to plot the CFR for Wave1 vs Wave2
# note that Wave 1 : Dec 2019 - Jun 2020
# Wave 2 : Sep 2020: Mar 2021
wave1 = df[(df["date"] >= "2019-12-01") & ((df["date"] <= "2020-06-01"))]
wave2 = df[(df["date"] >= "2020-09-01") & (df["date"] <= "2021-03-01")]
plt.figure(figsize = (12,6))
plt.plot(wave1["date"], wave1["cfr_7d"], label = "Wave1 CFR")
plt.plot(wave2["date"], wave2["cfr_7d"], label = "Wave 2 CFR")
plt.legend()
plt.title("CFR comparison between wave 1 and wave 2")
plt.grid(True)
plt.show()
# we also want to make a scatter plot to see if there is a relationship between the number of active cases and CFR
# do overwhelmed hospitals lead to excess deaths 
# we will first do this for only the first wave
plt.figure(figsize = (8,6))
plt.scatter(wave1["active_cases"], wave1["cfr_7d"])
plt.xlabel("Active Cases")
plt.ylabel("cfr")
plt.title("Relationship between cfr and active cases in wave1")
plt.grid(True)
plt.show()
# we need to find a ml algorithm which can handle non-linear patterns well
# we can use a random forest regressor 
# we want to buld  a full time-series linear regression model that predicts
# tomorrows daily new cases using data from previous days 
# to predict tomorrows cases the model needs yesterdays cases
# the 7day average, cases from last week , yesterdays deaths(optional but useful)
df["cases_yesterday"] = df["daily_new_cases"].shift(1)
df["cases_7d_avg"] = df["daily_new_cases"].rolling(7).mean()
df["cases_last_week"] = df["daily_new_cases"].shift(7)
df["deaths_yesterday"] = df["daily_new_deaths"].shift(1)
df["active_yesterday"] = df["active_cases"].shift(1)
# we want to create the target variable which is tomorrows cases
df["cases_tomorrow"] = df["daily_new_cases"].shift(-1)
df_model = df.dropna()
features = ["cases_yesterday","cases_7d_avg","cases_last_week",
            "deaths_yesterday","active_yesterday"] 
X = df_model[features]
y = df_model["cases_tomorrow"]
train_size = int(len(df_model) * 0.8)
# note that we must not shuffle time series data
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 300, random_state = 42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# now we test the model using the root mean squared error metric
from sklearn.metrics import mean_squared_error 
rmse = np.sqrt(mean_squared_error(y_test,predictions))
# the lower the value of the rmse the better the model 
print(rmse) # rmse of 4.8 seems pretty good
# now we will plot the actual values vs the predicted values 
# this will allow us to visualise how good the distribution actually is 
plt.figure(figsize = (12,6))
plt.plot(y_test.values, label = "Actual cases", linewidth = 2)
plt.plot(predictions, label = 'Predicted Cases', linewidth =2)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Cases')
plt.show()
# when random forest regression is the best choice of model 
# the data has strong non-linear patters, e.g. covid waves
# we have good lag features
# the future depends mostly on the recent past 
# When random forest is not the best choice 
# if we want to predict far into the future then random forest tends to break down 
# when seasonality or long-range structure is important 
# so we use random forest when the future case depends mostly on the recent past 
# need to take an in detail look at decision tree regression and 
# random forest regression 
# also may want to look at gradient boosting regressors 
# lets use an SGBoost Time-Series Model to Predict Tomorrow's cases
# target = tomorrows daily_new_cases
df["cases_1d"] = df["daily_new_cases"].shift(1)
df["cases_2d"] = df["daily_new_cases"].shift(2)
df["cases_3d"] = df["daily_new_cases"].shift(3)
df["cases_7d"] = df["daily_new_cases"].shift(7)
df["cases_14d"] = df["daily_new_cases"].shift(14)
df["cases_7d_avg"] = df["daily_new_cases"].rolling(7).mean()
df["cases_14d_avg"] = df["daily_new_cases"].rolling(14).mean()
df['target'] = df["daily_new_cases"].shift(-1)
df = df.dropna()
features = ["cases_1d","cases_2d","cases_3d",
            "cases_7d", "cases_14d","cases_7d_avg",
            "cases_14d_avg"]
X = df[features]
y = df['target']
split = int(len(df)*0.8)
X_train, X_test =X[:split],X[split:]
y_train,y_test = y[:split],y[split:]

from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators = 300,
    learning_rate = 0.05,
    max_depth = 5, 
    subsample = 0.8, 
    colsample_bytree = 0.8, 
    objective="reg:squarederror"
        )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# finally we are going to plot predictions vs real values 
plt.figure(figsize = (12,5))
plt.plot(y_test.values, label = 'Actual')
plt.plot(y_pred, label = 'Predicted')
plt.legend()
plt.title("XGBoost - Predicted Daily New Cases")
plt.show()
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse) # this model has a much larger RMSE and only fits well in certain domains 
# the first model was more appropriate for predicting the daily new cases tomorrow using the Random Forest Regressor 
