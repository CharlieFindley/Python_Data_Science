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
# now we are going to do some more growth rate analysis 
# first we are going to compute and visualise the covid-19 case growth rate
df["daily_new_cases"].isna().sum()
df["growth_rate"] = (df["daily_new_cases"] - df["daily_new_cases"].shift(1)) / df["daily_new_cases"].shift(1)
df["growth_rate"].isna().sum() # we can see that there is 1 na value
# the first value of the growth_rate column must be na so we should expect this value to be 1
df["growth_rate"].head() # as we expected the first value in this column has been stored as nan 
df["growth_rate"] = df["growth_rate"].fillna(0) # cleaned the data by replacing the nan with 0
# as usual though, covid19 data is very noisy so we plot a 7 day rolling average 
df["growth_rate_7d"] = df["growth_rate"].rolling(7).mean()
df["growth_rate_7d"].head()
df["growth_rate_7d"] = df["growth_rate_7d"].fillna(0) # the 7 day rolling average is useful as it smooths as the noise created by weekend / weekday effects
# the next task is to plot the growth rate 
plt.figure(figsize = (12,6))
plt.plot(df["date"], df["growth_rate"], color = "red", label = "Daily Growth Rate")
plt.tight_layout()
plt.ylim(-2,5) # this limits the y axis to -5, 5
plt.plot(df["date"],df["growth_rate_7d"], color = "blue", label = "7d Growth Rate")
plt. axhline(0, color = "black", linestyle = '--', linewidth = 1)
# note that this draws a horizontal line across the enire plot at y = 0 
# this is useful as it shows us when cases are growing vs when cases are shrinking determined by whether the y value is positive or negative 
plt.xlabel("Date")
plt.ylabel("Growth Rate")
plt.title("Covid-19 Daily Case Growth Rate")
plt.legend()
plt.grid(True, linestyle = "--", alpha = 0.3)
plt.show()
# finally we can compute a powerful epidemiological measure which is the doubling time
# the doubling time is the amount of time it takes for the number of cases to double 
# if the doubling time is negative, this tells us the number of days it takes for the number of cases to halve
# this negative doubling time would be useful when considering case drop off due to measures such as lockdowns or vaccines
df["doubling_time"] = np.log(2) / np.log(1+ df["growth_rate_7d"])
plt.figure(figsize = (10,6))
plt.plot(df["date"], df["doubling_time"], label = "Doubling Time")
plt.axhline(0, color = 'black', linewidth = 1)
plt.title('Covid-19 Case Doubling Time (UK)')
plt.ylabel("Days")
plt.grid(True, linestyle = '--', alpha = 0.3)
plt.legend()
plt.show()
# when we see this plot we can see that extreme values make the plot unreadable 
# so we need to limit the values that the y axis can take 
plt.figure(figsize = (10,6))
plt.plot(df["date"], df["doubling_time"], label = "Doubling Time")
plt.axhline(0, color = 'black', linewidth = 1)
plt.title('Covid-19 Case Doubling Time (UK)')
plt.ylabel("Days")
plt.ylim(-100,100) # we add this so that the plot becomes readable
plt.grid(True, linestyle = '--', alpha = 0.3)
plt.legend()
plt.show()
# the graph for doubling time is extremely volatile 
# there are several reasons for this , for example the doubling time is 
# extremely sensitive to small changes in the growth rate and when 
# the growth rate is near zero the doubling time blows up which leads to extreme values
# for example when the growth rate is approximately zero we get a near infinite doubling value
df["doubling_time"].nlargest(10)
# these infinite values mean that we need to clean the dataset
df["doubling_time"] = df["doubling_time"].replace([np.inf, -np.inf], np.nan)
df["Rt"] = df["Rt"].fillna(0)
# for example in the df[doubling] column we have 6 values where the doubling time has evaluated at infinity
# now we can consider some epidemiological metrics 
# note that the current dataset is missing testing data so we may need to find another dataset to find the positivity rate
# once we have estimated the reproduction rate we can look for a more complete dataset which will allow us to compute other metrics 
# note that the reproductive rate is the average number of people that one infected person will infect
# we are going to have a serial interval be the average time between infections
# for covid-19 this is usually between 4-6 days so we will simplify and take the series interval to be 5
series_interval = 5
df["Rt"] = (1 + df["growth_rate_7d"]) ** series_interval
df["Rt"].nlargest(10) # for this dataset we do not have infinite values 
# this means that we do not need to clean the column by replacing values which have evaluated to infinity 
df["Rt"].isna().sum() # we also have no na values so this column is already clean
# next we can plot the value of the reproductive time to see how it changes over time 
plt.figure(figsize = (10,5))
plt.plot(df["date"], df["Rt"], label = "Estimated_Rt")
plt.axhline(1, colo = 'red', linestyle = '--', label = 'Rt = 1 threshold')
plt.legend()
plt.title("Effective Reprdoction Number Rt (Estimated From Case Growth)")
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show() # once again we need to scale the y axis so that the graph is readable 

plt.figure(figsize = (10,5))
plt.plot(df["date"], df["Rt"], label = "Estimated_Rt")
plt.axhline(1, color = 'red', linestyle = '--', label = 'Rt = 1 threshold')
plt.legend()
plt.title("Effective Reprdoction Number Rt (Estimated From Case Growth)")
plt.ylim(0,10)
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show() 
# reproductive numbers and doubling times which are extremely high are not that useful to me
# i may as well use the clip method to clean the data which will make all plots more readable in the future
df["Rt"] = df["Rt"].clip(upper = 10) # this sets all values above 10 to 10
# now lets see the original plot again after we have clipped the Rt values
plt.figure(figsize = (10,5))
plt.plot(df["date"], df["Rt"], label = "Estimated_Rt")
plt.axhline(1, colo = 'red', linestyle = '--', label = 'Rt = 1 threshold')
plt.legend()
plt.title("Effective Reprdoction Number Rt (Estimated From Case Growth)")
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show()
# lets see if we get an even better plot if we limit the y value further, but we will not clip the data in this case
plt.figure(figsize = (10,5))
plt.plot(df["date"], df["Rt"], label = "Estimated_Rt")
plt.axhline(1, color = 'red', linestyle = '--', label = 'Rt = 1 threshold')
plt.legend()
plt.title("Effective Reprdoction Number Rt (Estimated From Case Growth)")
plt.ylim(0,5)
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show()
# and even further still 
plt.figure(figsize = (10,5))
plt.plot(df["date"], df["Rt"], label = "Estimated_Rt")
plt.axhline(1, color = 'red', linestyle = '--', label = 'Rt = 1 threshold')
plt.legend()
plt.title("Effective Reprdoction Number Rt (Estimated From Case Growth)")
plt.ylim(0,3)
plt.grid(True, linestyle = '--', alpha = 0.4)
plt.show()
# we may as well also clean the doubling_time column in case we use it again
df["doubling_time"].nlargest(10) # we can see that we have really large doubling t
(df["doubling_time"] > 250).sum()
# from the looks of the data it appears as though a maximum limit for the doubling time of 250 seems appropriate
df["doubling_time"] = df["doubling_time"].clip(upper = 250)
# we also have another method for estimating the reproductive number
# since the series interval is five days we can take daily new cases at day t/ daily new cases 5 days earlier
df["Rt_lag"] = df["cases_7d_avg"] / df["cases_7d_avg"].shift(5)
# now we will just clean the data 
df["Rt_lag"] = df["Rt_lag"].replace([np.inf, -np.inf], np.nan)
df["Rt_lag"] = df["Rt_lag"].fillna(0)
df["Rt_lag"].nlargest(10) # none of the values for this data seem to be too exteme so we will attempt to plot this without using the clip method 
plt.figure(figsize = (10,5))
plt.plot(df["date"], df["Rt_lag"], label = "Rt (Lag Methdd)")
plt.axhline(1, color = "red", linestyle = '--', linewidth = 1)
plt.xlabel('date')
plt.ylabel('Rt')
plt.legend()
plt.grid(True)
plt.show()
# now lets plot both measures of Rt against each other
plt.figure(figsize = (10,5))
plt.plot(df['date'], df['Rt'], color = 'blue', label = 'Rt (Method 1)')
plt.plot(df['date'], df['Rt_lag'], color = 'red', label = 'Rt (Method 2)' )
plt.axhline(1, color = 'black', linestyle = '--' )
plt.xlabel('Date')
plt.ylabel('Rt')
plt.title("Effective Reproduction Number")
plt.grid(True)
plt.show() # the plot shows a lot of similarities between the two methods of calculating the effective reproductive number
# we can calculate the correlation between the two measures using Pearon's coefficient 
df["Rt"].isna().sum()
df["Rt_lag"].isna().sum()
correlation = df["Rt"].corr(df["Rt_lag"])
print(correlation) # a correlation value of 0.71 is strong especially for covid19 data which is typically noisy 
# we could also see the full correlation matrix for all epidemilogical features
df[["Rt","Rt_lag","growth_rate_7d","doubling_time"]].corr()
# we can visualise this using a correlation heatmap on seaborn 
plt.figure(figsize = (6,4))
sns.heatmap(df[["Rt","Rt_lag","growth_rate_7d","doubling_time"]].corr(),
            annot = True, cmap = 'coolwarm')
plt.title("Correlation Between Epidemic Metrics")
plt.show()
# we can use log-scale case and death plots 
# log scale plots are extremely useful in epidemiology because they 
# reveal early exponential growth, make multiple waves comparable and show straight lines during exponential growth
# our first step is to plot cases on a log scale 
plt.figure(figsize = (10,5))
plt.plot(df["date"], df["cases_7d"], label = "Caes (7-day avg)", color = 'black')
plt.yscale('log')
plt.xlabel('Date')
plt.ylabel('Daily new cases (log scale)')
plt.title('UK COVID-19 Daily New Cases (Log Scale)')
plt.grid(True)
plt.legend()
plt.show()    
# now we do the same for deaths, we plot deaths on a log scale 
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["deaths_7d"], label="Deaths (7-day avg)", color="red")

plt.yscale("log")

plt.xlabel("Date")
plt.ylabel("Daily New Deaths (log scale)")
plt.title("UK COVID-19 Daily New Deaths (Log Scale)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.show()
# and we can compare them by plotting them on the same graph 
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["deaths_7d"], label="Deaths (7-day avg)", color="red")
plt.plot(df["date"], df["cases_7d"], label = "Caes (7-day avg)", color = 'blue')
plt.yscale("log")

plt.xlabel("Date")
plt.ylabel("Daily New Deaths (log scale)")
plt.title("UK COVID-19 Daily New Deaths (Log Scale)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.show()   
# we have already used random forest tree regression to see if we can predict covid-19 cases 1 day in advance
# lets now see if random forest tree regression is able to predict cases 7 days in advance 
df["cases_in_7_days"] = df["daily_new_cases"].shift(-7)  
df_model = df.dropna()    
features= ["cases_1d", "cases_2d", "cases_3d", "cases_7d",
           "cases_7d", "cases_14d", "cases_7d_avg", "cases_14d_avg",
           "deaths_yesterday", "active_yesterday"]
X = df_model[features]    
y = df_model["cases_in_7_days"]  
train_size = int(len(df_model) * 0.8)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]
model = RandomForestRegressor(n_estimators = 300,
                              random_state = 42,
                              max_depth = None,)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(rmse)
print("7-day ahead RMSE:", rmse)
# i now want to plot the predicted values versus the test values
# note the we have a large mean squared error so the model does not accurately predict covid 19 cases 7 days in advance 
plt.figure(figsize = (12,6))
plt.plot(y_test.values, label = "Actual Cases (t+7)", linewidth = 2)
plt.plot(predictions, label = "Predicted Cases (t+7)", linewidth = 2)
plt.xlabel("Day")
plt.ylabel("Daily New Cases")
plt.title("Random Forest- Predicting New Cases 7 Days in Advance")
plt.legend()
plt.grid(True)
plt.show()
# note that a random forest can become more accurate as we add more features but only if we add useful information 
# so now we could redo this random forest regression and add more data to see if this reduces the RMSE 
df["cases_4d"] = df["daily_new_cases"].shift(4)
df["cases_5d"] = df["daily_new_cases"].shift(5)
df["cases_6d"] = df["daily_new_cases"].shift(6)
df["cases_10d"] = df["daily_new_cases"].shift(10)
df["cases_12d"] = df["daily_new_cases"].shift(12)
df["cases_15d"] = df["daily_new_cases"].shift(15)
df["cases_18d"] = df["daily_new_cases"].shift(18)
df["cases_20d"] = df["daily_new_cases"].shift(20)
df["cases_21d"] = df["daily_new_cases"].shift(21)
df["cases_25d"] = df["daily_new_cases"].shift(25)
df["cases_28d"] = df["daily_new_cases"].shift(28)
df["cases_21d_avg"] = df["daily_new_cases"].rolling(21).mean()
df["cases_28d_avg"] = df["daily_new_cases"].rolling(28).mean()
df["deaths_7d_avg"] = df["daily_new_deaths"].rolling(7).mean()
df["deaths_14d_avg"] = df["daily_new_deaths"].rolling(14).mean()
df["deaths_21d_avg"] = df["daily_new_deaths"].rolling(21).mean()
df["deaths_28_avg"] = df["daily_new_deaths"].rolling(28).mean()

df["cases_in_7_days"] = df["daily_new_cases"].shift(-7)  
df_model = df.dropna() 
features = ["cases_1d","cases_2d", "cases_3d", "cases_4d","cases_5d",
            "cases_6d","cases_7d","cases_10d","cases_12d","cases_14d",
            "cases_15d","cases_18d","cases_20d","cases_21d",
            "cases_25d","cases_28d","cases_7d_avg","cases_14d_avg",
            "cases_21d_avg","cases_218_avg","deaths_7d_avg",
            "deaths_14d_avg","deaths_21d_avg","deaths_28d_avg"]
X = df_model[features]    
y = df_model["cases_in_7_days"]  

train_size = int(len(df_model) * 0.8)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]
model = RandomForestRegressor(n_estimators = 300,
                              random_state = 42,
                              max_depth = None,)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(rmse) # the rmse value has only dropped slightly it seems as though we need to include far more data to make this predictor accurate 
























































































