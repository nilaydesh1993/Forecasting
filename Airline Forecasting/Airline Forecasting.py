"""
Created on Wed May 20 14:41:03 2020
@author: DESHMUKH
FORECASTING
"""
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf 

from statsmodels.tsa.seasonal import seasonal_decompose
pd.set_option('display.max_column',None)

# =====================================================================================================
# Business Problem - Forecast the Airlines Passengers data set.
# =====================================================================================================

airline = pd.read_excel('Airlines Data.xlsx')
airline.head()

# Time series Plot
airline.Passengers.plot() # Upward Trend And Multiplicative Seasonality

#################################### - Data Preprocessing & EDA - #####################################

# Extracting Months and saving in new columns.
airline["month"] = airline.Month.dt.strftime("%b")

# Creating dummy variables for month columns.
Month_Dummies = pd.get_dummies(airline['month'], drop_first=True)

# Adding dummy variable column in Dataset
airline = pd.concat([airline,Month_Dummies],axis = 1)

# Creating new column t (time period)
airline["t"] = np.arange(1,97)

# Creating new column t_Square (time period)
airline["t_square"] = airline["t"]**2

# Creating new column Log Yt
airline["log_Passengers"] = np.log(airline["Passengers"])

# Line plot for Passengers based on months
sns.lineplot(x="month",y="Passengers",data=airline)

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(airline.Passengers,model="additive",freq=12)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(airline.Passengers,model="multiplicative",freq=12)
decompose_ts_mul.plot()

############################### - Splitting Data into Train and Test - ################################

Train = airline.head(84)
Test = airline.tail(12) # Last season as test data

##################################### - Building Linear Model - #######################################

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear = linear_model.predict(Test)
rmse_linear = np.sqrt(np.mean((Test['Passengers']-pred_linear)**2))
rmse_linear # 53.2

################################### - Building Exponential Model - ####################################

Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = Exp.predict(Test)
rmse_Exp = np.sqrt(np.mean((Test['Passengers']-(np.exp(pred_Exp)))**2))
rmse_Exp # 46.05

#################################### - Building Quadratic Model - #####################################

Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
pred_Quad = Quad.predict(Test)
rmse_Quad = np.sqrt(np.mean((Test['Passengers']-pred_Quad)**2))
rmse_Quad # 48.05

############################## - Building Additive seasonality Model - ################################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_sea = add_sea.predict(Test)
rmse_add_sea = np.sqrt(np.mean((Test['Passengers']-pred_add_sea)**2))
rmse_add_sea # 132.81

###################### - Building Additive seasonality Quadratic Trend Model - ########################

add_sea_Quad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Dec+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = add_sea_Quad.predict(Test)
rmse_add_sea_quad = np.sqrt(np.mean((Test['Passengers']-pred_add_sea_quad)**2))
rmse_add_sea_quad # 26.36

########################### - Building Multiplicative Seasonality Model - #############################

Mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Dec+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((Test['Passengers']-(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea # 140.06

###################### - Building Multiplicative Additive Seasonality Model - #########################

Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Dec+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = Mul_Add_sea.predict(Test)
rmse_Mult_add_sea = np.sqrt(np.mean((Test['Passengers']-(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea # 10.51


# Creating Table with RMSE Values of All above Models.  
mo_data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea",]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
mo_table_rmse=pd.DataFrame(mo_data)
mo_table_rmse

# Selecting the model with minimum RMSE value i.e. Multiplicative Additive Seasonality.

#-------------------------------------- Testing Selected Model ---------------------------------------#

# Saving Data Set
#airline.to_csv("Airlinemodified.csv",encoding="utf-8")

# Bulding Final Model on complete Data
final_model = smf.ols('log_Passengers~t+Jan+Feb+Mar+Dec+May+Jun+Jul+Aug+Sep+Oct+Nov',data = airline).fit()

# Opening data set containing Feture t value for Forcasting.
predict_data = pd.read_csv("for prediction.csv") 

# Forcasting
pred_new = np.exp(final_model.predict(predict_data))
pred_new

# Adding forcasted value in Prefict data dataset
predict_data["forecasted_Passengers"] = round(pred_new)


                         # ---------------------------------------------------- #





