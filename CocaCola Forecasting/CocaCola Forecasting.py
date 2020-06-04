"""
Created on Thu May 21 10:41:07 2020
FORECASTING
"""
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf 

from statsmodels.tsa.seasonal import seasonal_decompose
pd.set_option('display.max_column',None)

# =====================================================================================================
# Business Problem - Forecast the CocaCola Sales Rawdata Sales data set.
# =====================================================================================================

coca = pd.read_excel('CocaCola_Sales_Rawdata.xlsx')
coca.head()

# Time series Plot
coca.Sales.plot() # Upward Trend And Additive Seasonality

#################################### - Data Preprocessing & EDA - #####################################

# Extracting Quarters (as first 2 latter from Quarter column) and saving in new columns.
coca['quarters']=0
for i in range(42):
    p = coca["Quarter"][i]
    coca['quarters'][i]= p[0:2]

# Creating dummy variables for quarters columns.
quarters_Dummies = pd.get_dummies(coca['quarters'])

# Adding dummy variable column in Dataset
coca = pd.concat([coca,quarters_Dummies],axis = 1)

# Creating new column t (time period)
coca["t"] = np.arange(1,43)

# Creating new column t_Square (time period)
coca["t_square"] = coca["t"]**2

# Creating new column Log Y
coca["log_Sales"] = np.log(coca["Sales"])

# Line plot for Sales based on quarterss
sns.lineplot(x="quarters",y="Sales",data=coca)

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(coca.Sales,model="additive",freq=4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(coca.Sales,model="multiplicative",freq=4)
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(coca.Sales,lags=4)
tsa_plots.plot_pacf(coca.Sales,lags=4)

############################### - Splitting Data into Train and Test - ################################

Train = coca.head(38)
Test = coca.tail(4) # Last Season as test data

##################################### - Building Linear Model - #######################################

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear = linear_model.predict(Test)
rmse_linear = np.sqrt(np.mean((Test['Sales']-pred_linear)**2))
rmse_linear # 591.55

################################### - Building Exponential Model - ####################################

Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = Exp.predict(Test)
rmse_Exp = np.sqrt(np.mean((Test['Sales']-(np.exp(pred_Exp)))**2))
rmse_Exp # 466.24

#################################### - Building Quadratic Model - #####################################

Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = Quad.predict(Test)
rmse_Quad = np.sqrt(np.mean((Test['Sales']-pred_Quad)**2))
rmse_Quad # 475.56

############################## - Building Additive seasonality Model - ################################

add_sea = smf.ols('Sales~Q1+Q2+Q3',data=Train).fit()
pred_add_sea = add_sea.predict(Test)
rmse_add_sea = np.sqrt(np.mean((Test['Sales']-pred_add_sea)**2))
rmse_add_sea # 1860

###################### - Building Additive seasonality Quadratic Trend Model - ########################

add_sea_Quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3',data=Train).fit()
pred_add_sea_quad = add_sea_Quad.predict(Test)
rmse_add_sea_quad = np.sqrt(np.mean((Test['Sales']-pred_add_sea_quad)**2))
rmse_add_sea_quad # 301.73

########################### - Building Multiplicative Seasonality Model - #############################

Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((Test['Sales']-(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea # 1963.38

###################### - Building Multiplicative Additive Seasonality Model - #########################

Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3',data = Train).fit()
pred_Mult_add_sea = Mul_Add_sea.predict(Test)
rmse_Mult_add_sea = np.sqrt(np.mean((Test['Sales']-(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea # 225.52


# Creating Table with RMSE Values of All above Models.  
mo_data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
mo_table_rmse=pd.DataFrame(mo_data)
mo_table_rmse

# Selecting the model with minimum RMSE value i.e. Multiplicative Additive Seasonality.

#-------------------------------------- Testing Selected Model ---------------------------------------#

# Saving Data Set
#coca.to_csv("cocamodified.csv",encoding="utf-8")

# Bulding Final Model on complete Data
final_model = smf.ols('log_Sales~t+Q1+Q2+Q3',data = coca).fit()

# Opening data set containing Feture t value for Forcasting.
predict_data = pd.read_csv("for prediction.csv") 

# Forcasting
pred_new = np.exp(final_model.predict(predict_data))
pred_new

# Adding forcasted value in Prefict data dataset
predict_data["forecasted_Sales"] = round(pred_new)


                         # ---------------------------------------------------- #





