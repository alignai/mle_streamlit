import streamlit as st
import os 
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

from components.components import build_header
from components.ml_functions import Linear_Regression_Model

st.set_page_config(
    page_title="Linear Regression",
    page_icon="🚀",
    )

st.sidebar.title("Linear Regression")

build_header("Linear Regression")


if st.session_state.get('data') is not None:
    data = st.session_state.get('data')
else:
    data_file = os.path.join('data', 'space_trip.csv')
    data = pd.read_csv(data_file, index_col = 0)
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace = True)
    data['Time'] = np.arange(len(data.index))
    st.session_state['data'] = data
    

output_markdown = """
    ## Step 2 - Training - Linear Regression Model
    Lets begin our first model, a simple linear regression model to fit the equation below:

    `target = weight_1 * lag_1 + weight_2 * lag_2 + bias`

    Our goals for this section are:
    1. Create Linear Regression Model
    2. Hyperparameter Tuning <- Not Done, Linear Regression doesn't have
    """
st.markdown(output_markdown)


# Parameters file
target = '# of Trips'
features = ['Total Passengers', 'Fuel']

normalize = True
test_size = 0.2

return_metrics = True
plot_result = True

# Separate Data
data_int = data[[target]+features].copy().dropna()
X = data_int.loc[:,features]
y = data_int.loc[:,target]

xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=test_size, shuffle = False)

# Normalize Feature Data
if normalize:
  scaler =  MinMaxScaler()
  xtrain = scaler.fit_transform(xtrain)
  xtest = scaler.transform(xtest)

# Initiate Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)

# Make Predictions
ytrain_pred=lr_model.predict(xtrain)
train_mse = mean_squared_error(ytrain_pred,ytrain)
train_r2 = r2_score(ytrain_pred,ytrain)

ypred=lr_model.predict(xtest)
test_mse = mean_squared_error(ytest, ypred)
test_r2 = r2_score(ytest, ypred)

df = data.loc[ '1992-01-01':'2019-12-01']
x_ax = X.iloc[-len(xtest):].index
# x_ax, ytest, s=5, color="blue", label="original"
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_ax, y= ytest,
                    mode='markers',
                    name='markers'))
fig.add_trace(go.Scatter(x=x_ax, y= ypred,
                    mode='lines+markers',
                    name='lines+markers'))
fig.update_layout(title='Model Performance')

st.plotly_chart(fig)

Linear_Regression_Model(data, target, features, normalize,  test_size, return_metrics, plot_result)