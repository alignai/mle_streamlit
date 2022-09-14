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

st.markdown('### Training the model:')
st.markdown('#### Set training Parameters with the following code:')
st.code("""
target = '# of Trips'
features = ['Total Passengers', 'Fuel']

normalize = True
test_size = 0.2

return_metrics = True
plot_result = True
""")

# Separate Data
data_int = data[[target]+features].copy().dropna()
X = data_int.loc[:,features]
y = data_int.loc[:,target]

xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=test_size, shuffle = False)

st.markdown('#### Separate the data:')
st.code("""
data_int = data[[target]+features].copy().dropna()
X = data_int.loc[:,features]
y = data_int.loc[:,target]

xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=test_size, shuffle = False)
""")

# Normalize Feature Data
if normalize:
  scaler =  MinMaxScaler()
  xtrain = scaler.fit_transform(xtrain)
  xtest = scaler.transform(xtest)

st.markdown('#### Normalize the feature data:')
st.code("""
scaler =  MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
""")

# Initiate Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)

st.markdown('#### Initiate Linear Regression Model:')
st.code("""
lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)
""")

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

st.markdown('#### Make Predictions:')
st.code("""
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

fig.show()
""")

st.markdown('#### Linear Regression Model')
st.code("""
def Linear_Regression_Model(data, target, features, normalize, test_size, return_metrics, plot_result):
  '''Returns Linear Regression Model, with performance metrics and plot of test period performance

  Args:
    data: (pandas dataframe) time series data
    target: (str) target column for model
    features: (list[str]) features to be used in model 
    normalize: (bool) if true, training and testing features will be normalized
    test_size: (float) value between 0 and 1 for size of test data split
    return_metrics: (bool) if true, will print model training and testing set MSE and R2
    plot_result: (bool) if true, will plot model testing period performance

  Returns:
    Print of model performance metrics if True
    Plot Object if True
  '''
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

  # Plot
  if plot_result:
    x_ax = X.iloc[-len(xtest):].index
    fig, x = plt.subplots()
    x.scatter(x_ax, ytest, s=5, color="blue", label="original")
    x.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    x.set_title('Linear Regresssion Model Test Performance')
    x.legend()
    st.pyplot(fig)
  
  # Return performance
  if return_metrics:
    print("**Train MSE**: %.2f" % train_mse)
    print("**Train R2**: %.2f" % train_r2)
    print("**Test MSE**: %.2f" % test_mse)
    print("**Test R2**: %.2f" % test_r2)


Linear_Regression_Model(data, target, features, normalize,  test_size, return_metrics, plot_result)
""")
st.markdown('#### Model Output')
Linear_Regression_Model(data, target, features, normalize,  test_size, return_metrics, plot_result)