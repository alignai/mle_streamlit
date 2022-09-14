import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def plot_time_series(column_to_plot, data):
    """ Returns time series plot of specified column of data

      Args:
        column_to_plot: (str) column name in dataframe
        data: (pandas dataframe) time series data

      Returns:
        Plot object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot('Time', column_to_plot, data=data, color='gray', alpha=0.75)
    ax = sns.regplot(x='Time', y=column_to_plot, data=data,
                     ci=None, scatter_kws=dict(color='0.25'))
    ax.set_title(f'Time Series Plot of Space Trips {column_to_plot} Sales')
    return fig


def plotly_ts_plot(df, value, time_range=['1992-01-01', '2019-12-01']):
    """ Returns plotly plot of data over specified time range 
    Args:
      df: (Dataframe) timeseries data with index as datetime type
      value: (str) column name to plot
      time_range: (list of str) two string elements signaling start and end date
    Returns:
      plotly plot object

    """
    df = df.loc[time_range[0]: time_range[1]]
    fig = px.line(df, x=df.index, y=value, markers=True)
    fig.update_layout(title=f'Space Agency {value}')
    return fig


def Linear_Regression_Model(data, target, features, normalize, test_size, return_metrics, plot_result):
    """Returns Linear Regression Model, with performance metrics and plot of test period performance

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
    """
    # Separate Data
    data_int = data[[target]+features].copy().dropna()
    X = data_int.loc[:, features]
    y = data_int.loc[:, target]

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, shuffle=False)

    # Normalize Feature Data
    if normalize:
        scaler = MinMaxScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

    # Initiate Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(xtrain, ytrain)

    # Make Predictions
    ytrain_pred = lr_model.predict(xtrain)
    train_mse = mean_squared_error(ytrain_pred, ytrain)
    train_r2 = r2_score(ytrain_pred, ytrain)

    ypred = lr_model.predict(xtest)
    test_mse = mean_squared_error(ytest, ypred)
    test_r2 = r2_score(ytest, ypred)
    
    # Return performance
    if return_metrics:
        st.markdown("##### Train MSE: %.2f" % train_mse)
        st.markdown("##### Train R2: %.2f" % train_r2)
        st.markdown("##### Test MSE: %.2f" % test_mse)
        st.markdown("##### Test R2: %.2f" % test_r2)

    # Plot
    if plot_result:
        x_ax = X.iloc[-len(xtest):].index
        fig, x = plt.subplots()
        x.scatter(x_ax, ytest, s=5, color="blue", label="original")
        x.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
        x.set_title('Linear Regresssion Model Test Performance')
        x.legend()
        st.pyplot(fig)


def RF_Model(xtrain, ytrain, tuning_dict):
    """Returns Random Forest model object with hyperparameter tuning if selected

    Args:
      xtrain: (pandas dataframe) time series features data
      ytrain: (pandas series) time series target column data
      tuning_dict: (dictionary) contains model training parameters and hyperparameter tuning parameters

    Returns:
      sklearn RandomForest model object
    """

    if tuning_dict['hyperparameter_tune']:
        # Initiate Random Forest Model
        rf = RandomForestRegressor()
        # Begin Random Grid Search
        rf_random = RandomizedSearchCV(estimator=rf,
                                       param_distributions=tuning_dict['random_grid'],
                                       n_iter=tuning_dict['n_iter'],
                                       cv=tuning_dict['cv_folds'], verbose=2, random_state=tuning_dict['random_state'], n_jobs=-1)
        rf_random.fit(xtrain, ytrain)

        return rf_random.best_estimator_

    else:
        # Include the option to fit a fixed model
        rf = RandomForestRegressor(**tuning_dict['fixed_grid'])
        rf.fit(xtrain, ytrain)
        return rf
