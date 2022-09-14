import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_time_series(column_to_plot, data):
  """ Returns time series plot of specified column of data
    
    Args:
      column_to_plot: (str) column name in dataframe
      data: (pandas dataframe) time series data
    
    Returns:
      Plot object
  """
  fig, ax = plt.subplots(figsize = (10,6))
  ax.plot('Time', column_to_plot, data=data, color='gray', alpha = 0.75)
  ax = sns.regplot(x='Time', y=column_to_plot, data=data, ci=None, scatter_kws=dict(color='0.25'))
  ax.set_title(f'Time Series Plot of Space Trips {column_to_plot} Sales');
  return fig

def plotly_ts_plot(df, value, time_range= ['1992-01-01','2019-12-01']):
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