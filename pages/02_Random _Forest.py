import streamlit as st
import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from components.components import build_header
from components.ml_functions import RF_Model

st.set_page_config(
    page_title="Random Forest",
    page_icon="🚀",
    )

st.sidebar.title("Random Forest")

build_header("Random Forest")


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
    ## Step 3 - Training Random Forest w/ Hyperparameter Tuning

    In this section we will implement a random grid search hyperparameter tuning script, this will automatically optimize our model over the default parameters
    """
st.markdown(output_markdown)


# Parameters file
target = '# of Trips'
features = ['Total Passengers', 'Fuel']
test_size = 0.2

# RF Hyperparameters
n_estimators = [int(x) for x in range(200,2000,200)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Hyperparam tuning dict
tuning_dict = {
    "hyperparameter_tune": False,
    "cv_folds": 5,
    "n_iter": 100,

    "random_grid" : {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap},
    
    'random_state': 42,

    'fixed_grid': {
        'n_estimators': 100,
               'max_features': 1,
               'max_depth': 4,
               'min_samples_split': 5,
               'min_samples_leaf': 3,
               'bootstrap': True
    }

}

# Separate Data
data_int = data[[target]+features].copy().dropna()
X = data_int[[features[0]]]
y = data_int.loc[:,target]

# Split Data
xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=test_size, shuffle = False)

rf_model = RF_Model(xtrain, ytrain, tuning_dict)

