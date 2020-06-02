"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    placement = ['Placement - Day of Month',
                 'Placement - Weekday (Mo = 1)',
                 'Placement - Time',
                 'Confirmation - Day of Month',
                 'Confirmation - Weekday (Mo = 1)',
                 'Confirmation - Time']

    f_testx = feature_vector_df.drop(columns=['User Id', 'Order No', 'Rider Id'] + placement)

    def cleaner(input_df, nullthreshold=0.9, correlation_thresh=0.95, day_of_month_cols=[], day_of_week_cols=[]):
        input_dfc = input_df.copy()

        #########################################################################################
        # The Code below drops columns that have null values exceeding threshold and Columns that have ONLY one value
        input_dfc.drop(columns=['Vehicle Type', 'Precipitation in millimeters'], inplace=True)

        #########################################################################################

        #########################################################################################
        # This code converts time given by am and pm into seconds then applies cosine and sine
        def time_to_seconds(input_df):
            input_dfc = input_df.copy()

            from datetime import datetime

            for time_col in [col for col in input_df.columns if 'Time' in [col[-4:]]]:

                input_dfc[time_col + '_sin(seconds)'] = \
                    input_df[time_col].apply(lambda time: np.sin(
                        (datetime.strptime(time, '%I:%M:%S %p') - datetime(1900, 1, 1)).total_seconds()
                        * (2. * np.pi / 86400)))  # there are 86400 seconds in a day

                input_dfc[time_col + '_cos(seconds)'] = \
                    input_df[time_col].apply(lambda time: np.cos(
                        (datetime.strptime(time, '%I:%M:%S %p') - datetime(1900, 1, 1)).total_seconds()
                        * (2. * np.pi / 86400)))

                input_dfc.drop(columns=[time_col], inplace=True)

            return input_dfc

        input_dfc2 = time_to_seconds(input_dfc)
        #########################################################################################

        #########################################################################################
        # This code encodes ['Platform Type', 'Personal or Business']

        def one_encoder(input_df):
            for plat in ['Platform Type_2', 'Platform Type_3', 'Platform Type_4']:
                input_df[plat] = [1 if int(plat[-1]) == x else 0 for x in input_df['Platform Type']]

            input_df['Personal or Business_Personal'] = [1 if x == 'Personal' else 0
                                                         for x in input_df['Personal or Business']]

            return input_df.drop(columns=['Platform Type', 'Personal or Business'])

        input_dfc2 = one_encoder(input_dfc2)
        #########################################################################################

        def cyclic_days(input_df, month_days_cols, weekdays_cols):
            input_dfc = input_df.copy()

            for mday_col in month_days_cols:
                if mday_col in input_dfc.columns:

                    input_dfc[mday_col + '_sin(day)'] = input_df[mday_col].apply(lambda day: np.sin(
                        day * (2. * np.pi / 31))
                    )

                    input_dfc[mday_col + '_cos(day)'] = input_df[mday_col].apply(lambda day: np.cos(
                        day * (2. * np.pi / 31))
                    )

                    input_dfc.drop(mday_col, inplace=True, axis=1)

            for wday_col in weekdays_cols:
                if wday_col in input_dfc.columns:

                    input_dfc[wday_col + '_sin(day)'] = input_df[wday_col].apply(lambda day: np.sin(
                        day * (2. * np.pi / 7))
                    )

                    input_dfc[wday_col + '_cos(day)'] = input_df[wday_col].apply(lambda day: np.cos(
                        day * (2. * np.pi / 7))
                    )

                    input_dfc.drop(wday_col, inplace=True, axis=1)
                else:
                    continue

            return input_dfc

        input_dfc2 = cyclic_days(input_dfc2, day_of_month_cols, day_of_week_cols)
        #########################################################################################

        input_dfc2.drop(columns=['Pickup - Time_sin(seconds)',
                                 'Pickup - Time_cos(seconds)',
                                 'Pickup - Day of Month_sin(day)',
                                 'Pickup - Day of Month_cos(day)',
                                 'Pickup - Weekday (Mo = 1)_sin(day)',
                                 'Pickup - Weekday (Mo = 1)_cos(day)'], inplace=True)
        #########################################################################################

        return input_dfc2
    day_of_month_cols = ['Arrival at Pickup - Day of Month', 'Pickup - Day of Month']
    day_of_week_cols = ['Arrival at Pickup - Weekday (Mo = 1)', 'Pickup - Weekday (Mo = 1)']

    return cleaner(f_testx, day_of_month_cols=day_of_month_cols, day_of_week_cols=day_of_week_cols)


def load_model(path_to_model: str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction.tolist()
