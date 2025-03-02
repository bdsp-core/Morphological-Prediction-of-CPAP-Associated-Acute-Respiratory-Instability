import numpy as np

# import feature functions
from preprocessing_functions.Class_Preprocessing import * 


def bad_signal_check(data, Fs):
    # signal is flat ..
    breathing_std = data['Ventilation_combined'].rolling(Fs, center=True).std().fillna(-1)
    data['low_std'] = np.logical_and(breathing_std<0.01, breathing_std>-0.01)
    # spo2 is flat 
    spo2 = data.SpO2 if not 'smooth_saturation' in data.columns else data.smooth_saturation
    data['flat_spo2'] = np.logical_or(spo2<=60, np.isnan(spo2))

    # combined is bad signal
    data['bad_sig'] = np.logical_and(data['low_std'], data['flat_spo2'])
    data['bad_signal'] = data['bad_sig'].rolling(3*Fs, center=True).min().fillna(1)

    # add flat signal detection
    data['low_std'] = np.logical_and(breathing_std<0.05, breathing_std>-0.05)
    flat = data['low_std'].rolling(2*Fs, center=True).max().fillna(1)
    # keep only segments > 5min
    data['flat_signal'] = 0
    for st, end in find_events(flat):
        if (end-st) > 5*60*Fs:
            data.loc[st:end, 'flat_signal'] = 1

    # remove some columns
    data = data.drop(columns=['bad_sig', 'flat_spo2', 'low_std'])

    return data

def remove_error_events(data, trace, columns):
    for col in columns:
        for st, end in find_events(data[col].values>0):
            # remove flat breathing segments
            flat_cols = [f'flat_signal_{trace}', f'bad_signal_{trace}']
            if np.any(data.loc[st:end, flat_cols] > 0):
                data.loc[st:end, col] = 0
            # remove if 75% of event, SpO2 is unavailable
            if sum(data.loc[st:end, 'SpO2'].isna()) > 0.75*(end-st):
                data.loc[st:end, col] = 0

    return data