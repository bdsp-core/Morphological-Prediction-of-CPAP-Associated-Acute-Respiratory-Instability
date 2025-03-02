import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler



# preprocessing from 200Hz --> 10Hz (all CAISR channels)
def do_initial_preprocessing(signals, new_Fs, original_Fs):
    from mne.filter import filter_data, notch_filter
    from scipy.signal import resample_poly
    notch_freq_us = 60.                 # [Hz]
    notch_freq_eur = 50.                # [Hz]
    bandpass_freq_eeg = [0.1, 20]       # [Hz] [0.5, 40]
    bandpass_freq_airflow = [0., 10]    # [Hz]
    bandpass_freq_ecg = [0.3, None]     # [Hz]

    # setup new signal DF
    new_df = pd.DataFrame([], columns=signals.columns)

    for sig in signals.columns:
        # 1. Notch filter
        image = signals[sig].values
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2',
                        'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
            image = notch_filter(image.astype(float), original_Fs, notch_freq_us, verbose=False)
            # image = notch_filter(image, 200, notch_freq_eur, verbose=False)

        # 2. Bandpass filter
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2']:
            image = filter_data(image, original_Fs, bandpass_freq_eeg[0], bandpass_freq_eeg[1], verbose=False)
        if sig in ['abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            image = filter_data(image, original_Fs, bandpass_freq_airflow[0], bandpass_freq_airflow[1], verbose=False)
        if sig == 'ecg':
            image = filter_data(image, original_Fs, bandpass_freq_ecg[0], bandpass_freq_ecg[1], verbose=False)

        # 3. Resample data
        if new_Fs != original_Fs:
            if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 
                            'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
                image = resample_poly(image, new_Fs, original_Fs)
            else:
                image = np.repeat(image, new_Fs)
                image = image[::original_Fs]                

        # 4. Insert in new DataFrame
        new_df.loc[:, sig] = image
    
    del signals
    return new_df

# clip normalizce (all CAISR channels)
def clip_normalize_signals(signals, sample_rate, br_trace, split_loc, min_max_times_global_iqr=20):
    # run over all channels
    for chan in signals.columns:
        # skip labels
        skip_cols = ['stage', 'arousal', 'resp', 'cpap_pressure', 'cpap_on', 'spo2_desat', 'spo2_artifact']
        if np.any([t in chan for t in skip_cols]): continue
        if np.all(signals[chan] == 0): continue

        signal = signals.loc[:, chan].values
        # clips spo2 @60%
        if chan == 'spo2':
            signals.loc[:, chan] = np.clip(signal.round(), 60, 100)
            continue

        # for all EEG (&ECG) traces
        if chan in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 'ecg']:
            # Compute global IQR
            iqr = np.subtract(*np.percentile(signal, [75, 25]))
            threshold = iqr * min_max_times_global_iqr

            # clip outliers
            signal_clipped = np.clip(signal, -threshold, threshold)

            # normalize channel
            sig = np.atleast_2d(signal_clipped).T
            transformer = RobustScaler().fit(sig)
            signal_normalized = np.squeeze(transformer.transform(sig).T)        

        # for all breathing traces
        elif chan in ['abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            # cut split-night recordings and do only local normalization
            region = np.arange(len(signal))
            if split_loc is not None:
                replacement_signal = np.empty(len(signals)) * np.nan
                if chan in [br_trace[0], 'airflow']:
                    region = region[:split_loc]
                elif chan == br_trace[1]:
                    region = region[split_loc:]
                replacement_signal[region] = signal[region]
                signal = replacement_signal
            # ski if all zeros
            if np.all(signal[region]==0): continue
            
            # normalize signal
            signal_clipped = np.clip(signal, np.nanpercentile(signal[region],5), np.nanpercentile(signal[region],95))
            signal_normalized = np.array((signal - np.nanmean(signal_clipped)) / np.nanstd(signal_clipped))
            
            # clip extreme values
            clp = 0.01 if chan in ['abd', 'chest'] else 0.001
            factor = 10 if chan in ['abd', 'chest'] else 20
            quan = 0.2 
            thresh = np.mean((np.abs(np.nanquantile(signal_normalized[region], quan)), np.abs(np.nanquantile(signal_normalized[region], 1-quan))))
            thresh = factor*thresh
            if region[0] == 0:
                signal_normalized[np.concatenate([signal_normalized[region] < -thresh, np.full(len(signal)-len(region), False)])] = -thresh
                signal_normalized[np.concatenate([signal_normalized[region] > thresh, np.full(len(signal)-len(region), False)])] = -thresh
            else:
                signal_normalized[np.concatenate([np.full(len(signal)-len(region), False), signal_normalized[region] < -thresh])] = -thresh
                signal_normalized[np.concatenate([np.full(len(signal)-len(region), False), signal_normalized[region] > thresh])] = thresh
            
        # replace original signal
        signals.loc[:, chan] = signal_normalized
        
    return signals
