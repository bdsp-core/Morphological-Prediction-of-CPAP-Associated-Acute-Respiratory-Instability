import gc
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.integrate import simpson as simps


def add_spo2_prior_window_to_list(data, spo2_collection_100s_prior, idx_tmp, fs):

    spo2_reference_lookback=100  # in seconds
    spo2_prior = data.loc[idx_tmp - timedelta(seconds=spo2_reference_lookback): idx_tmp, 'spo2'].values
    spo2_prior = spo2_prior[:spo2_reference_lookback * fs]

    if not len(spo2_prior) == spo2_reference_lookback * fs:
        if (idx_tmp < data.iloc[[spo2_reference_lookback * fs + 1]].index)[0]:  # spo2_reference_lookback*fs+1:
            spo2_prior = np.concatenate([np.ones(spo2_reference_lookback * fs - spo2_prior.shape[0], ) * spo2_prior[0], spo2_prior])
        elif (idx_tmp > data.iloc[[-spo2_reference_lookback * fs - 1]].index)[0]:  # data.shape[0] - spo2_reference_lookback*fs - 1:
            spo2_prior = np.concatenate([spo2_prior, np.ones(spo2_reference_lookback * fs - spo2_prior.shape[0], ) * spo2_prior[-1]])
    if not len(spo2_prior) == spo2_reference_lookback * fs:
        import pdb; pdb.set_trace()

    spo2_collection_100s_prior.append(spo2_prior)
    del spo2_prior

    return spo2_collection_100s_prior



    return data


def add_spo2_window_to_list(data, spo2_collection, idx_tmp, search_window_max_prior, search_window_max_post, fixed_length, fs):
    w0 = idx_tmp + search_window_max_prior*fs
    w1 = idx_tmp + search_window_max_post*fs

    spo2_window = data.loc[w0:w1, 'spo2'].values
    spo2_window = spo2_window[:fixed_length]

    if not len(spo2_window) == fixed_length:
        # pad spo2_window (first or last 50 seconds) at beginning and end so that it has correct shape.
        spo2_window = np.concatenate(
            [np.ones((fixed_length - spo2_window.shape[0]) // 2 + 1, ) * spo2_window[0], spo2_window,
             np.ones((fixed_length - spo2_window.shape[0]) // 2 + 1, ) * spo2_window[-1]])
        spo2_window = spo2_window[:fixed_length]
        assert(len(spo2_window) == fixed_length)

    if any(np.isfinite(spo2_window)):
        spo2_collection.append(spo2_window)
    del spo2_window

    return spo2_collection


def add_spo2_prior_window_to_list(data, spo2_collection_100s_prior, idx_tmp, fs):

    spo2_reference_lookback = 100  # in seconds
    spo2_prior = data.loc[idx_tmp - (fs*spo2_reference_lookback): idx_tmp, 'spo2'].values
    spo2_prior = spo2_prior[:spo2_reference_lookback * fs]

    if not len(spo2_prior) == spo2_reference_lookback * fs:
        if (idx_tmp < data.iloc[[spo2_reference_lookback * fs + 1]].index)[0]:  # spo2_reference_lookback*fs+1:
            spo2_prior = np.concatenate([np.ones(spo2_reference_lookback * fs - spo2_prior.shape[0], ) * spo2_prior[0], spo2_prior])
        elif (idx_tmp > data.iloc[[-spo2_reference_lookback * fs - 1]].index)[0]:  # data.shape[0] - spo2_reference_lookback*fs - 1:
            spo2_prior = np.concatenate([spo2_prior, np.ones(spo2_reference_lookback * fs - spo2_prior.shape[0], ) * spo2_prior[-1]])
    if not len(spo2_prior) == spo2_reference_lookback * fs:
        import pdb; pdb.set_trace()

    if any(np.isfinite(spo2_prior)):
        spo2_collection_100s_prior.append(spo2_prior)
    del spo2_prior

    return spo2_collection_100s_prior


def compute_hypoxia_burden(data, fs, apnea_name = 'Apnea', hypoxia_name = 'hypoxic_area', hours_sleep = None, search_window_type='patient_specific', 
                        search_window_max_prior = -50, search_window_max_post = 50, verbose=False):
    '''
    hours_sleep: can be either string - name for sleep stage column in dataframe. or int/float for manual hours_of_sleep value.
    '''
    fs = int(fs)

    if not apnea_name in data.columns:
        print(f"compute_hypoxia_burden(): '{apnea_name}' not in data.columns, return data unchanged.")
        return data, np.nan

    data['Apnea_Binary'] = int(0)
    for i_chunk in np.arange(0, data.shape[0]//(fs*3600)):
        data.loc[data.index[i_chunk*fs*3600 : (i_chunk+1)*fs*3600+fs], 'Apnea_Binary'] = np.isin(data[apnea_name].iloc[i_chunk*fs*3600 : (i_chunk+1)*fs*3600+fs].fillna(method='ffill', limit=fs), [1,2,3,4]).astype(int)

    data['Apnea_End'] = data['Apnea_Binary'].diff() == -1 # MUST BE BOOLEAN.
    apnea_end_idx = data.loc[data['Apnea_End'] == True].index
    data[hypoxia_name] = 0
    if apnea_end_idx.shape[0] == 0:
        hypoxic_burden = 0
        return data, hypoxic_burden
    spo2_collection = []
    spo2_collection_100s_prior = []
    fixed_length = int(fs*(np.abs(search_window_max_prior) + np.abs(search_window_max_post))) # slicing/selection operation with fixed length because length of arrays can vary by 1 index due to second-based-"index selection


    data_spo2 = data.loc[:, ['spo2']].copy()
    for idx_tmp in apnea_end_idx:

        spo2_collection = add_spo2_window_to_list(data_spo2, spo2_collection, idx_tmp, search_window_max_prior, search_window_max_post, fixed_length, fs)

        spo2_collection_100s_prior = add_spo2_prior_window_to_list(data_spo2, spo2_collection_100s_prior, idx_tmp, fs)

        gc.collect()

    spo2_refs = np.array(spo2_collection_100s_prior)
    spo2_refs = np.nanquantile(spo2_refs, 0.99, axis=1)
    mean_spo2 = np.nanmean(np.array(spo2_collection), axis=0)
    mean_spo2_firsthalf = mean_spo2[:mean_spo2.shape[0]//2]
    
    if search_window_type == 'patient_specific':
        # as done in original publication, get the max locations of Spo2 before and after apnea event end:
        search_window0 = len(mean_spo2_firsthalf) - np.argmax(mean_spo2_firsthalf[::-1]) - 1
        search_window1 = int(mean_spo2.shape[0]//2+fs*10 + np.argmax(mean_spo2[int(mean_spo2.shape[0]//2+fs*10):]))
        x = np.arange(search_window_max_prior,search_window_max_post+1/fs,1/fs)
        sec_prior = np.round(x[search_window0],1)
        sec_post = np.round(x[search_window1],1)
        
    elif search_window_type == 'fixed_length':
        # do not use patient specific window but a fixed length for SpO2 AUC computation.
        sec_prior = search_window_max_prior
        sec_post = search_window_max_post      


    idx_prior = data.index[data['Apnea_End']] + (fs*sec_prior)
    idx_post = data.index[data['Apnea_End']] + (fs*sec_post)
    idx_last_apneapred = data.index[data['Apnea_End']] - fs
    areas = []
    for i0, i1, i_apneaend, ref in list(zip(idx_prior, idx_post, idx_last_apneapred, spo2_refs)):
        spo2_tmp = data.loc[i0:i1,'spo2'].values
        if sum(pd.isna(spo2_tmp))/len(spo2_tmp) > 0.3:
            # if more than 30% are nan values, don't compute hypoxic burden
            data.loc[i0:i1, hypoxia_name] = np.nan
        else: 
            spo2_tmp = spo2_tmp[np.logical_not(pd.isna(spo2_tmp))]
            spo2_drop = ref - spo2_tmp
            if not all(np.isfinite(spo2_drop)):
                continue
            spo2_drop[spo2_drop<0] = 0
            area_tmp = simps(spo2_drop, dx=1/fs)
            areas.append(area_tmp)
            data.loc[i_apneaend, hypoxia_name] = area_tmp
    # compute the actual hypoxic burden for this recording:
    if hours_sleep is not None:

        if (type(hours_sleep) == int) | (type(hours_sleep) == float):
            hours_sleep = hours_sleep # nothing :) manual hours of sleep setting.

        if type(hours_sleep) == str:
            hours_sleep = sum(data[hours_sleep].dropna()<5)/fs/3600 # compute hours of sleep with sleep_stage available.

        areas = np.array(areas)/60 # in minutes
        areas = areas[pd.notna(areas)]
        areas_robust = areas[areas < np.std(areas)*3]
        hypoxic_burden = np.sum(areas_robust)/hours_sleep
        hypoxic_burden = np.round(hypoxic_burden,1)

    if verbose:
        print(f'hypoxic burden (%min)/h: {hypoxic_burden}')
    # print('end hypoxia computation')
    return data, hypoxic_burden


def hypoxaemic_burden_minutes(spo2_signal, fs, minimum_spo2_level=90, verbose=False, return_startendpoints=False):
    desaturation_spo2 = desaturation_detection(spo2_signal)
    spo2_signal_tmp = spo2_signal.copy()
    spo2_signal_tmp[np.isnan(spo2_signal_tmp)] = 100
    desaturation_spo2[spo2_signal_tmp >= minimum_spo2_level] = 0

    start_points = np.where(np.diff(desaturation_spo2) == 1)[0]
    end_points = np.where(np.diff(desaturation_spo2) == -1)[0]

    resaturation, resaturation_events = calculate_resaturation_periods(spo2_signal, desaturation_spo2, fs)

    desat_durations_min = (end_points - start_points) / fs / 60

    hypoxaemic_burden = np.sum(desat_durations_min)
    T90desaturation = np.sum(desat_durations_min[resaturation_events == 1])
    T90nonspecific = np.sum(desat_durations_min[resaturation_events == 0])

    if verbose:
        print(f'total burden (min): {hypoxaemic_burden}')
        print(f'T90 desaturation (min): {T90desaturation}')
        print(f'T90 nonspecific (min): {T90nonspecific}')

    if return_startendpoints:
        return hypoxaemic_burden, T90desaturation, T90nonspecific, start_points, end_points

    return hypoxaemic_burden, T90desaturation, T90nonspecific


def desaturation_detection(spo2_signal):
    spo2_derivative = np.diff(spo2_signal)
    threshold = 4
    desaturation = np.zeros(spo2_signal.shape)
    max_duration = len(spo2_signal)

    for k in range(len(spo2_derivative)):
        duration, samples_limit = 0, 0
        if not np.isfinite(spo2_derivative[k]):
            continue
        if spo2_derivative[k] < 0:
            while spo2_derivative[k + duration] <= 0:
                level = spo2_signal[k] - spo2_signal[k + duration + 1]
                duration += 1
                if level < threshold:
                    samples_limit = duration + 1
                if k + duration >= len(spo2_derivative):
                    break
            if (samples_limit <= max_duration - 1) & (level >= threshold):
                desaturation[k: k + duration] = 1

    return desaturation


def calculate_resaturation_periods(spo2_signal, desaturation, fs):
    max_resat_duration = 150 * fs
    start_points = np.where(np.diff(desaturation) == 1)[0]
    end_points = np.where(np.diff(desaturation) == -1)[0]
    resaturation = np.zeros(spo2_signal.shape)
    resaturation_events = np.zeros((len(start_points), 1))

    if len(start_points) == 0:
        resaturation_events = np.squeeze(resaturation_events)
        return resaturation, resaturation_events

    if (len(end_points) < len(start_points)) & (end_points[-1] < start_points[-1]):
        end_points = list(end_points) + [len(desaturation)]
        end_points = np.array(end_points)

    assert len(start_points) == len(end_points)

    drop = []
    for start, end in zip(start_points, end_points):
        drop.append(np.nanmax(spo2_signal[start:end]) - np.nanmin(spo2_signal[start:end]))
    drop = np.array(drop)

    resat_threshold = drop - np.ceil(drop * 2 / 3)

    total_break = False
    for k in range(len(start_points)):

        resat_duration = 0
        while spo2_signal[end_points[k] + resat_duration] < spo2_signal[start_points[k]] - resat_threshold[k]:
            resat_duration += 1
            if end_points[k] + resat_duration + 1 >= len(spo2_signal): 
                total_break = True
                break
        if total_break: break

        while spo2_signal[end_points[k] + resat_duration] == spo2_signal[start_points[k]] - resat_threshold[k]:
            resat_duration -= 1

        resat_flag = 0
        try:
            if (resat_duration <= max_resat_duration) & (spo2_signal[end_points[k] + resat_duration - 1] < spo2_signal[
                int(start_points[k] - resat_threshold[k])]):
                resat_flag = 1

            while (spo2_signal[int(end_points[k] + resat_duration)] <= spo2_signal[
                int(end_points[k] + resat_duration + 1)]) & \
                    (spo2_signal[int(start_points[k])] >= spo2_signal[int(end_points[k] + resat_duration + 1)]) & \
                    (resat_flag == 1):
                resat_duration += 1
                if end_points[k] + resat_duration + 1 >= len(spo2_signal) or end_points[k] + resat_duration + 1 >= len(spo2_signal): 
                    break
        except: import pdb; pdb.set_trace()

        while spo2_signal[end_points[k] + resat_duration] == spo2_signal[end_points[k] + resat_duration - 1]:
            resat_duration -= 1
        if resat_flag == 1:
            resaturation[end_points[k] + 1: end_points[k] + resat_duration] = 1
            resaturation_events[k] = 1
        
        # break if end of flist is reached
        if end_points[k] + resat_duration + 1 >= len(spo2_signal): break

    resaturation_events = np.squeeze(resaturation_events)

    return resaturation, resaturation_events


def sleep_stage_aswti(data, stage_columnname, min_sleep=30, fs=10, verbose=False):
    """
    compute a sleep stage that changes W to N1 in amplified sleep wake transition instabilities.
    input:   data: dataframe with {stage_columnname} 
             stage_columnname: name of the sleep stage column
             min_sleep: minimum of sleep (minutes) that has to be in 1 hour window so that sleep stage is considered to be corrected.
    output:  data with new columnname {stage_columnname + '_aswti'}
    """ 

    if stage_columnname not in data.columns:
        if verbose:
            print(f'{stage_columnname} not in data.columns, therefore no ASWTI version can be computed.')
        return data

    data['sleep_binary'] = (data[stage_columnname] < 5).astype(int)
    data['sleep_30min_present'] = data['sleep_binary'].rolling('1H').sum() >= fs * 60 * min_sleep
    data['sleep_30min_present'] = np.concatenate([data['sleep_30min_present'].values[fs*60*min_sleep:] , np.zeros(fs*60*min_sleep,)])
    data['wake_30min_present'] = (data['sleep_binary'] == 0).rolling(f'{min_sleep}min').sum() == fs * 60 * min_sleep
    data['wake_30min_present'] = pd.Series(data['wake_30min_present'].values[::-1]).rolling(fs * 60 * min_sleep).max().values[::-1]
    data['wake_30min_present'].fillna(method='ffill', limit = fs*60*min_sleep+1, inplace=True)
    data['aswti_to_replace'] = ((data['sleep_binary'] == 0) & (data['sleep_30min_present'] == 1) & (data['wake_30min_present'] == 0)).astype(int)
    data[f'{stage_columnname}_aswti'] = data[stage_columnname].values
    data.loc[data['aswti_to_replace'].values == 1, f'{stage_columnname}_aswti'] = 3 # set as N1.
    # data[f'{stage_columnname}_aswti1'] = data[f'{stage_columnname}_aswti'].values

    data['aswti_to_replace_and_wake'] = 0
    data.loc[data.index[1]:, 'aswti_to_replace_and_wake'] = np.multiply(data['aswti_to_replace'].diff().values[1:], (data[f'{stage_columnname}_aswti'] == 5).astype(int).values[:-1])
    data.loc[data.index[1]:, 'aswti_to_replace_and_wake'] += np.multiply(data['aswti_to_replace'].diff().values[:-1], (data[f'{stage_columnname}_aswti'] == 5).astype(int).values[1:])

    idx_wake_pre = np.where(data.aswti_to_replace_and_wake == -1)[0]
    idx_wake_post = np.where(data.aswti_to_replace_and_wake == 1)[0]
    for idx_wake_pre_tmp in idx_wake_pre:
        idx_start_wake = np.where(data.sleep_binary == 1)[0]
        idx_start_wake = idx_start_wake[idx_start_wake < idx_wake_pre_tmp][-1]
        data.loc[data.index[idx_start_wake : idx_wake_pre_tmp], 'aswti_to_replace'] = 0

    for idx_wake_post_tmp in idx_wake_post:
        idx_start_wake = np.where(data.sleep_binary == 1)[0]
        idx_start_wake = idx_start_wake[idx_start_wake > idx_wake_post_tmp][0]
        data.loc[data.index[idx_wake_post_tmp : idx_start_wake], 'aswti_to_replace'] = 0

    data[f'{stage_columnname}_aswti'] = data[f'{stage_columnname}_aswti'].values
    data.loc[data['aswti_to_replace'].values == 1, f'{stage_columnname}_aswti'] = 3 # set as N1.

    data.drop(['sleep_binary', 'sleep_30min_present', 'wake_30min_present', 'aswti_to_replace', 'aswti_to_replace_and_wake'], axis=1, inplace=True)
    
    return data


def compute_sleep_indices(data_timerange, column_stage, fs, name='result'):

    signal_available_h = len(data_timerange[column_stage].dropna()) / fs / 3600 + 0.000001
    hours_sleep = sum(
        data_timerange[column_stage].dropna() < 5) / fs / 3600 + 0.000001  # (just add for numerical stability)
    perc_W = sum(data_timerange[column_stage].dropna() == 5) / fs / 3600 / signal_available_h
    perc_S = sum(data_timerange[column_stage].dropna() < 5) / fs / 3600 / signal_available_h
    perc_R = sum(data_timerange[column_stage].dropna() == 4) / fs / 3600 / hours_sleep
    perc_N1 = sum(data_timerange[column_stage].dropna() == 3) / fs / 3600 / hours_sleep
    perc_N2 = sum(data_timerange[column_stage].dropna() == 2) / fs / 3600 / hours_sleep
    perc_N3 = sum(data_timerange[column_stage].dropna() == 1) / fs / 3600 / hours_sleep

    stages = data_timerange[column_stage].values

    # sleep fragmentation index:
    w_or_n1 = np.isin(stages, [5, 3])
    deep_sleep = np.isin(stages, [1, 2, 4])
    fragmentation_shift = np.logical_and(w_or_n1[1:], deep_sleep[:-1])
    fragmentation_pos = np.where(fragmentation_shift)[0]
    sfi = np.round(len(fragmentation_pos) / hours_sleep, 1)

    # SFI_W, based only on transitions to W from N2, N3 or R
    stages_without_n1 = stages[np.isin(stages, [1, 2, 4, 5])]
    w = np.isin(stages_without_n1, [5])
    deep_sleep = np.isin(stages_without_n1, [1, 2, 4])
    fragmentation_shift = np.logical_and(w[1:], deep_sleep[:-1])
    fragmentation_pos = np.where(fragmentation_shift)[0]
    sfi_w = np.round(len(fragmentation_pos) / hours_sleep, 1)

    # arousal index, based on transitions to W from sleep
    w = np.isin(stages, [5])
    sleep = np.isin(stages, [1, 2, 3, 4])
    fragmentation_shift = np.logical_and(w[1:], sleep[:-1])
    fragmentation_pos = np.where(fragmentation_shift)[0]
    #     fragmentation_pos = smooth_fragmentation_index(fragmentation_pos, fs=10)
    arousali = np.round(len(fragmentation_pos) / hours_sleep, 1)

    statistics_result = pd.DataFrame(
        data=[signal_available_h, hours_sleep, perc_W, perc_S, perc_R, perc_N1, perc_N2, perc_N3, sfi, sfi_w, arousali],
        index=['hours_data', 'hours_sleep', 'perc_W', 'perc_S', 'perc_R', 'perc_N1', 'perc_N2', 'perc_N3', 'sfi',
               'sfi_w', 'arousali'],
        columns=[name])

    return statistics_result


def compute_spo2_clean(data, fs=10):
    data['spo2_clean'] = data['spo2'].copy()

    spo2_badqualityth = data.spo2.median() - 30
    spo2_badq = np.where(data.spo2 < spo2_badqualityth)[0]

    if len(spo2_badq) > 0:
        spo2_badq_start = np.concatenate([[spo2_badq[0]], spo2_badq[1:][np.diff(spo2_badq) > 1]])
        spo2_badq_end = np.concatenate([[spo2_badq[-1]], spo2_badq[::-1][:-1][np.diff(spo2_badq[::-1]) < -1]])[::-1]

        # NOTE: I've inserted drop=True now.
        bad_quality = (data.spo2 < spo2_badqualityth).astype(int).reset_index(drop=True)
        bad_quality = bad_quality.rolling(20 * fs, min_periods=0).max()
        bad_quality = (bad_quality[::-1].rolling(20 * fs, min_periods=0).max())[::-1]
        bad_quality.index = data.index
        data.loc[bad_quality.values.astype(bool).flatten(), 'spo2_clean'] = np.nan

        bad_quality_prior = bad_quality.reset_index(drop=True)[::-1].rolling(5 * 60 * fs, min_periods=0).max()[::-1]
        bad_quality_post = bad_quality.reset_index(drop=True).rolling(5 * 60 * fs, min_periods=0).max()

        bad_quality_prior_10min = bad_quality.reset_index(drop=True)[::-1].rolling(10 * 60 * fs, min_periods=0).max()[::-1]
        bad_quality_post_10min = bad_quality.reset_index(drop=True).rolling(10 * 60 * fs, min_periods=0).max()
        # bad_q_area = bad_quality_prior['spo2'].values.astype(bool) & bad_quality_post['spo2'].values.astype(bool)
        bad_q_area = bad_quality_prior.values.astype(bool) & bad_quality_post.values.astype(bool)
        bad_q_area = pd.Series(bad_q_area)
        bad_q_area.index = data.index
        data.loc[bad_q_area.values.astype(bool).flatten(), 'spo2_clean'] = np.nan
        # once a bad quality has been detected, remove all data under 80 in the area, if still included:
        data.loc[bad_quality_prior_10min.values.astype(bool).flatten() & (data.spo2_clean <= 80), 'spo2_clean'] = np.nan
        data.loc[bad_quality_post_10min.values.astype(bool).flatten() & (data.spo2_clean <= 80), 'spo2_clean'] = np.nan

    return data


def compute_spo2_perc_below_90(data):
    if not 'spo2_clean' in data.columns:
        return data

    spo2_clean = data.spo2_clean.dropna().values
    if len(spo2_clean) == 0:
        return np.nan

    spo2_perc_below_90 = len(spo2_clean[spo2_clean < 90]) / len(spo2_clean)

    return spo2_perc_below_90


def hypoxia_drops(data, drop_magnitude=3, max_gap=10, fs=10):
    if not 'spo2_clean' in data.columns:
        return data, np.nan, np.nan
    spo2_clean = data.spo2_clean.dropna().values
    if len(spo2_clean) == 0:
        return data, np.nan, np.nan

    data['spo2_clean_drop_45s'] = np.nan
    data.loc[data.index[::fs], 'spo2_clean_drop_45s'] = data['spo2_clean'][::fs].astype('float32').rolling(45,
                                                                                                           min_periods=5).apply(
        lambda x: x.max() - x[-1], raw=False)

    # get the start and end_idx of the drops
    data_drop = data.spo2_clean_drop_45s.dropna()  # supposed to be in 1-second resolution here.
    start = (data_drop >= drop_magnitude).astype(int).diff() == 1
    end = (data_drop >= drop_magnitude).astype(int).diff() == -1
    if len(start[start == True]) == len(end[end == True]) + 1:
        end.iloc[-1] = 1

    if len(start[start == True]) == len(end[end == True]) - 1:
        start.iloc[-1] = 1

    start_idx = np.where(start)[0]
    end_idx = np.where(end)[0]

    # for end-start times of drops that are close together, bridge the drop value so they are connected
    end_indices_close_to_next_start = np.where(start_idx[1:] - end_idx[:-1] < max_gap)[0]
    for idx_tmp in end_indices_close_to_next_start:
        loc_to_change = data_drop.iloc[end_idx[idx_tmp]: start_idx[idx_tmp + 1]].index
        data.loc[loc_to_change, 'spo2_clean_drop_45s'] = drop_magnitude

    # after connection, let's get updated start and end_idx
    data_drop = data.spo2_clean_drop_45s.dropna()
    start = (data_drop >= drop_magnitude).astype(int).diff() == 1
    end = (data_drop >= drop_magnitude).astype(int).diff() == -1
    if len(start[start == True]) == len(end[end == True]) + 1:
        end.iloc[-1] = 1
    if len(start[start == True]) == len(end[end == True]) - 1:
        start.iloc[-1] = 1

    if start_idx.shape != end_idx.shape:
        import pdb; pdb.set_trace()


    start_idx = np.where(start)[0]
    end_idx = np.where(end)[0]

    min_drop_duration = 10
    hypoxia_drop_short = end_idx - start_idx < min_drop_duration
    hypoxia_drop_long = end_idx - start_idx >= min_drop_duration

    # start_loc = data_drop[start_idx[hypoxia_drop_short]].index
    # end_loc = data_drop[end_idx[hypoxia_drop_short]].index

    no_hypoxia_short = len(start_idx[hypoxia_drop_short])
    no_hypoxia_long = len(start_idx[hypoxia_drop_long])

    return data, no_hypoxia_short, no_hypoxia_long



def compute_avg_spo2(data, hdr):
    spo2 = data.spo2.values
    spo2 = spo2[np.where(data.patient_asleep)[0]]
    spo2 = spo2[spo2>60]
    
    # for now it's just average spo2 during sleep
    hdr['avg. spo2'] = np.mean(spo2)

    return data, hdr


def compute_hypoxic_burden(data, fs, label_version='4%'):
    
    # compute hypoxia variables
    data = compute_spo2_clean(data.copy(), fs=fs)
    data['spo2'] = data['spo2_clean']
    col = 'apnea' if label_version == '4%' else 'V5_labels'
    data['apnea_binary'] = np.array(data[col] > 0).astype(int)
    data['apnea_end'] = np.isin(data['apnea_binary'].diff(), [-1])

    # compute sleep time
    stage = data.sleep_stages.values
    stage[~np.isfinite(stage)] = 0
    sleep_time = np.sum(np.logical_and(stage<5, stage>0)==1) / (3600*fs)

    # compute hypoxic burden
    data, hypoxia_burden = compute_hypoxia_burden(data, fs, hours_sleep=sleep_time, apnea_name=col)    
    
    # compute related time metrics
    T90burden, T90desaturation, T90nonspecific = hypoxaemic_burden_minutes(data['spo2'].values, fs)
    
    # save data
    data = data.drop(columns=['apnea_binary', 'apnea_end', 'Apnea_Binary', 'Apnea_End', 'spo2_clean'])

    return hypoxia_burden, [T90burden, T90desaturation, T90nonspecific]

