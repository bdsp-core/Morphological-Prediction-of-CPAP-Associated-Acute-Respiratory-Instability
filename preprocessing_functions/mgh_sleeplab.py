import re
import numpy as np
import pandas as pd
import os
import h5py
import scipy.io as sio
import datetime



def get_grass_start_end_time(starttime_raw, endtime_raw):
    
    time_str_elements = starttime_raw.flatten()
    start_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
    time_str_elements = endtime_raw.flatten()
    end_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))

    start_time = start_time.split(':')
    second_elements = start_time[-1].split('.')
    start_time = datetime.datetime(1990,1,1,hour=int(float(start_time[0])), minute=int(float(start_time[1])),
        second=int(float(second_elements[0])), microsecond=int(float('0.'+second_elements[1])*1000000))
    end_time = end_time.split(':')
    second_elements = end_time[-1].split('.')
    end_time = datetime.datetime(1990,1,1,hour=int(float(end_time[0])), minute=int(float(end_time[1])),
        second=int(float(second_elements[0])), microsecond=int(float('0.'+second_elements[1])*1000000))

    return start_time, end_time


def chin_name_standardize(signal_channel_names):
    """Looks like there are different chin EMG configurations, standardize channel name here to 'chin_emg' """
    chin_channels = [x for x in signal_channel_names if 'chin' in x]
    if len(chin_channels) == 0:
        return signal_channel_names

    emg_channel = chin_channels[0]
    signal_channel_names = [x.replace(emg_channel, 'chin_emg') for x in signal_channel_names]

    return signal_channel_names

def eog_name_standardize(signal_channel_names):
    """Different/multiple EOG channels are usually available. rename one to 'eog' here """
    eog_channels = ['e2-m1', 'e1-m2', 'e2-m2', 'e1-m1']
    for eog_channel in eog_channels:
        if eog_channel in signal_channel_names:
            break

    signal_channel_names = [x.replace(eog_channel, 'eog') for x in signal_channel_names]

    return signal_channel_names

def load_mgh_signal(signal_path, channels=None, reverse_sign=False, rename_chin=False, rename_eog=False):

    start_time = None
    end_time = None
    try: # usually Grass, saved as pre 7.3 format:
        ff = sio.loadmat(signal_path)
        data_path = os.path.basename(signal_path)
        if 's' not in ff:
            raise Exception('No signal found in %s.'%data_path)
        signal = ff['s']
        if reverse_sign:
            signal = -signal
        channel_names = [ff['hdr'][0,i]['signal_labels'][0].upper().replace('EKG','ECG') for i in range(ff['hdr'].shape[1])]
        if 'grass' in signal_path:
            Fs = 200.
        else:
            raise Exception('Safety check to make sure this is Grass data with Fs=200')

    except: # saved as .mat 7.3. new grass or natus files:

        with h5py.File(signal_path,'r') as ff:

            hdr = ff['hdr']
            signal_labels = hdr['signal_labels'][:]
            channel_names = [np.squeeze(ff[signal_labels[i,0]][:]) for i in range(signal_labels.shape[0])]
            channel_names = [''.join([chr(y) for y in x]) if x.size>1 else chr(x) for x in channel_names]

            signal = ff['s'][()]
            signal = np.transpose(signal)

            if 'recording' in ff.keys(): # only available for natus:
                
                Fs = int(np.squeeze(ff['recording']['samplingrate']))
                
                if pd.notna(np.squeeze(ff['recording']['year'])):
                    year = int(np.squeeze(ff['recording']['year']))
                    month = int(np.squeeze(ff['recording']['month']))
                    day = int(np.squeeze(ff['recording']['day']))
                    hour = int(np.squeeze(ff['recording']['hour']))
                    if (hour >= 7) and (hour <=10):         # 'typo' by sleep techs
                        hour = hour + 12
                    minute = int(np.squeeze(ff['recording']['minute']))
                    second = int(np.squeeze(ff['recording']['second']))
                    millisecond = int(np.squeeze(ff['recording']['millisecond']))

                    start_time = datetime.datetime(1990,1,1,hour=hour, minute=minute,
                            second=second, microsecond=int(millisecond*1000))
                    end_time = start_time+datetime.timedelta(seconds=max(signal.shape)/Fs)

            else: # grass:
                if 'grass' in signal_path:
                    Fs = 200.
                else:
                    raise Exception('Safety check to make sure this is Grass data with Fs=200')

    # end of loading part
    ##################################
    
    # only take signal channels to study
    if channels is None:
        signal_channel_ids = list(range(len(channel_names)))
        signal_channel_names = channel_names
        
    elif 'SumEffort' in channels:
        signal_channel_ids = []
        signal_channel_names = []
        for ichannel in ['ABD', 'CHEST']:
            found = False
            for j in range(len(channel_names)):
                if channel_names[j]==ichannel.upper():
                    signal_channel_ids.append(j)
                    signal_channel_names.append(channel_names[j])
                    found = True
                    break
            if not found:
                raise Exception('Channel %s is not found.'%ichannel)
        signal = signal[signal_channel_ids,:]#.T
        # do effort belt average here:
        signal = np.sum(signal,0,keepdims=1)/2
        
    else:
        signal_channel_ids = []
        signal_channel_names = []
        for i in range(len(channels)):
            found = False
            for j in range(len(channel_names)):
                #if channel_names[j]==channels[i].upper():
                if re.search(channels[i], channel_names[j], re.IGNORECASE):
                    signal_channel_ids.append(j)
                    signal_channel_names.append(channel_names[j])
                    found = True
                    break
            if not found:
                raise Exception('Channel %s is not found.'%channels[i])
        signal = signal[signal_channel_ids,:]#.T

    # check whether the signal contains NaN
    if np.any(np.isnan(signal)):
        raise Exception('Found Nan in signal in %s'%data_path)

    signal_channel_names = [x.lower().replace('sao2', 'spo2').replace('ekg', 'ecg') for x in signal_channel_names]

    if rename_chin:
        signal_channel_names = chin_name_standardize(signal_channel_names)
    if rename_eog:
        signal_channel_names = eog_name_standardize(signal_channel_names)
    params = {'Fs':Fs*1.0, 'channel_ids': signal_channel_ids, 'channel_names': signal_channel_names, 'start_time':start_time, 'end_time':end_time}

    signal = pd.DataFrame(data=signal.transpose(), columns=signal_channel_names)
    
    return signal, params


def annotations_preprocess(annotations, fs):
    """
    input: dataframe annotations.csv
    output: dataframe annotations with new columns: event starts/ends in seconds and ends
    """

    annotations['time'] = pd.to_datetime(annotations['time'], infer_datetime_format=1)
    annotations.loc[pd.isna(annotations.duration), 'duration'] = 1/fs
    annotations['duration'].apply(lambda x: datetime.timedelta(seconds=x))

    t0 = annotations.time.iloc[0]

    annotations['dt_start'] = (annotations['time'] - t0).apply(lambda x: x.seconds)
    annotations['dt_end'] = annotations['dt_start'] + annotations['duration']
    try:
        annotations['idx_start'] = np.floor(annotations['dt_start'] * fs).astype(int)
        annotations['idx_end'] = np.ceil(annotations['dt_end'] * fs).astype(int)
    except:
        annotations['idx_start'] = np.floor(annotations['dt_start'] * fs).astype(float)
        annotations['idx_end'] = np.ceil(annotations['dt_end'] * fs).astype(float)
    annotations = annotations.sort_values('idx_start', ignore_index=True)
    return annotations

def vectorization(events_annotations_selection, mapping, signal_len):
    """
    Inputs: 
    events_annotations_selection: dataframe (annotations.csv), only rows selected that shall be vectorized.
    mapping: dataframe defining the vectorization mapping.
    Output: 1D numpy array, vectorized annotations
    """
    
    events_vectorized = np.zeros(signal_len, )
    for jloc, row in events_annotations_selection.iterrows():
        for event_type in mapping.index:
            keyword = mapping.loc[event_type, 'keyword']
            if keyword.lower() in row.event.lower():
                value = mapping.loc[event_type, 'value']
                try:
                    events_vectorized[int(row['idx_start']) : int(row['idx_end'])] = value
                except:
                    import pdb; pdb.set_trace()
                break # event saved, proceed with next.
                
    return events_vectorized

def vectorize_respiratory_events(annotations, signal_len):
    """
    Input: annotations.csv as dataframe
    Output: vectorized respiratory array (OA: 1, CA: 2, MA: 3, HY: 4, RA: 5)
    """
    
    # definition of the following categories of respiratory event, their keyword and the vectorization-mapping:
    keyword_value_pairs = np.array([['obstructive', 1],
                                   ['central', 2],
                                   ['mixed', 3],
                                   ['hypopnea', 4],
                                   ['rera', 5],
                                   ])

    mapping = pd.DataFrame(index = ['OA', 'CA', 'MA', 'HY', 'RA'], columns=['keyword', 'value'], data=np.array(keyword_value_pairs))
    
    resp_events = annotations.copy().dropna(axis='index')
    resp_events = resp_events.loc[resp_events.event.str.lower().apply(lambda x: (('resp' in x) & ('event' in x)) | ('apnea' in x) | ('hypopnea' in x) | ('rera' in x)), :]
    resp_events_vectorized = vectorization(resp_events, mapping, signal_len)
    
    return resp_events_vectorized

def vectorize_sleep_stages(annotations, signal_len, noscore_fill=np.nan):
    """
    Input: annotations.csv as dataframe
    Output: vectorized sleepstage array (W: 5, R: 4, N1: 3, N2: 2, N3: 1)
    """
    
    # definition of the following categories of respiratory event, their keyword and the vectorization-mapping:
    keyword_value_pairs = np.array([['3', 1],
                                   ['2', 2],
                                   ['1', 3],
                                   ['r', 4],
                                   ['w', 5],
                                   ])

    mapping = pd.DataFrame(index = ['N3', 'N2', 'N1', 'R', 'W'], columns=['keyword', 'value'], data=np.array(keyword_value_pairs))
    
    sleep_stages = annotations.copy().dropna(axis='index')
    sleep_stages = sleep_stages.loc[sleep_stages.event.apply(lambda x: 'sleep_stage' in str(x).lower()), :]

    sleep_stages_vectorized = vectorization(sleep_stages, mapping, signal_len)
    sleep_stages_vectorized[sleep_stages_vectorized == 0] = noscore_fill # set no-scored sleep stage to NaN instead of 0.

    return sleep_stages_vectorized

def vectorize_arousals(annotations, signal_len):
    """
    Input: annotations.csv as dataframe
    Output: vectorized sleepstage array (Non-arousal: 0, arousal: 1 )
    """
    
    # definition of the following categories of respiratory event, their keyword and the vectorization-mapping:
    keyword_value_pairs = np.array([['arousal', 1],
                                   ])

    mapping = pd.DataFrame(index = ['arousal'], columns=['keyword', 'value'], data=np.array(keyword_value_pairs))
    
    arousal_events = annotations.copy().dropna(axis='index')
    arousal_events = arousal_events.loc[arousal_events.event.str.lower().apply(lambda x: ('arousal' in x) & ('post' not in x)), :]
    
    arousal_events_vectorized = vectorization(arousal_events, mapping, signal_len)

    return arousal_events_vectorized