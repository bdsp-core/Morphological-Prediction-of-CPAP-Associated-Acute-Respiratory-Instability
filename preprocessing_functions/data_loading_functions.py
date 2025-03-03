import h5py
import numpy as np
import pandas as pd

# import feature functions
from .Class_Preprocessing import * 
from .caisr_functions import *


from .preprocessing_utils import do_initial_preprocessing, clip_normalize_signals


## read CAISR prepared data ##
def load_prepared_data(path, fs=200):
    # read in signals
    f = h5py.File(path, 'r')
    keys = [k for k in f.keys()]
    data = pd.DataFrame([])

    # for accurate channel names
    for key in keys:
        subkeys = f[key].keys()
        for subkey in subkeys:
            val = np.squeeze(f[key][subkey][:])
            if len(data) == 0:
                data[subkey] = np.squeeze(f[key][subkey][:])
            else:
                diff = np.abs(len(val)-len(data))
                assert diff < fs/2, f'Incorrect column lengths in prepared data'
                data[subkey] = np.squeeze(f[key][subkey][:len(data)])            
    f.close()

    return data

def load_prepared_data_old(file_path:str, get_signals:list=[]):
    """ 
    load prepared data format (post March 22, 2023). Currently, all signals are read. 
    Inputs:
    file_path: path to prepared .h5 file.
    """
    # init DF
    Xy = pd.DataFrame([])
    
    # read file
    with h5py.File(file_path, 'r') as f:
        # Loop over each group and dataset within each group and get data
        group_names = list(f.keys())
        for group_name in group_names:        
            group = f[group_name]

            # list all existing datasets
            dataset_names = list(group.keys())
            # get all, or take from get_signals
            get_names = dataset_names if len(get_signals)==0 else [c for c in dataset_names if c in get_signals]
            for dataset_name in get_names:
                dataset = group[dataset_name][:]
                assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
                Xy[dataset_name] = dataset.flatten()
        
        # save attributes
        params = {}
        for attrs in f.attrs.keys():
            params[attrs] = f.attrs[attrs]

    return Xy, params

## breathing trace seletion ##
def cpapOn_select_breathing_trace(signals):
    # take selection    
    split_loc = np.where(signals.cpap_on==1)[0]
    if len(split_loc) == 0:
        split_loc = None
        selection = ['ptaf']
    else:
        split_loc = split_loc[0]
        first_sleep = np.where(np.logical_and(signals.stage>0, signals.stage<5))[0]
        assert len(first_sleep)>0, 'No sleep found for this patient!'
        if split_loc < first_sleep[0]:
            split_loc = None
            selection = ['cflow']
        else:
            selection = ['ptaf', 'cflow']

    return selection, split_loc

def mgh_select_breathing_trace(signals, path, Fs):
    # find recording type from table1
    csv_file = dbx_pfx + 'Thijs tnassi@bidmc.harvard.edu/Thijs_main/sleep_lookup_TIMES.csv'
    mapping_table = pd.read_csv(csv_file) 
    row = mapping_table.loc[np.where(mapping_table.HashFileName == path.split('/')[-1].split('.')[0])[0]]
    if len(row) == 0:
        row = mapping_table.loc[np.where(mapping_table.FilePath == path.split('/')[-1].split('.')[0])[0]]
        if len(row) == 0: raise Exception('WARNING: No breathing information! (not in lookup table)') 
    # retrieve info from table
    psg_type = row.psg_type.values[0]
    st_time = row.start_time.values[0]
    cpap_time = row.CPAP_time.values[0]
    treat_time = row.treatment_time.values[0]
    # check psg type
    if psg_type == 'diagnostic':
        signals['cpap_on'] = 0
        selection, split_loc = ['ptaf'], None
    elif psg_type == 'titration':
        if (type(treat_time) != str and type(cpap_time) != str): raise Exception('WARNING: No breathing information! (titration w/o tag)') 
        signals['cpap_on'] = 1
        selection, split_loc = ['cflow'], None
    # if recording is split-night, find split location
    elif psg_type == 'split':
        selection = ['ptaf', 'cflow']
        if type(st_time) != str: raise Exception('WARNING: No breathing information! (no start_time)') 
        st = pd.to_datetime(st_time)
        # select treatment time if possible
        if treat_time != '-1':
            end = pd.to_datetime(treat_time) 
        else:
            if cpap_time != '-1':
                end = pd.to_datetime(cpap_time)
            else:  raise Exception('WARNING: No breathing information! (split w/o tag)') 
        # set split location
        if end < st: end += pd.Timedelta(days=1)
        loc = len(pd.date_range(start=st, end=end, freq=str(1/Fs)+'S'))
        signals['cpap_on'] = 0
        signals.loc[loc:, 'cpap_on'] = 1        
        split_loc = np.where(signals.cpap_on==1)[0][0]
    else:
        raise Exception('WARNING: No breathing information! (incorrect psg type)') 

    return selection, split_loc

def morphology_select_breathing_trace(signals, Fs):
    st = np.where(signals.stage < 5)[0][0]
    end = len(signals) - np.where(np.flip(signals.stage.values) < 5)[0][0]

    # if one signal has a std of 0, take other
    br_traces = ['ptaf', 'cflow']
    stds = np.array([np.std(signals.ptaf), np.std(signals.cflow)])
    if sum(stds==0) == 1:
        return [br_traces[np.where(stds>0)[0][0]]], None

    # check for noise 
    noise = []
    for c, col in enumerate(['ptaf', 'cflow']):    
        is_noise, _, _ = check_for_noise(signals.loc[st:end, col], Fs)
        noise.append(is_noise)
    if sum(noise) == 1:
        return [br_traces[np.where(np.array(noise)==False)[0][0]]], None
    
    # this may be a split night!
    newDF = pd.DataFrame([])
    window = 30*60*Fs
    ptaf_std = signals['ptaf'].rolling(int(60*Fs), center=True).std() 
    newDF['ptaf_norm'] = ptaf_std/np.max(ptaf_std) > .1
    ptaf_on = newDF['ptaf_norm'].rolling(int(window), center=True).max().values
    cflow_std = signals['cflow'].rolling(int(60*Fs), center=True).std() 
    newDF['cflow_norm'] = cflow_std/np.max(cflow_std) > .1
    cflow_on =  newDF['cflow_norm'].rolling(int(window), center=True).max().values

    # determine split loc
    loc1 = st + np.where(ptaf_on[st:end] == 0)[0] - window/2
    loc2 = end - np.where(np.flip(cflow_on[st:end]) == 0)[0]+ window/2
    if len(loc1) == len(loc2) == 0: 
        return ['ptaf'], None
    elif len(loc1) > 0 and len(loc2) == 0:
        loc1 = loc1[0]
        loc2 = loc1
    elif len(loc2) > 0 and len(loc1) == 0:
        loc2 = loc2[0]
        loc1 = loc2
    else:
        loc1 = loc1[0]
        loc2 = loc2[0]

    # split locs should be withing 60min in both traces
    if np.abs(loc2-loc1) < Fs*60*30:
        split_loc = int(np.mean([loc1, loc2]))
    else:
        # print('WARNING: Seems to be split night, but weird split loc found..!')
        split_loc = loc2
        return [], None

    # correct for unlikely split-nights
    if split_loc < 0.1*len(signals):
        return ['cflow'], None
    elif split_loc > 0.9*len(signals):
        return ['ptaf'], None
    
    return ['ptaf', 'cflow'], split_loc

def check_for_noise(sig, Fs):
    sig = sig - np.median(sig)

    # test signal for white noise
    ps = np.abs(np.fft.fft(sig))**2
    time_step = 1 / Fs
    freqs = np.fft.fftfreq(sig.size, time_step)
    idx = np.argsort(freqs)

    min_freq = 0.01
    max_freq = 0.5
    locs = (freqs[idx] > min_freq) * (freqs[idx] < max_freq) 

    f_range = freqs[idx][locs]
    power = ps[idx][locs]

    # power peak
    peak = f_range[np.argmax(power)]
    # compute percentage over 75% peak
    percentage = len(np.where(power > 0.75*max(power))[0]) / len(power) * 100

    # peak = round(np.abs(f_range[np.nanquantile(power, 0.8)]), 2)
    is_noise = True if peak < 0.1 else False

    return is_noise, peak, percentage

## archived ## 
def find_epoch_start(signals, Fs):
    epoch_size = int(round(30*Fs))
    valids = [True if s in range(1, 6) else False for s in signals['stage']]
    # import pdb; pdb.set_trace()
    if np.all(valids): 
        stages = signals['stage'].values.astype(int)
    else:
        signals = cut_leading_and_trailing_NaNs(signals, valids)
        stages = signals['stage'].values.astype(int)

    # start checking idices for initial epoch start
    start_loc, cnt = 0, 0
    check = False
    while check == False:
        cnt += 1
        if cnt > 1000: raise Exception('No good epoch start is found..!')
        # set epoch indices
        start_ids = np.arange(start_loc, len(stages)-epoch_size+1, epoch_size)
        seg_ids = list(map(lambda x:np.arange(x,x+epoch_size), start_ids))

        # verify if each segment includes one label
        for i, seg_id in enumerate(seg_ids):
            num_labels = len(np.unique(stages[seg_id]))
            if num_labels != 1:
                start_loc += 1
                break
            if i == len(seg_ids)-1:
                # print('Epoch start found @ i==%s'%start_loc)
                check = True
        
        if start_loc > epoch_size:
            print('Try another cut..')
            signals = try_another_cut(signals, Fs)
            

    return signals, start_ids

def cut_leading_and_trailing_NaNs(signals, valids):
    nans = np.diff(valids)
    locs = np.where(nans)[0]

    if len(locs) == 1 and not valids[-1]:
        st = 0
        end = locs[0]
        # cut st and end from signals
        signals = signals.loc[st:end, :]
    elif len(locs) == 2 and signals.loc[locs[0]+1, 'stage'] in range(1,6):
        st = locs[0] + 1
        end = locs[1]
        # cut st and end from signals
        signals = signals.loc[st:end, :]
    else:
        # st = locs[0] + 1
        # end = locs[1]
        print('Intermittent invalid indices found: %s (%s)'%(locs, len(signals)))
        signals = signals.loc[valids,:]
    
    
    # reset indices
    signals.reset_index(drop=True, inplace=True)


    assert(np.all([True if s in range(1, 6) else False for s in signals['stage']]))

    return signals

def try_another_cut(signals, Fs):
    # cut leading and trailing epoch
    locs = np.where(np.diff(signals['stage'].values.astype(int)))[0]
    st = locs[0] + 1
    end = locs[-1]
    # cut st and end from signals
    signals = signals.loc[st:end, :]
    stages = signals['stage'].values.astype(int)
    epoch_size = int(round(30*Fs))
    start_ids = np.arange(0, len(stages)-epoch_size+1, epoch_size)
    seg_ids = list(map(lambda x:np.arange(x,x+epoch_size), start_ids))

    # verify if each segment includes one label
    for i, seg_id in enumerate(seg_ids):
        num_labels = len(np.unique(stages[seg_id]))
        if num_labels != 1:
            raise Exception('No good epoch start is found..!')
        if i == len(seg_ids)-1:
            # print('Epoch start found @ i==0*')
            check = True

    return signals

## data formating ##
def setup_header(path, new_Fs, original_Fs, br_trace, split_loc):
    # setup hdr
    hdr = {}
    hdr['newFs'] = new_Fs
    hdr['Fs'] = original_Fs
    tt = 'diagnostic'
    if len(br_trace) == 0:       tt = 'unknown'
    elif len(br_trace) == 2:     tt = 'split-night' 
    elif br_trace[0] == 'cflow': tt = 'titration'
    hdr['test_type'] = tt
    hdr['rec_type'] = 'CAISR'
    hdr['cpap_start'] = split_loc
    hdr['patient_tag'] = path.split('/')[-1].split('.')[0]

    return hdr


###################################
## main data-loaders per dataset ##
###################################

def load_data_from_prepared_dataset(path, dataset_folder, original_Fs=200, new_Fs=10, signals=[], scorer_3_labels=True):  
    # get cohort channel legend
    channels = get_cohort_channels(dataset_folder)

    # set to read-in channels
    br_channels = ['ptaf', 'cflow']
    th = ['airflow']
    st = ['stage']
    re = ['resp']
    ar = ['arousal']
    if len(signals) == 0: signals = channels
    select_channels = signals + br_channels + st + re + ar
    if 'airflow' in channels: select_channels += th
    if 'robert' in dataset_folder:
        select_channels += ['cpap_on', 'cpap_pressure']
        if scorer_3_labels: 
            select_channels += ['stage_0', 'resp_0', 'stage_1', 'resp_1', 'stage_2', 'resp_2']

    # find cohort dependent channel indices 
    select_channels = np.unique(select_channels)
    idxs = np.array([])
    for ch in select_channels:
        idxs = np.concatenate([idxs, get_channel_idxs(channels, [ch])])

    sort_order = np.argsort(idxs)
    sorted_channels = select_channels[sort_order]
    sorted_idxs = idxs[sort_order].astype(int)

    # read in signals
    f = h5py.File(path, 'r')
    sig = f['Xy'][sorted_idxs].astype(float)
    signals_df = pd.DataFrame(sig.T, columns=sorted_channels)            
    f.close()

    # first select breathing trace based on morphology
    br_trace, split_loc = morphology_select_breathing_trace(signals_df, new_Fs)

    # apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)
    # correct split loc
    if type(split_loc) == int: split_loc = int(split_loc*new_Fs/original_Fs)

    # find best breathing trace
    if np.all([True if b in signals_df.columns else False for b in ['ptaf', 'cflow']]):
        if 'robert' in dataset_folder:
            br_trace, split_loc = cpapOn_select_breathing_trace(signals_df)
            # if titration, drop ptaf/airflow
            if split_loc == 0:
                singals_df = signals_df.drop(columns=['ptaf', 'airflow'])
        elif 'mgh' in dataset_folder:
            try:
                br_trace, split_loc = mgh_select_breathing_trace(signals_df, path, new_Fs)
            except Exception as error:
                # here we just keep our br_trace based on morphology
                print(error)
                if len(br_trace) > 0:
                    tag = f'.. so we keep >{br_trace}< based on morphology'
                    if len(br_trace) > 1: tag += f' --> split @{split_loc}\n'
                    print(tag)
                else:
                    print(' and couldn\'t determine a breathing trace based on morphology..')
                    if br_channels in signals:
                        raise Exception('\t--> SO SKIP')
                    else: print('\t--> OK, not requested.')
                
    # drop cpap_on channel
    if 'cpap_on' in signals_df.columns: signals_df.drop(columns=['cpap_on'], inplace=True)

    # clip / normalize
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # set breathing trace
    if np.all([True if b in signals_df.columns else False for b in br_trace]):
        if split_loc is None:
            signals_df['breathing_trace'] = signals_df[br_trace].values
        else:
            signals_df['breathing_trace'] = np.nan
            signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
            signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]
    else:
        import pdb; pdb.set_trace()

    # setup Header & DF
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)
    
    return signals_df, hdr

def load_data_from_prepared_dataset_Jedi(path, original_Fs=200, new_Fs=10, signals=[], show=True):
    # read in signals
    signals_df = load_prepared_data(path)
    signals_df['stage'] = signals_df['stage_expert_1']
    signals_df['arousal'] = signals_df['arousal-platinum_converted_0']
    signals_df['resp'] = signals_df['resp-h3_expert_1']
    del_cols = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'e2-m1', 'chin1-chin2', 
                'ecg', 'lat', 'rat', 'cpres']
    del_cols += [c for c in signals_df.columns if 'limb' in c or 'majority' in c]
    signals_df = signals_df.drop(columns=del_cols)

    # apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)

    # set breathing trace
    br_trace, split_loc = cpapOn_select_breathing_trace(signals_df)

    # drop cpap_on channel
    if 'cpap_on' in signals_df.columns: signals_df.drop(columns=['cpap_on'], inplace=True)

    # clip / normalize
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # set breathing trace
    if split_loc is None:
        signals_df['breathing_trace'] = signals_df[br_trace].values
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    # setup Header & DF
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)
    if show:
        print(hdr['test_type'] + ' study (based on "cpap_on")')

    return signals_df, hdr

def load_data_from_prepared_dataset_MGH(path, original_Fs=200, new_Fs=10, signals=[], show=True):  
    # read in signals
    signals_df = load_prepared_data(path)
    signals_df = signals_df.rename(columns={"arousal-shifted_converted_0": "arousal"})
    signals_df = signals_df.rename(columns={"resp-h3_converted_0": "resp"})
    signals_df = signals_df.rename(columns={"stage_expert_0": "stage"})
    drop_cols = ['resp-h4_expert_0', 'arousal_expert_0', 'limb_expert_0', 'c3-m2', 'c4-m1', 'cpres', 'rat', 
                    'e1-m2', 'e2-m1', 'ecg', 'f3-m2', 'f4-m1', 'lat', 'o1-m2', 'o2-m1','chin1-chin2']
    signals_df = signals_df.drop(columns=drop_cols)

    

    # apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)
    # correct split loc
    # if type(split_loc) == int: split_loc = int(split_loc*new_Fs/original_Fs) 

    # first select breathing trace based on morphology
    br_trace, split_loc = morphology_select_breathing_trace(signals_df, new_Fs)     
                
    # clip / normalize
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # set breathing trace
    if split_loc is None:
        if len(br_trace) > 0:
            signals_df['breathing_trace'] = signals_df[br_trace].values
        else: 
            signals_df['breathing_trace'] = signals_df['abd'] + signals_df['chest']
            print('"Abd + Chest" is used as breathing trace (replacing ptaf)')
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]
        
    # setup Header & DF
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)
    if show:
        print(hdr['test_type'] + ' study (based on "morphology")')
    
    return signals_df, hdr  
    
def load_data_from_prepared_dataset_QA(path, original_Fs=200, new_Fs=10, signals=[]):
    # read in signals
    signals_df = load_prepared_data(path)
    signals_df['stage'] = signals_df['stage_expert_bi-gold']
    signals_df['arousal'] = signals_df['arousal_expert_bi-gold']
    signals_df['resp'] = signals_df['resp-h3_expert_bi-gold']
    del_cols = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'e2-m1', 'chin1-chin2', 
                'ecg', 'lat', 'rat', 'cpap_pressure']
    del_cols += [c for c in signals_df.columns if 'limb' in c or 'majority' in c]
    signals_df = signals_df.drop(columns=del_cols)

    # apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)

    # Select breathing trace based on morphology
    br_trace, split_loc = cpapOn_select_breathing_trace(signals_df)
    print(f'we keep >{br_trace}< based on CPAP ON')

    # drop cpap_on channel
    if 'cpap_on' in signals_df.columns: signals_df.drop(columns=['cpap_on'], inplace=True)

    # clip / normalize
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # set breathing trace
    if split_loc is None:
        signals_df['breathing_trace'] = signals_df[br_trace].values
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    # setup Header & DF
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)

    return signals_df, hdr
    
def load_data_from_prepared_dataset_RERA(path, original_Fs=200, new_Fs=10, signals=[]):
    # read in signals
    signals_df = load_prepared_data(path)
    signals_df['stage'] = signals_df['stage_expert_0']
    signals_df['arousal'] = signals_df['arousal_expert_0']
    signals_df['resp'] = signals_df['resp_expert_0']
    del_cols = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'e2-m1', 'chin1-chin2', 
                    'ecg', 'lat', 'rat', 'stage_expert_0', 'arousal_expert_0', 'resp_expert_0', 'limb_expert_0']
    signals_df = signals_df.drop(columns=del_cols)

    # determine CPAP on
    signals_df['cpap_on'] = signals_df.cpres.rolling(int(120*original_Fs*2), center=True, min_periods=1)\
                                            .median().round(decimals=1) > 0
    signals_df['cpap_on'] = signals_df['cpap_on'].astype(int)

    # apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)

    # Select breathing trace based on morphology
    br_trace, split_loc = cpapOn_select_breathing_trace(signals_df)
    print(f'we keep >{br_trace}< based on CPAP ON')

    # drop cpap_on channel
    if 'cpap_on' in signals_df.columns: signals_df.drop(columns=['cpap_on'], inplace=True)

    # clip / normalize
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # set breathing trace
    if split_loc is None:
        signals_df['breathing_trace'] = signals_df[br_trace].values
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    # setup Header & DF
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)

    return signals_df, hdr

def load_data_from_prepared_dataset_MROS(path, original_Fs=200, new_Fs=10, signals=[], show=True):
    # read in signals
    signals_df = load_prepared_data(path)
    signals_df['stage'] = signals_df['stage_expert_0']
    signals_df['arousal'] = signals_df['arousal_expert_0']
    signals_df['resp'] = signals_df['resp-h3_expert_0']
    del_cols = ['chin1-chin2', 'e1-m2', 'ecg', 'leg l', 'leg r', 'c3-m2', 'c4-m1']
    del_cols += [c for c in signals_df.columns if 'limb' in c]
    signals_df = signals_df.drop(columns=del_cols)

    # first select breathing trace based on morphology
    br_trace, split_loc = morphology_select_breathing_trace(signals_df, new_Fs)

    # apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)
    # correct split loc
    if type(split_loc) == int: split_loc = int(split_loc*new_Fs/original_Fs)

    # clip / normalize
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # set breathing trace
    if split_loc is None:
        signals_df['breathing_trace'] = signals_df[br_trace].values
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    # setup Header & DF
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)
    if show:
        print(hdr['test_type'] + ' study (based on "signals morphology")')

    return signals_df, hdr

def load_data_from_prepared_dataset_MESA(path, original_Fs=200, new_Fs=10, signals=[], show=True):
    # read in signals
    signals_df = load_prepared_data(path)
    signals_df['stage'] = signals_df['stage_expert_0']
    signals_df['arousal'] = signals_df['arousal_expert_0']
    signals_df['resp'] = signals_df['resp-h3_expert_0']
    del_cols = ['chin1-chin2', 'e1-m2', 'ecg', 'leg', 'c4-m1']
    del_cols += [c for c in signals_df.columns if 'limb' in c]
    signals_df = signals_df.drop(columns=del_cols)

    # first select breathing trace based on morphology
    br_trace, split_loc = morphology_select_breathing_trace(signals_df, new_Fs)

    # apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)
    # correct split loc
    if type(split_loc) == int: split_loc = int(split_loc*new_Fs/original_Fs)

    # clip / normalize
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # set breathing trace
    if split_loc is None:
        signals_df['breathing_trace'] = signals_df[br_trace].values
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    # setup Header & DF
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)
    if show:
        print(hdr['test_type'] + ' study (based on "signals morphology")')

    return signals_df, hdr

def load_data_from_prepared_dataset_SHHS(path, original_Fs=200, new_Fs=10, signals=[]):
    # read in signals
    signals_df = load_prepared_data(path)
    signals_df['stage'] = signals_df['stage_expert_0']
    signals_df['arousal'] = signals_df['arousal_expert_0']
    signals_df['resp'] = signals_df['resp-h3_expert_0']
    del_cols = ['chin1-chin2', 'e1-m2', 'ecg', 'c3-m2', 'c4-m1']
    del_cols += [c for c in signals_df.columns if 'limb' in c or 'expert' in c]
    signals_df = signals_df.drop(columns=del_cols)

    if np.all(signals_df[['ptaf', 'cflow']]==0):
        abd = data['ABD'].rolling(int(0.5*original_Fs), center=True).median().fillna(0)
        chest = data['CHEST'].rolling(int(0.5*original_Fs), center=True).median().fillna(0) 
        data['ptaf'] = abd + chest
        br_trace, split_loc = ['ptaf'], None
        print('Abd + Chest is used as breathing trace (replacing ptaf)')
    else:
        # first select breathing trace based on morphology
        br_trace, split_loc = morphology_select_breathing_trace(signals_df, new_Fs)

    # apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)
    # correct split loc
    if type(split_loc) == int: split_loc = int(split_loc*new_Fs/original_Fs)

    # clip / normalize
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # set breathing trace
    if split_loc is None:
        signals_df['breathing_trace'] = signals_df[br_trace].values
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    # setup Header & DF
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)
    print(hdr['test_type'] + ' study (based on "morphology")')

    return signals_df, hdr