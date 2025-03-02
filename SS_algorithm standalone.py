import os, glob, mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.signal import savgol_filter, convolve, find_peaks
import warnings
warnings.filterwarnings("ignore")

def show_credits():
    # header
    print("\n\n*** Notice: Ownership and Patent Information ***")
    credit = 'This program, including its underlying algorithms and methodologies '

    # names
    credit += 'is owned and patented by: T. Nassi, E. Oppersma, M.B. Westover, and R.J. Thomas. '
    # credit += 'Beth Israel Deaconess Medical Center and the University of Twente. '

    # warning
    credit += 'Any unauthorized use, reproduction, or distribution of this program '
    credit += 'or its components is strictly prohibited and may result in legal action. '
    
    # info
    credit += 'For licensing information, please contact the corresponding author (R.J. Thomas).\n'

    # end
    credit += '---\n'

    print(credit)
    
def list_files(edf_folder):
    # get all edf paths in this folder
    edf_paths = glob.glob(os.path.join(edf_folder, '*.edf'))  
  
    # if result folder does not exist, create
    results_dir = 'Results'
    os.makedirs(results_dir, exist_ok=True)

    return edf_paths, results_dir

# Loading Functions
def multiple_channel_search(edf):
    # set respiratory channels
    edf_channels = edf.info['ch_names']
    search_channels = {
        'npt': 'ptaf',
        'n press': 'ptaf',
        'therm': 'airflow',
        'flow': 'airflow',
        'abd': 'abd',
        'chest': 'chest',
        'thorax': 'chest',
        'sum effort': 'effort',
        'effort': 'effort',
    }
    
    # run over all search channels
    found_channels, columns = [], []
    for ch in search_channels.keys():
        # skip if already found channel
        if search_channels[ch] in columns: continue
        # run over all edf channels
        for c in edf_channels:
            if ch in c.lower():
                found_channels.append(c)
                columns.append(search_channels[ch])
                break

    # pick effort trace
    signals = edf.get_data(picks=found_channels) 
    data = pd.DataFrame(np.squeeze(signals).T, columns=columns)

    return data, found_channels

def original_effort_search(edf):
    # find respiratory effort channels
    respiratory_effort_channels = ['Abdomen', 'Abdominal', 'Chest', 'ABDOMEN', 'CHEST', 'ABDOMINAL', 'Effort THO']
    if not sum(np.isin(respiratory_effort_channels, edf_channels)) > 0:
        print(f'{edf_path}:')
        print(f'Expected Channel Name not found. Try flexible search for anything with "effort", "chest", "tho", or "abd" in it.')
        respiratory_effort_channels = [x for x in edf_channels if any([y in x.lower() for y in ['effort', 'chest', 'tho', 'abd']])]
        if len(respiratory_effort_channels) > 0:
            print(f'Success. Channel used: {respiratory_effort_channels}')
    if not sum(np.isin(respiratory_effort_channels, edf_channels)) > 0:
        print(f'Code cannot be performed: No effort belt channel found in the EDF file. \nThe file contains:{edf_channels}.')
        raise Exception

    # pick effort trace
    respiratory_effort_channels = [x for x in respiratory_effort_channels if x in edf_channels]
    respiratory_effort_channels.sort()
    channel_name = 'abd' if 'abd' in respiratory_effort_channels[0].lower() else 'chest'
    signals = edf.get_data(picks=respiratory_effort_channels[0])  # signals.shape=(#channel, T)
    data = pd.DataFrame(np.squeeze(signals), columns=[channel_name])

    return data

# Class Preprocessing 
def remove_nans(data, add_rem=50):
	# find all rows including NaN's
	ignores = ['patient_asleep', 'Pleth', 'EEG_events_anno']
	drops = []
	for ig in ignores:
		if ig in data.columns:
			drops.append(ig)
	nan_array = np.array(data.drop(columns=drops).isna().any(axis=1)).astype('int')

	nans = np.argwhere(nan_array>0)[:,0]  

	# define shift
	shift = add_rem

	# run over all nan rows
	n = 0
	while n < len(nans)-1:
		nan = nans[n]
		beg = nan-shift if nan-shift >= 0 else 0
		end = nan+shift if nan+shift <= data.shape[0]+1 else data.shape[0]+1
		s = 1
		# skip all consequetive NaN's
		while nan+s == nans[n+s]:
			end = nan+shift+s if nan+shift+s <= data.shape[0]+1 else data.shape[0]+1
			s += 1
			if nan+s-1 == nans[-1] or n+s == len(nans):
				break
		n += s
		data.loc[beg:end, :] = None

	return data  

def window_correction(array, window_size):
	half_window = int(window_size/2)
	events = find_events(array)
	corr_array = np.array(array)
	
	# run over all events in array
	for st, end in events:
		label = array[st]
		corr_array[st-half_window-1: end+half_window] = label

	return corr_array.astype(int) 

def find_events(signal):
	# ensure np array type
	signal = np.array(signal)
		
	# ini lists
	starts, ends = [], []
	# add start if array starts with an event
	if signal[0] > 0:
		starts.insert(0,0)

	# compute diff of channel
	diff_drops = pd.DataFrame(signal).diff()

	# find starts and ends of events
	starts, ends = define_events_start_ends(signal, diff_drops, starts, ends)

	# add last ind if event did not end
	if signal[-1] > 0:
		ends.append(diff_drops.shape[0])
		signal[-1] = 0

	# check basic conditions
	if len(ends) == len(starts) == 0:
		return []
		
	assert len(ends) == len(starts), 'ERROR in <method> find_events'

	# zip start with ends
	grouped_events = list(zip(starts,ends))
	
	return grouped_events

def define_events_start_ends(signal, diff_drops, starts, ends):
	for v in np.where(diff_drops[1:])[0]:
		loc = v + 1

		step = signal[loc]
		step_min_one = signal[loc-1]

		if step > step_min_one:
			starts.append(loc)
		elif step < step_min_one:
			ends.append(loc)

	return starts, ends

def label_correction(starts, ends, signal, Fs):
	# run over all found merged events
	merged_locs = search_for_merged_labels(signal)
	for p, loc, n in merged_locs:
		# split the two events
		event1 = signal[loc-1]
		event2 = signal[loc+1]
		# check wich event has priority
		priority_loc = np.argmin([event1, event2])
		if priority_loc == 0:
			# remove second start when priority = event1
			starts = [s for s in starts if p != s]
			# and convert second end into first end
			w = np.where(ends==n)[0]
			if len(w) > 0:
				ends[w[0]] = loc
			# print('converted merged event >%s-%s< to >%s-X< at loc=%s'%(event1, event2, event1, loc))
		else:
			# remove first end when priority = event2
			ends = [e for e in ends if loc != e]
			# and convert first start into second start
			starts[np.where(starts==p)[0][0]] = loc
			# print('converted merged event >%s-%s< to >X-%s< at loc=%s'%(event1, event2, event2, loc))

	return starts, ends

def events_to_array(events, len_array, labels=[]):
	array = np.zeros(len_array)
	if len(labels)==0: labels = [1]*len(events)
	for i, (st, end) in enumerate(events):
		array[st:end] = labels[i]
		
	return array


# More preprocessing functions
def cut_flat_signal(data, fs):
    # find start location of common flat trailing data
    all_flat = np.all(data.rolling(2).std().values==0, 1)
    if not all_flat[-1]:
        return data
    
    # cut trailing flat signal
    cut = np.where(all_flat==False)[0][-1]
    cut = np.min((cut + 5*60*fs, len(data)))

    return data.iloc[:cut]

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
                        'abd', 'chest', 'effort', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
            image = notch_filter(image.astype(float), original_Fs, notch_freq_us, verbose=False)
            # image = notch_filter(image, 200, notch_freq_eur, verbose=False)

        # 2. Bandpass filter
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2']:
            image = filter_data(image, original_Fs, bandpass_freq_eeg[0], bandpass_freq_eeg[1], verbose=False)
        if sig in ['abd', 'chest', 'effort', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            image = filter_data(image, original_Fs, bandpass_freq_airflow[0], bandpass_freq_airflow[1], verbose=False)
        if sig == 'ecg':
            image = filter_data(image, original_Fs, bandpass_freq_ecg[0], bandpass_freq_ecg[1], verbose=False)

        # 3. Resample data
        if new_Fs != original_Fs:
            if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 
                            'abd', 'chest', 'effort', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
                image = resample_poly(image, new_Fs, original_Fs)
            else:
                image = np.repeat(image, new_Fs)
                image = image[::original_Fs]                

        # 4. Insert in new DataFrame
        new_df.loc[:, sig] = image
    
    del signals
    return new_df

def clip_normalize_signals(signals, sample_rate, min_max_times_global_iqr=20):
    # run over all channels
    for chan in signals.columns:
        # skip labels
        if np.any([t in chan for t in ['stage', 'arousal', 'resp', 'cpap_pressure', 'cpap_on']]): continue
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
        elif chan in ['abd', 'chest', 'effort', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            # only select region with valid signal
            region = np.arange(len(signal)) # use entire signal

            # skip if all zeros
            if np.all(signal[region]==0): continue
            
            # normalize signal
            signal_clipped = np.clip(signal, np.nanpercentile(signal[region],5), np.nanpercentile(signal[region],95))
            signal_normalized = np.array((signal - np.nanmean(signal_clipped)) / np.nanstd(signal_clipped))
            
            # clip extreme values
            quan = 0.2
            factor = 20
            clp = 0.001 
            if chan in ['abd', 'chest', 'effort']:
                clp = 0.01
                factor = 10
            elif chan == 'ptaf':
                factor = 30
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


# Flow reduction analysis
def assess_ventilation(data, hdr, drop_hyp, drop_apnea, dur_apnea, dur_hyp, quant, extra_smooth=False):
    # compute dynamic excursion threshold, both for apnea and hypopneas.
    Fs = hdr['newFs']
    excursion_duration = 60     # the larger the interval, the less dynamic a baseline gets computed.
    excursion_q = quant         # ventilation envelope quantile

    # use lagging moving windonw, events are found based on future eupnea / recovery breaths
    pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
    pos_excursion[:-excursion_duration*Fs*2] = pos_excursion[excursion_duration*Fs*2:]
    pos_excursion[-excursion_duration*Fs*2:] = np.nan
    neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values
    neg_excursion[:-excursion_duration*Fs*2] = neg_excursion[excursion_duration*Fs*2:]
    neg_excursion[-excursion_duration*Fs*2:] = np.nan

    # add additional envelope smoothing
    if extra_smooth:
        minutes = 20
        win = int(Fs*60*minutes)
        pos = pd.DataFrame(data=pos_excursion).rolling(win, center=True, min_periods=1).quantile(0.4).values
        neg = pd.DataFrame(data=neg_excursion).rolling(win, center=True, min_periods=1).quantile(0.6).values
        pos_excursion = np.squeeze(pos)
        neg_excursion = np.squeeze(neg)  
        # compute smoothed uncalibrated apnea excursion
        middle = np.mean([pos_excursion, neg_excursion], 0)
        pos_distance_to_baseline = np.abs(pos_excursion-middle)
        neg_distance_to_baseline = np.abs(neg_excursion-middle)

    # Relative pos/neg excursion (Hypopnneas)
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_baseline'])
    data['pos_excursion_hyp'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_hyp))
    data['neg_excursion_hyp'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_hyp))
    ### add soft hypopneas ###
    data['pos_excursion_soft_hyp'] = data['Ventilation_default_baseline'] + (pos_distance_to_baseline * (1-drop_hyp/1.02))
    data['neg_excursion_soft_hyp'] = data['Ventilation_default_baseline'] - (neg_distance_to_baseline * (1-drop_hyp/1.02))

    # Average pos/neg excursion (Apneas)
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_default_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_default_baseline'])
    dist_to_baseline = np.mean([pos_distance_to_baseline, neg_distance_to_baseline], 0)
    data['pos_excursion_apnea'] = data['Ventilation_default_baseline'] + (dist_to_baseline * (1-drop_apnea))
    data['neg_excursion_apnea'] = data['Ventilation_default_baseline'] - (dist_to_baseline * (1-drop_apnea))

    # find drops in ventilation signal for apneas and hypopneas
    data = locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp)

    # combine positive and negative excursion flow limitations
    data = pos_neg_excursion_combinations(data, hdr)

    if False:
        plt.figure(figsize=(9.5,6)) 

        # signal and baseline
        plt.plot(data.Ventilation_combined.mask(data.patient_asleep==0),'y', lw=0.5, alpha=0.5)
        plt.plot(data.Ventilation_combined.mask(data.patient_asleep==1),'r', lw=0.5, alpha=0.5)
        plt.plot(data.Ventilation_baseline.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'k', lw=0.5)
        plt.plot(data.Ventilation_default_baseline.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'k--', lw=0.5)
        # envelopes
        plt.plot(pos_excursion,'b--')
        plt.plot(neg_excursion,'b--')
        
        # Main apnea / hypopnea threshold lines
        for col, c, lw in zip(['excursion_apnea', 'excursion_hyp', 'excursion_soft_hyp'], ['r', 'm'], [0.8, 0.5]):
            for pn in ['pos_', 'neg_']:
                plt.plot(data[pn+col].mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)), c, lw=lw)
        # Secondary version
        for col, c in zip(['excursion_apnea1', 'excursion_hyp1'], ['r', 'm']):
            for pn in ['pos_', 'neg_']:
                if not pn+col in data.columns: continue
                plt.plot(data[pn+col].mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),c+'--', lw=0.5)
        
        # Labels 
        plt.plot(-data.Ventilation_drop_apnea.mask(data.Ventilation_drop_apnea!=1), 'g', lw=2)
        plt.plot(-4*data.Ventilation_drop_hypopnea.mask(data.Ventilation_drop_hypopnea!=1), 'm', lw=2)
        plt.plot(-3.5*data.pos_Ventilation_drop_hypopnea.mask(data.pos_Ventilation_drop_hypopnea!=1), 'k')
        plt.plot(-4.5*data.neg_Ventilation_drop_hypopnea.mask(data.neg_Ventilation_drop_hypopnea!=1), 'k')
        plt.show()
        import pdb; pdb.set_trace()

    return data

def remove_non_forward_drops(data, hdr, drop_hyp, drop_apnea, dur_apnea, dur_hyp, quant, extra_smooth=False):
    # compute dynamic excursion threshold, both for apnea and hypopneas.
    Fs = hdr['newFs']
    excursion_duration = 60     # the larger the interval, the less dynamic a baseline gets computed.
    excursion_q = quant         # ventilation envelope quantile

    # use lagging moving windonw, events are found based on future eupnea / recovery breaths
    pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
    neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values

    # add additional envelope smoothing
    if extra_smooth:
        minutes = 5
        win = int(Fs*60*minutes)
        pos = pd.DataFrame(data=pos_excursion).rolling(win, center=True, min_periods=1).quantile(0.4).values
        neg = pd.DataFrame(data=neg_excursion).rolling(win, center=True, min_periods=1).quantile(0.6).values
        pos_excursion = np.squeeze(pos)
        neg_excursion = np.squeeze(neg)  

    # set excursion into DF
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_baseline'])
    data['pos_excursion_apnea'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_apnea))
    data['pos_excursion_hyp'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_hyp))
    data['neg_excursion_apnea'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_apnea))
    data['neg_excursion_hyp'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_hyp))
    ### add soft hypopneas ###
    data['pos_excursion_soft_hyp'] = data['Ventilation_default_baseline'] + (pos_distance_to_baseline * (1-drop_hyp/1.02))
    data['neg_excursion_soft_hyp'] = data['Ventilation_default_baseline'] - (neg_distance_to_baseline * (1-drop_hyp/1.02))
   
    # find drops in ventilation signal for apneas and hypopneas
    data = locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp)

    apnea_cols = [c for c in data.columns if '_Ventilation_drop_apnea' in c]
    hyp_cols = [c for c in data.columns if '_Ventilation_drop_hypopnea' in c]
    soft_hyp_cols = [c for c in data.columns if '_Ventilation_drop_soft_hypopnea' in c]
    
    # run over all found apneas
    for st, end in find_events(data['Ventilation_drop_apnea']>0):
        # if no apnea is found in either trace, remove apnea
        if np.all(data.loc[st:end, apnea_cols]==0):
            data.loc[st:end, 'Ventilation_drop_apnea'] = 0
            # if hypopnea is found, replace apnea by hypopnea
            if np.any(data.loc[st:end, hyp_cols]>0):
                data.loc[st:end, 'Ventilation_drop_hypopnea'] = 1

    # run over all found hypopneas
    for st, end in find_events(data['Ventilation_drop_hypopnea']>0):
        # remove when no forward hypopnea is found
        if np.all(data.loc[st:end, hyp_cols]==0):
            data.loc[st:end, 'Ventilation_drop_hypopnea'] = 0

    # run over all found soft hypopneas
    for st, end in find_events(data['Ventilation_drop_soft_hypopnea']>0):
        # remove when no forward hypopnea is found
        if np.all(data.loc[st:end, soft_hyp_cols+hyp_cols]==0):
            data.loc[st:end, 'Ventilation_drop_soft_hypopnea'] = 0

    return data

def locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp):
    # remove NaN values only for selected channels -->
    selected_columns = ['pos_excursion_apnea', 'pos_excursion_hyp', 'pos_excursion_soft_hyp',
                        'neg_excursion_apnea', 'neg_excursion_hyp', 'neg_excursion_soft_hyp',
                        'Ventilation_baseline']
    new_df = data[selected_columns + ['Ventilation_combined']].copy()

    # put selected columns in original dataframe
    for col in selected_columns:
        data[col] = new_df[col]

    # *add smoothed exursion thresholds*
    win = int(hdr['newFs']*60*10)
    data['pos_excursion_apnea_smooth'] = data['pos_excursion_apnea'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_apnea_smooth'] = data['neg_excursion_apnea'].rolling(win, center=True, min_periods=1).median()
    data['pos_excursion_hyp_smooth'] = data['pos_excursion_hyp'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_hyp_smooth'] = data['neg_excursion_hyp'].rolling(win, center=True, min_periods=1).median()
    data['pos_excursion_soft_hyp_smooth'] = data['pos_excursion_soft_hyp'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_soft_hyp_smooth'] = data['neg_excursion_soft_hyp'].rolling(win, center=True, min_periods=1).median()

    # find areas with potential apnea / hypopnea flow limitations
    sig = data.Ventilation_combined
    data['pos_Ventilation_drop_apnea'] = np.logical_or(sig<data.pos_excursion_apnea, sig<data.pos_excursion_apnea_smooth)
    data['neg_Ventilation_drop_apnea'] = np.logical_or(sig>data.neg_excursion_apnea, sig>data.neg_excursion_apnea_smooth)
    data['pos_Ventilation_drop_hypopnea'] = np.logical_or(sig<data.pos_excursion_hyp, sig<data.pos_excursion_hyp_smooth)
    data['neg_Ventilation_drop_hypopnea'] = np.logical_or(sig>data.neg_excursion_hyp, sig>data.neg_excursion_hyp_smooth)
    data['pos_Ventilation_drop_soft_hypopnea'] = np.logical_or(sig<data.pos_excursion_soft_hyp, sig<data.pos_excursion_soft_hyp_smooth)
    data['neg_Ventilation_drop_soft_hypopnea'] = np.logical_or(sig>data.neg_excursion_soft_hyp, sig>data.neg_excursion_soft_hyp_smooth)

    # run over the various ventilation drop options, and find flow limitations
    data['either_hypes'] = 0  
    tag_window = [dur_apnea*0.8, dur_apnea, dur_hyp, dur_hyp]
    for ex in ['pos', 'neg']:
        tag_list = [f'{ex}_soft_ventilation_drop_apnea', 
                    f'{ex}_Ventilation_drop_apnea', 
                    f'{ex}_Ventilation_drop_hypopnea',
                    f'{ex}_Ventilation_drop_soft_hypopnea']
        data = find_flow_limitations(data, tag_list, tag_window, hdr['newFs'])

    return data

def find_flow_limitations(data, tag_list, tag_window, Fs):
    for t, tag in enumerate(tag_list):
        win = int(tag_window[t]*Fs)
        # find events with duration <win>
        if t==0:
            col = [c for c in tag_list if 'Ventilation_drop_apnea' in c]
            data[tag] = data[col].rolling(win, center=True).mean() > 0.75 # allow for small peaks exceeding threshold
        else:
            data[tag] = data[tag].rolling(win, center=True).mean() == 1 # <win> should stay below threshold

        # apply window correction
        data[tag] = np.array(data[tag].fillna(0))
        cut = Fs//2 if 'hypopnea' in tag else 0 # slightly shorten event for Hypopneas
        data[tag] = window_correction(data[tag], window_size=win-cut)

        # remove all apnea events with a duration > .. sec
        max_dur = 150 if 'hypopnea' in tag else 120
        data = remove_long_events(data, tag, Fs, max_duration=max_dur)

    return data

def pos_neg_excursion_combinations(data, hdr):
    # run over apnea options
    apnea_cols = ['Ventilation_drop_apnea', 'soft_ventilation_drop_apnea']
    hypopnea_cols = ['Ventilation_drop_hypopnea', 'Ventilation_drop_soft_hypopnea']
    for col in apnea_cols + hypopnea_cols:
        # for soft apneas, pos and neg flow limitation has to occur simultaniously
        if 'soft_ventilation' in col:
            data[col] = (data['pos_%s'%col] * data['neg_%s'%col]) > 0
        # for hypopneas, either positive or negative flow limitation is saved (prioiritize pos) 
        elif 'hypopnea' in col:
            data = hyp_flow_limitations(data, col)
        # for apneas, pos and neg flow limitation has to occur simultaniously
        else:
            data[col] = (data['pos_%s'%col] * data['neg_%s'%col]) > 0
            # connect apneas if pos or neg criteria continues
            data[col] = connect_apneas(data, col, 10, hdr['newFs'], max_dur=120)
            
        # connect events, within 5 sec (only if total event < 20sec)
        events = find_events(data[col].fillna(0)>0)
        if len(events) == 0: continue
        events, _ = connect_events(events, 3, hdr['newFs'], max_dur=20)
        data[col] = events_to_array(events, len(data))
        
        # remove events < 4 sec
        data[col] = remove_short_events(data[col], 4*hdr['newFs'])

    # remove soft ventilation drop apnea, if apnea found
    for st, end in find_events(data['soft_ventilation_drop_apnea']>0):
        if np.any(data.loc[st:end, 'Ventilation_drop_apnea']==1):
            data.loc[st:end, 'soft_ventilation_drop_apnea'] = 0
    # remove soft ventilation drop hypopnea, if apnea/hypopnea found
    for st, end in find_events(data['Ventilation_drop_soft_hypopnea']>0):
        if np.any(data.loc[st:end, ['Ventilation_drop_apnea', 'Ventilation_drop_hypopnea']]==1):
            data.loc[st:end, 'Ventilation_drop_soft_hypopnea'] = 0

    return data

def hyp_flow_limitations(data, col):
    data[col] = data[f'pos_{col}'].values
    neg = find_events(data[f'neg_{col}'].fillna(0)>0)

    # run over flow limitation regions
    for st, end in neg:
        region = list(range(st, end))
        
        # skip if already saved in pos array
        if any(data.loc[region, f'pos_{col}']): continue

        # save in array
        data.loc[region, col] = 1

    return data

def combine_flow_reductions(data, hdr):
    # set data arrays
    Fs = hdr['newFs']
    apneas = data.Ventilation_drop_apnea
    hypopneas = data.Ventilation_drop_hypopnea
    grouped_hypopneas = find_events(hypopneas>0)
    data['flow_reductions'] = apneas

    # add hypopneas to apnea array
    for st, end in grouped_hypopneas:
        region = list(range(st, end))
        # insert when no apnea is found in that region
        if np.all(apneas[region] == 0):
            data.loc[region, 'flow_reductions'] = 2
        else:
            reg = region[2*Fs:-2*Fs]
            if len(reg)<10*Fs: continue
            if np.all(apneas[reg] == 0):
                data.loc[reg, 'flow_reductions'] = 2
    
    return data

def connect_apneas(data, col, win, Fs, max_dur=False):
    # set events
    events = find_events(data[col].fillna(0)>0)
    if len(events) == 0: return data[col].values

    # connect apneas if pos or neg negative treshold remains
    new_events = []
    cnt = 0
    win = win*Fs
    while cnt < len(events)-1:
        st = events[cnt][0]
        end = events[cnt][1]
        dist1 = events[cnt+1][0] - end 
        condition1 = (dist1<win) if max_dur == False else (dist1<win) and ((events[cnt+1][1]-st) < max_dur*Fs)
        condition2 = any(np.all(data.loc[st:end+dist1, [f'pos_{col}', f'neg_{col}']] == 1, 0))
        if condition1 and condition2:      
            new_events.append((st, events[cnt+1][1]))
            cnt += 2
        else:
            new_events.append((st, end))
            cnt += 1  
    new_events.append((events[-1]))

    # convert back to array
    new_array = events_to_array(new_events, len(data))

    return new_array

##
def connect_events(events, win, Fs, max_dur=False, labels=[]):
    new_events, new_labels = [], []
    if len(events) > 0 :
        cnt = 0
        win = win*Fs
        if len(labels)==0: labels = [1]*len(events)
        while cnt < len(events)-1:
            st = events[cnt][0]
            end = events[cnt][1]
            dist1 = events[cnt+1][0] - end 
            condition1 = (dist1<win) if max_dur == False else (dist1<win) and ((events[cnt+1][1]-st) < max_dur*Fs)
            if condition1:            
                new_events.append((st, events[cnt+1][1]))
                lab = labels[cnt] if end-st > events[cnt+1][1]-events[cnt+1][0] else labels[cnt+1]
                new_labels.append(lab)
                cnt += 2
            else:
                new_events.append((st, end))
                new_labels.append(labels[cnt])
                cnt += 1  
        new_events.append((events[-1]))
        new_labels.append((labels[-1]))

    return new_events, new_labels

def merge_small_events(data, Fs):
    # define global hypopnea flow reduction threshold
    global_hyp_thresh = np.nanmedian(data['pos_excursion_hyp'])
    while True:
        # run over all flow reductions
        all_flow_reductions = find_events(data['flow_reductions']>0)
        if len(all_flow_reductions)<2: return data
        for i, (st, end) in enumerate(all_flow_reductions[:-1]):
            next_st = all_flow_reductions[i+1][0]
            next_end = all_flow_reductions[i+1][1]
            ss = int(np.median((st, end)))
            ee = int(np.median((next_st, next_end)))
            region = list(range(ss, ee))
            # skip events that would become >2min
            if (next_end-st) > 60*Fs: continue

            # if ventilation trace stays below local or global hyp threshold, merge events (by filling inbetween region)
            if len(data.loc[region, 'pos_excursion_hyp'].dropna()) == 0: continue
            local_thresh = np.nanmedian(data.loc[region, 'pos_excursion_hyp'])
            local_trace = data.loc[region, 'Ventilation_combined']
            check1 = np.sum(local_trace < local_thresh) > 0.9*len(region)
            check2 = np.sum(local_trace < global_hyp_thresh) > 0.9*len(region)
            if check1 or check2:
                # only if there is intermittant sleep
                if not np.any(data.loc[list(range(end, next_st)), 'Stage'] == 5):
                    vals, cnts = np.unique(data.loc[region, 'flow_reductions'], return_counts=True)
                    num = 1 if len(vals[1:]) == 1 and vals[1] == 1 else 2
                    data.loc[list(range(st, next_end)), 'flow_reductions'] = num
                    break
        if i == len(all_flow_reductions)-2: 
            break
        
    return data

def remove_wake_events(data):
    # run over all flow reductions
    all_flow_reductions = find_events(data['flow_reductions']>0)
    for i, (st, end) in enumerate(all_flow_reductions[:-1]):
        region = list(range(st, end))
        if np.sum(data.loc[region, 'patient_asleep']==0) > 0.75*len(region):
            data.loc[region, 'flow_reductions'] = 0

    return data


# Post-processing 
def remove_long_events(data, tag, Fs, max_duration=60):
    data['too_long_events'] = 0
    # find and remove events with duration > 'max_duration'
    events = [ev for ev in find_events(data[tag]>0) if ev[1]-ev[0] > max_duration*Fs]
    for st, end in events:
        region = list(range(st, end))
        data.loc[region, 'too_long_events'] = 1
        # fill only part associated with sat-drop or arousal
        if tag == 'algo_apneas' and data.loc[st, 'algo_apneas']==2:
            data.loc[region, 'algo_apneas'] = 0
            # fill based on single saturation drop found
            drops = [a for a in find_events(data.loc[region, 'saturation_drop']>0) if a[0] > 10*Fs]
            arousals = [a for a in find_events(data.loc[region, 'EEG_arousals']>0) if a[0] > 10*Fs]
            if len(drops) > 0:
                if len(drops) != 1: continue
                loc = int(st+np.mean(drops[0]))
                fill = list(range(loc-30*Fs, loc))
            # otherwise do based on arousals
            elif len(arousals) > 0:
                if len(arousals) != 1: continue
                loc = int(st+np.mean(arousals[0]))
                fill = list(range(loc-30*Fs, loc))
            # if none found, don't fill
            else: continue
            # fill based on location
            data.loc[fill, 'algo_apneas'] = 2
        else:
            # remove region
            data.loc[region, tag] = 0

    return data

def remove_short_events(array, duration):
    # find and remove events with duration < 'duration'
    array = np.array(array)
    for st, end in find_events(array>0): 
        region = list(range(st, end))
        if len(region) < duration:
            array[region] = 0

    return array


# Envelope analysis
def compute_envelopes(data, Fs, channels='abd'):
    # set ABD trace to ventilation combined
    data['Ventilation_combined'] = data[channels] if type(channels) != list else data[channels].mean(axis=1)

    # compute envelope and baseline
    new_df = compute_envelope(data['Ventilation_combined'], int(Fs), env_smooth=5)
    data.loc[:, 'Ventilation_pos_envelope'] = new_df['pos_envelope'].values
    data.loc[:, 'Ventilation_neg_envelope'] = new_df['neg_envelope'].values
    data.loc[:, 'Ventilation_default_baseline'] = new_df['baseline'].values
    data.loc[:, 'Ventilation_baseline'] = new_df['correction_baseline'].values

    return data

def compute_envelope(signal, Fs, base_win=30, env_smooth=5):
    new_df = pd.DataFrame()
    new_df['x'] = signal

    # determine peaks of signal
    x = new_df['x']
    pos_peaks, _ = find_peaks(x, distance=int(Fs*1.5), width=int(0.4*Fs), rel_height=1)
    neg_peaks, _ = find_peaks(-x, distance=int(Fs*1.5), width=int(0.4*Fs), rel_height=1)

    new_df['pos_envelope'] = x[x.index[0] + pos_peaks]
    new_df['neg_envelope'] = x[x.index[0] + neg_peaks]

    # compute envelope of signal
    new_df['pos_envelope'] = new_df['pos_envelope'].interpolate(method='cubic', order=1, limit_area='inside')
    new_df['neg_envelope'] = new_df['neg_envelope'].interpolate(method='cubic', order=1, limit_area='inside')        
    new_df['pos_envelope'] = new_df['pos_envelope'].rolling(env_smooth*Fs, center=True).median()
    new_df['neg_envelope'] = new_df['neg_envelope'].rolling(env_smooth*Fs, center=True).median()
    check_invalids = new_df['pos_envelope'] < new_df['neg_envelope']
    new_df.loc[check_invalids, 'pos_envelope'] = 0
    new_df.loc[check_invalids, 'neg_envelope'] = 0
    
    new_df['baseline'], new_df['baseline2'], new_df['correction_baseline'] = compute_baseline(new_df, Fs, base_win)

    return new_df

def compute_baseline(new_df, Fs, base_win, correction_ratio=2):
    # compute baseline of signal
    pos = new_df['pos_envelope'].rolling(base_win*Fs, center=True).mean()
    neg = new_df['neg_envelope'].rolling(base_win*Fs, center=True).mean()
    base = (pos + neg) / 2

    base1 = new_df['x'].rolling(base_win*Fs, center=True).median().rolling(base_win*Fs, center=True).mean()
    base2 = base.rolling(base_win*Fs, center=True).mean()

    base_corr = (correction_ratio*base1 + base2) / (1+correction_ratio)
    base_corr = base_corr.rolling(base_win*Fs, center=True).mean()
    
    return base1, base2, base_corr

def compute_smooth_envelope(data, region):
    # analyze the two envelope traces
    for env_tag in ['Smooth_pos_envelope', 'Smooth_neg_envelope']:
        # create smoothed envelope
        original_env = env_tag.replace('Smooth', 'Ventilation')
        data.loc[region, env_tag] = savgol_filter(data.loc[region, original_env], 51, 1)

    return data


# Self-Similarity 
def assess_potential_self_sim_spots(data, Fs):
    # binarize labels
    data['Smooth_pos_envelope'] = 0
    data['Smooth_neg_envelope'] = 0
    data['TAGGED'] = 0

    # run over each potential self similarity region
    for self_sim_st, self_sim_end in find_events(data.potential_self_sim.values>0):
        # get all flow reductions in self similarity region
        self_sim_region = list(range(self_sim_st, self_sim_end))
        flow_lims = find_events(data.loc[self_sim_region, 'flow_reductions']>0)

        for i, (st, end) in enumerate(flow_lims[:-1]):
            # define corrected event locations
            st, end = self_sim_st + st, self_sim_st + end
            next_start, next_end = self_sim_st + flow_lims[i+1][0], self_sim_st + flow_lims[i+1][1]

            # set search regions
            region1 = list(range(st, end))
            region2 = list(range(next_start, next_end))
            region_full = list(range(int(np.median((st, end))), int(np.median((next_start, next_end)))))

            # compute envelope + cycle
            data = compute_smooth_envelope(data, region_full)

            # determine cycle spots
            cycle = find_cycle_spots(data, [region1, region2])

            # compare 1st half to 2nd half of smooth envelope
            conv_scores = convolve_envelope(data, cycle, Fs)

            # apply self-sim tests
            data, tests = do_self_sim_tests(data, cycle, conv_scores, Fs)

    return data

def tag_potential_self_sim_spots(data, Fs):
    data['potential_self_sim'] = 0
    labels = np.array(data.flow_reductions.fillna(0) > 0).astype(int)

    epoch_size = int(180*Fs)
    epoch_inds = np.arange(0, len(labels)-epoch_size+1, 5*Fs)
    seg_ids = list(map(lambda x:np.arange(x, x+epoch_size), epoch_inds))

    for seg_id in seg_ids:
        if len(find_events(labels[seg_id]>0)) >= 3:
            data.loc[seg_id, 'potential_self_sim'] = 1

    return data

def post_process_self_sim(data, Fs, SS_threshold):
    data['self similarity'] = 0
    data['consecutive complexes'] = 0
    data['ss_conv_score'] = np.nan

    # find 3 consecutive HLG-looking breathing oscillations    
    window = 180*Fs # use a sliding window with length <window>
    rolling_sum = data.TAGGED.rolling(window, center=True).sum()
    data.loc[rolling_sum>=3, 'consecutive complexes'] = 2
    data['consecutive complexes'] = window_correction(data['consecutive complexes'], window_size=window)

    # assess self-similarity for each three complexs
    for st, end in find_events(data['consecutive complexes']):
        complexes = find_events(data.loc[list(range(st,end)), 'TAGGED']) + st
        for t in range(len(complexes)-2):
            tags = complexes[t:t+3]
            conv_score = assess_three_breathing_oscillations(data, tags, Fs)
            num = 2 if conv_score >= SS_threshold else 1
            # save 'red' for self-similarity, 'black' for chains w/o self-similarity
            if t == 0:
                ss = tags[0][0]
                ee = tags[1][0] + (tags[2][0] - tags[1][0])//2
                data.loc[list(range(ss, ee)), 'self similarity'] = num
                data.loc[tags[0][0], 'ss_conv_score'] = conv_score
                data.loc[tags[1][0], 'ss_conv_score'] = conv_score
            if t == len(complexes)-3 or len(complexes)==3:
                ss = tags[0][0] + (tags[1][0] - tags[0][0])//2
                ee = tags[2][1]
                data.loc[tags[1][0], 'ss_conv_score'] = conv_score
                data.loc[tags[2][0], 'ss_conv_score'] = conv_score
            else:   
                ss = tags[0][0] + (tags[1][0] - tags[0][0])//2
                ee = tags[1][0] + (tags[2][0] - tags[1][0])//2
                data.loc[tags[1][0], 'ss_conv_score'] = conv_score
            
            data.loc[list(range(ss, ee)), 'self similarity'] = num

    # correct self-similarity array
    for st, end in find_events(data['self similarity']==2):
        if sum(data.loc[st:end, 'ss_conv_score'] > SS_threshold) < 3:
            data.loc[st:end, 'self similarity'] = 1

    # remove column
    data = data.drop(columns='consecutive complexes')

    return data

def find_cycle_spots(data, regions, iq=.1):
    # find bottoms in events
    cycle = []
    for i in range(2):
        # event_data = data.loc[regions[i], :]
        # mins = []
        # for env_tag in ['Smooth_pos_envelope', 'Smooth_neg_envelope']:
        #     if 'pos' in env_tag:
        #         thresh = event_data.loc[:, env_tag].quantile(iq)
        #         locs = np.where(event_data.loc[:, env_tag] < thresh)[0]
        #     elif 'neg' in env_tag:
        #         thresh = event_data.loc[:, env_tag].quantile(1-iq)
        #         locs = np.where(event_data.loc[:, env_tag] > thresh)[0]
        #     mins.append(locs[len(locs)//2] + event_data.index[0])
        # cycle.append(int(np.mean(mins)))
        cycle.append(regions[i][len(regions[i])//2])

    # find top inbetween
    local_data = data.loc[list(range(cycle[0], cycle[1])), :]
    top = np.argmax(local_data['Smooth_pos_envelope']) + local_data.index[0]
    bot = np.argmin(local_data['Smooth_neg_envelope']) + local_data.index[0]
    cycle.append(int(np.mean([top, bot])))

    return cycle

def convolve_envelope(data, cycle, Fs):
    conv_scores = np.zeros(3)

    # specify region of interest
    baseline = data.loc[cycle[0]:cycle[1], 'Ventilation_baseline'].values
    pos = data.loc[cycle[0]:cycle[1], 'Smooth_pos_envelope'].values - baseline
    neg = baseline - data.loc[cycle[0]:cycle[1], 'Smooth_neg_envelope'].values

    # normalize envelopes
    if not all(np.isnan(pos)):
        pos = (pos - np.nanmean(pos)) / (np.nanstd(pos) + 0.000001)
        neg = (neg - np.nanmean(neg)) / (np.nanstd(neg) + 0.000001)

    # apply convolution
    pos[np.isnan(pos)] = 0
    neg[np.isnan(neg)] = 0
    val = np.nanmax(convolve(pos, neg, mode='same')) / len(pos)
    conv_scores[2] = val

    return conv_scores

def assess_three_breathing_oscillations(data, tags, Fs):
    # retrieve envelopes from the three segments
    pos_envelopes, neg_envelopes = [], []
    for t in range(3):
        loc = tags[t][0]
        win = 20*Fs
        pos_envelopes.append(data.loc[list(range(loc-win, loc+win)), 'Smooth_pos_envelope'].values)
        neg_envelopes.append(data.loc[list(range(loc-win, loc+win)), 'Smooth_neg_envelope'].values)

    # compute convulion scores twice, previous and next oscillation
    conv_scores = []
    for oscillation, reference in [(pos_envelopes[1], pos_envelopes[0]), (pos_envelopes[1], pos_envelopes[2])]:
        # normalize envelopes
        first = (oscillation - np.mean(oscillation)) / (np.std(oscillation) + 0.000001)
        second = (reference - np.mean(reference)) / (np.std(reference) + 0.000001)

        # apply convolution
        conv_scores.append(np.nanmax(convolve(first, second, mode='same')) / len(first))

    return max(conv_scores)

def do_self_sim_tests(data, cycle, conv_scores, Fs):
    tests = {}

    # single peak
    d_t = (cycle[1] - cycle[0]) / Fs
    pass_test = True if d_t < 120 and d_t > 10 else False
    tests['duration test'] = (pass_test, d_t)
    
    # relative height test
    sig = data.loc[list(range(cycle[0],cycle[1])), 'Ventilation_combined'].values
    thresh = np.nanmean(data.loc[:, 'pos_excursion_hyp'].values)
    pass_test = True if np.any(sig > thresh) else False
    tests['relative height'] = (pass_test, int(pass_test))

    # peak timing score
    p_t = (1 - np.abs((cycle[2]-cycle[0]) - (cycle[1]-cycle[2])) / (cycle[1]-cycle[0])) * 100
    pass_test = True if p_t > 50 else False
    tests['peak timing'] = (pass_test, p_t)

    # vertical mirror score
    h_s = conv_scores[2] * 100
    pass_test = True if h_s > 50 else False
    tests['horizontal symmetry'] = (pass_test, h_s)

    if np.all([tests[key][0] for key in tests.keys()]):
        data.loc[cycle[2], 'TAGGED'] = 1

    return data, tests


# Central event count
def compute_central_events(flow_reductions, T_sim):
    # create central apnea/hypopnea cols
    central_dic = {}
    central_dic['central apneas'], central_dic['central hypopneas'] = 0, 0
    central_dic['central a'], central_dic['central h'] = np.zeros(len(T_sim)), np.zeros(len(T_sim))
    
    # run over all events
    i = 0
    events = find_events(flow_reductions>0)
    for st, end in events[:-1]:
        # set start and ends
        mid1 = int(np.mean([events[i][0], events[i][1]]))
        mid2 = int(np.mean([events[i+1][0], events[i+1][1]]))
        # if two events surround SS tag
        if any(T_sim[mid1:mid2] == 1):
            # save both events in their respective arrays
            for event in [events[i], events[i+1]]:
                val = flow_reductions[event[0]]
                tag = 'apneas' if val==1 else 'hypopneas'
                central_dic[f'central {tag[0]}'][event[0]:event[1]] = 1
                central_dic[f'central {tag}'] += 1
            i += 2
        else: i += 1
        if i >= len(events)-2: break

    return central_dic

def compute_main_channel(cols, channels):
    report_check = False
    for col, channel in zip(cols, channels):
        # perform main report/figure check
        check1 = 'effort' in cols and col=='effort'
        check2 = 'effort' not in cols and col=='abd'
        check3 = 'effort' not in cols and 'abd' not in cols and col =='chest'
        check4 = 'effort' not in cols and 'abd' not in cols and 'chest' not in cols and col=='ptaf'
        check5 = 'effort' not in cols and 'abd' not in cols and 'chest' not in cols and 'ptaf' not in cols and col=='airflow'
        if report_check==False and (check1 or check2 or check3 or check4 or check5):
            return (col, channel)
    
    return None

# Report
def create_report(output_data, hdr):
    # set sampling frequencies
    orinalFs = hdr['Fs']
    newFs = hdr['newFs']
    finalFs = 1

    # Init DF
    original_cols = ['flow_reductions', 'T_sim', 'TAGGED', 'ss_conv_score']
    data = pd.DataFrame([], columns=['second', 'start_idx', 'end_idx', 'SS'] + original_cols)
    
    # Resample data to 1 Hz
    for sig in original_cols: 
        image = np.repeat(output_data[sig].values , finalFs)
        image = image[::newFs]    
        # 4. Insert in new DataFrame
        data[sig] = image
    
    # save columns of interest
    factor = orinalFs // finalFs
    ind0 = np.arange(0, data.shape[0]) * factor
    ind1 = np.concatenate([ind0[1:], [ind0[-1]+factor]])
    data['second'] = range(len(data))
    data['start_idx'] = ind0
    data['end_idx'] = ind1
    data['SS'] = data['T_sim'].values
    data['tagged'] = data['TAGGED'].values
    data['score'] = data['ss_conv_score'].values
    
    # create summary report
    summary_report = pd.DataFrame([])
    duration = len(data)/finalFs/3600
    summary_report['signal duration (h)'] = [np.round(duration, 2)]
    summary_report['detected central apneas'] = [hdr[f'central apneas']]
    summary_report['detected central hypopneas'] = [hdr[f'central hypopneas']]
    summary_report['cai'] = [np.round(hdr[f'central apneas'] / duration, 1)]
    summary_report['cahi'] = [np.round((hdr[f'central apneas'] + hdr[f'central hypopneas']) / duration, 1)]
    summary_report['SS%'] = [np.round((np.sum(data['T_sim']==1) / (len(data))) * 100, 1)]

    # remove original columns
    for col in original_cols: 
        if col in data.columns: 
            data = data.drop(columns=col)   
    
    # save data into .csv files
    full_report = pd.concat([data, summary_report], axis=1)
    
    return full_report, summary_report

def dict_SS_per_channel(reports):
    SS_dict = {}

    # run over all summary reports
    keys = [k for k in reports.keys() if 'summary' in k]
    for key in keys:
        col = key.split('_')[1]
        SS_dict[col] = reports[key]['SS%'].values[0]
    
    return SS_dict

def update_any_SS_summary_report(reports, flow_reductions, summary_report):
    # run over all summary reports
    keys = [k for k in reports.keys() if 'full' in k]
    for i, key in enumerate(keys):
        col = key.split('_')[1]
        vals = reports[key]['SS'].values
        total = vals if i==0 else np.vstack([total, vals])

    # compute any over SS among channels
    any_SS = np.any(total, 0).astype(int)

    # recompute central events
    central_dic = compute_central_events(flow_reductions, np.repeat(any_SS, 10))

    # update summary report
    duration = summary_report['signal duration (h)']
    summary_report['detected central apneas'] = central_dic['central apneas']
    summary_report['detected central hypopneas'] = central_dic['central hypopneas']
    central_a = summary_report['detected central apneas']
    central_a_h = (summary_report['detected central apneas'] + summary_report['detected central hypopneas'])
    summary_report['cai'] = np.round(central_a / duration, 1)
    summary_report['cahi'] = np.round(central_a_h / duration, 1)
    summary_report['SS%'] = np.round((np.sum(any_SS==1) / (len(any_SS))) * 100, 1)  
    
    return summary_report, any_SS

# Plotting
def self_sim_plot(data, hdr, summary_report, main_ch, combined_flow_reductions=[], plot_all_tagged=False):
    # take middle 5hr segment --> // 10 rows == 30 min per row
    fs = hdr['newFs']
    final_plot = False
    if len(combined_flow_reductions)>0:
        final_plot = True
        main_col, main_ch = main_ch
        reports = summary_report
        summary_report = reports[f'summary_{main_col}']
        # create SS per channel list
        SS_per_channel = dict_SS_per_channel(reports)
        # compute any SS and update summary report
        summary_report, any_SS = update_any_SS_summary_report(reports, combined_flow_reductions, summary_report)
    
    # set signal variables
    signal = data.Ventilation_combined.values.astype(float)
    sleep_stages = data.Stage.values.astype(float)
    y_algo = data.flow_reductions.values.astype(int)
    tagged_breaths = data.tresh_TAGS.values.astype(int)
    ss_conv_score = data.ss_conv_score.values.astype(float)
    selfsim = data.T_sim.values.astype(int)
    if final_plot:
        any_SS = np.repeat(any_SS, fs)
        any_SS[selfsim==1] = 0

    # define the ids each row
    block = 60*60*fs
    row_ids = [np.arange(i*block, (i+1)*block) for i in range(len(signal)//block + 1)]
    row_ids.reverse()
    row_ids[0] = np.arange(row_ids[0][0], len(data))
    nrow = len(row_ids)
    
    # setup figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 15

    # set sleep array
    sleep = np.array(signal)
    sleep[np.isnan(sleep_stages)] = np.nan
    sleep[sleep_stages==5] = np.nan
    # set wake array
    wake = np.zeros(signal.shape)
    wake[np.isnan(sleep_stages)] += signal[np.isnan(sleep_stages)]
    wake[sleep_stages==5] += signal[sleep_stages==5]
    wake[wake==0] = np.nan
    # set rem array
    rem = np.array(signal)
    rem[sleep_stages!=4] = np.nan
    # set envelope + baseline
    env_pos = data.Smooth_pos_envelope.values
    env_neg = data.Smooth_neg_envelope.values
    baseline = data.Ventilation_baseline.values

    # PLOT SIGNALS
    for ri in range(nrow):
        # set clip and autoscale factor
        pos_clip = (ri+1)*row_height - row_height/1.8
        neg_clip = (ri-1)*row_height + row_height/1.8
        factor = max(1.5 / np.quantile(signal[row_ids[ri]], 0.90), 1)        

        # plot signal
        for array, c in zip([sleep, wake, rem], ['k', 'r', 'b']):
            array[:block//100] = np.nan
            array[-block//20:] = np.nan
            if not any(np.isfinite(array)): continue
            # fade large peaks
            arr = array[row_ids[ri]] * factor + ri*row_height
            fade = array[row_ids[ri]] * factor + ri*row_height
            arr[np.logical_or(arr>pos_clip, arr<neg_clip)] = np.nan
            # lower bound cut
            arr[arr < -row_height] = np.nan
            fade[fade < -row_height] = np.nan
            # upper bound cut
            arr[arr > nrow*row_height] = np.nan
            fade[fade > nrow*row_height] = np.nan
            # plot
            ax.plot(fade, c=c, lw=.3, alpha=0.0)
            ax.plot(arr, c=c, lw=.3)
            
        # set max_y
        if ri == nrow-1:
            max_y = np.nanmax([sleep[row_ids[ri]], wake[row_ids[ri]], rem[row_ids[ri]]]) + ri*row_height

    # PLOT LABELS
    ran = 4 if final_plot else 3
    for yi in range(ran):
        if yi==0:
            labels = y_algo                 # plot detected events
            label_color = [None, 'b', 'k']
        if yi == 1:
            labels = tagged_breaths         # '*' for HLG breathing oscillations
            label_color = [None, 'k', 'r']
        if yi == 2:
            labels = selfsim         # '*' for HLG breathing oscillations
            label_color = [None, 'b']
        if yi == 3:
            labels = any_SS         # '*' for HLG breathing oscillations
            label_color = [None, 'b']

        # run over each plot row
        for ri in range(nrow):            
            # group all labels and plot them
            loc = 0
            height = ri*row_height
            for i, j in groupby(labels[row_ids[ri]]):
                len_j = len(list(j))
                if not np.isnan(i) and label_color[int(i)] is not None:
                    # if yi == 0:
                    #     # add detected respiratory events
                    #     ax.plot([loc, loc+len_j], [height-2.5]*2, c=label_color[int(i)], lw=1)
                    if yi == 1:
                        # add tags
                        tag = '*' if i == 1 else '\''
                        c_score = np.round(ss_conv_score[row_ids[ri]][loc], 2)
                        c, sz = ('b', 12) if c_score >= hdr['SS_threshold'] else ('k', 8)
                        if c_score >= hdr['SS_threshold'] or plot_all_tagged:
                            offset = row_height/2
                            ax.text(loc, height + offset, tag, c=c, ha='center', va='top', fontsize=sz)
                            # ax.text(loc, height + offset+0.5, str(c_score), c='k', ha='center', va='bottom', fontsize=3)
                    if yi == 2:
                        # add SS bar
                        ymin = height - (row_height/2*0.9)
                        ymax = height + row_height/2
                        ax.fill_between([loc, loc+len_j], ymin, ymax, color='b', alpha=0.15, ec=None)
                    if yi == 3:
                        # add SS_any bar
                        ymin = height - (row_height/2*0.81)
                        ymax = height + (row_height/2*0.9)
                        ax.fill_between([loc, loc+len_j], ymin, ymax, color='k', alpha=0.15, ec=None)

                loc += len_j
                
    # plot layout setup
    ax.set_xlim([-0.01*block, 1.01*block])
    ax.axis('off')

    ### construct legend box ###
    len_x = len(row_ids[-1])

    # add <duration> min marking
    duration = 5
    offset = row_height*(nrow-1) + 17
    ax.plot([len_x-60*fs*duration, len_x], [offset]*2, color='k', lw=1)           # <duration>
    ax.plot([len_x-60*fs*duration]*2, [offset-0.5, offset+0.5], color='k', lw=1)  # left va
    ax.plot([len_x]*2, [offset-0.5, offset+0.5], color='k', lw=1)                 # right va
    ax.text(len_x-60*fs*(duration/2), offset+1, f'{duration} min', color='k', fontsize=8, ha='center', va='bottom')
    if len(main_ch) > 0:
        ax.text(len_x-60*fs*(duration/2), offset-1, f'({main_ch})', color='k', fontsize=8, ha='center', va='top')

    # add start/end marking
    arrow_dic = {'length_includes_head': True, 'width': block/1000, 'head_width': block/150, 'head_length': row_height*0.1, \
        'color': 'k', }
    ax.text(block/100, row_height*(nrow-0.5), 'Start rec.', color='k', fontsize=7, ha='left', va='top')
    ax.arrow(block/250, row_height*(nrow-0.5), 0, -row_height*0.25, **arrow_dic)
    array = np.array(signal)
    array[-block//20:] = np.nan
    loc = len(signal)%block - np.where(np.isfinite(np.flip(array)))[0][0]
    if loc<=0 or loc>block*0.95:
        ax.text(block-block/100, row_height*0.25, 'End rec.', color='k', fontsize=7, ha='right', va='bottom')
        ax.arrow(block-block/250, row_height*0.25, 0, row_height*0.25, **arrow_dic)
    else:
        ax.text(loc+block/100, -0.5*row_height, 'End rec.', color='k', fontsize=7, ha='left', va='bottom')
        ax.arrow(loc+block/250, -0.5*row_height, 0, row_height*0.25, **arrow_dic)

    # add summary report
    dx = len_x//12
    for i, key in enumerate(summary_report.keys()):
        tag = key.replace('detected ', '').replace('central', 'c.').replace('signal ', '')
        if final_plot and key=='SS%':
            tag = '$SS_{{any}}$%'
        tag += '\n' + str(summary_report[key].values[0])
        ax.text((i)*dx, offset, tag, fontsize=7, ha='left', va='bottom')

    # add metrics per channel to final plot
    if final_plot:
        # add border
        ax.plot([(i+0.9)*dx]*2, [offset-0.5, offset+4.5], color='k', lw=1)

        shift = len(summary_report.keys())*dx
        dxx = len_x//16
        for i, key in enumerate(SS_per_channel.keys()):
            tag = f'$SS_{{{key}}}$'
            tag += f'%\n{SS_per_channel[key]}'
            ax.text(shift + i*dxx, offset, tag, fontsize=7, ha='left', va='bottom')
        
    plt.tight_layout()
    return summary_report

def SS_range_plot(data, hdr, DFs=[]):
    # compute conv score data
    events = find_events(data['flow_reductions']>0)
    conv_scores = find_events(data['ss_conv_score']>0)
    total_events, total_convs = len(events), len(conv_scores)
    dic = {}
    step = 0.05
    for x in np.arange(0, 1+step, step):
        x = np.round(x, 3)
        # dic[x] = round(sum(data['ss_conv_score']>x) / total_convs * 100, 1)
        total = np.zeros(len(data))
        if DFs:
            for key in DFs.keys():
                df = post_process_self_sim(DFs[key], hdr['newFs'], x)
                total += df['self similarity'].values==2
                dic[f'{x}_{key}'] = np.round((np.sum(df['self similarity'].values==2) / (len(data))) * 100, 1)
            dic[x] = np.round((np.sum(total>0) / (len(data))) * 100, 1)
        else:
            df = post_process_self_sim(data.copy(), hdr['newFs'], x)
            dic[x] = np.round((np.sum(df['self similarity'].values==2) / (len(data))) * 100, 1)
    
    # create figure
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    marge = 0.015
    xs = [x for x in dic.keys() if type(x)!=str]
    ys = [dic[x] for x in xs]
    ax.plot(xs, ys, 'k', lw=2)
    if DFs:
        for i, col in enumerate(hdr['columns']):
            xs = [x for x in dic.keys() if col in str(x)]
            ys = [dic[x] for x in xs]
            xs = [float(x.split('_')[0]) for x in xs]
            ax.plot(xs, ys, 'k', lw=1, alpha=0.5)
            # add column name
            xloc = np.round(step*i, 3)
            ax.text(xloc+marge, ys[xs.index(xloc)]+(marge*50), col, fontsize=8, ha='left', va='bottom', alpha=0.5)

    # layout
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 101)
    ax.set_xlabel('1 - threshold', fontsize=12)
    ax.set_ylabel(f'SS%', fontsize=12)
    xticks = [item.get_text() for item in ax.get_xticklabels()]
    yticks = [item.get_text()+'%' for item in ax.get_yticklabels()]
    ax.set_xticklabels(xticks, fontsize=11)
    ax.set_yticklabels(yticks, fontsize=11)

    # plot layout setup
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # threshold
    thresh = hdr['SS_threshold']
    ax.plot([thresh]*2, [0, 100], 'k--', lw=0.75)
    ax.plot([0, 1], [dic[thresh]]*2, 'k--', lw=0.75)
    ax.plot(thresh, dic[thresh], 'k.', ms=16, alpha=0.5)
    if dic[thresh]>10:
        ax.text(1-marge, dic[thresh]-(marge*50), dic[thresh], fontsize=8, ha='right', va='top')
    else:
        ax.text(1-marge, dic[thresh]+(marge*50), dic[thresh], fontsize=8, ha='right', va='bottom')

# Apply algorithm
def run_SS_algorithm(data, hdr, col):
    # set hyperparamters
    hdr['SS_threshold'] = 0.80

    # compute envelope and baseline on ABD trace
    data = compute_envelopes(data, hdr['newFs'], channels=col)

    # set ventilation drop hyperparameters
    if col == 'ptaf':
        drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth = 0.41, 0.80, 8, 7.5, 0.70, False
    # set ABD+CHEST
    elif col in ['abd', 'chest', 'effort']:
        drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth = 0.48, 0.85, 7, 7, 0.65, True
    # set airflow, if possible
    elif col == 'airflow':
        drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth = 0.40, 0.85, 8, 7.5, 0.65, False

    # compute ventilation drops
    data = assess_ventilation(data, hdr, drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth=extra_smooth)

    # remove found flow reductions, with no clear forward drop
    data = remove_non_forward_drops(data, hdr, drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth=extra_smooth)

    # create flow reduction array
    data = combine_flow_reductions(data, hdr)

    # merge small separate event into long events
    data = merge_small_events(data, hdr['newFs'])

    # remove wake events
    data = remove_wake_events(data)

    # find potential self-similarity regions
    data = tag_potential_self_sim_spots(data, hdr['newFs'])

    # assess potential self-similarity regions
    data = assess_potential_self_sim_spots(data, hdr['newFs'])

    # apply AASM-rule post-processing
    data = post_process_self_sim(data, hdr['newFs'], hdr['SS_threshold'])

    # save SS score
    self_sim = data['self similarity'].values
    self_sim[self_sim!=2] = 0
    self_sim[self_sim==2] = 1
    data['T_sim'] = self_sim
    data['tresh_TAGS'] = np.array(data.ss_conv_score>hdr['SS_threshold']).astype(int)

    # compute number of central apneas hypopneas
    central_dic = compute_central_events(data.flow_reductions.values, data.T_sim.values)
    data['central apneas'], data['central hypopneas'] = central_dic['central a'], central_dic['central h']
    hdr['central apneas'], hdr['central hypopneas'] = central_dic['central apneas'], central_dic['central hypopneas']

    return data, hdr


# Main
if __name__ == '__main__':
    # show Credits
    show_credits()

    # list all files
    edf_folder = '.'
    edf_paths, results_dir = list_files(edf_folder)    

    # run over all .edf files
    for edf_path in edf_paths:
        # try:
        # 0. set out path
        fname = edf_path.replace('/', '').replace('\\', '').replace('.', '').replace('edf', '')
        edf_folder = results_dir + f'/{fname}/' 
        report_folder = edf_folder + 'reports/'
        figure_folder = edf_folder + 'figures/'
        os.makedirs(edf_folder, exist_ok=True)
        os.makedirs(report_folder, exist_ok=True)
        os.makedirs(figure_folder, exist_ok=True)
        final_report_path = edf_folder + 'Summary Table.csv'
        final_figure_path = edf_folder + 'Full-night SS.pdf'
        final_SS_path = edf_folder + 'SS% thresholding.pdf'
        # if os.path.exists(final_report_path) and os.path.exists(final_figure_path):
        #     print(f' < "{fname}" already processed >'); continue

        ## 1. load edf
        print(f'\n loading "{fname}.edf"')
        edf = mne.io.read_raw_edf(edf_path, stim_channel=None, preload=False, verbose=False)
        edf_channels = edf.info['ch_names']
        hdr = {'Fs': int(edf.info['sfreq']), 'newFs': 10}

        try:
            # try loading flow/effort signals
            print(' searching for breathing signals in .edf')
            data, channels = multiple_channel_search(edf)
            print(f'   and found {channels}\n')
        except:
            # if it doesn't work, try original effort signal search
            print('\nunsuccessful --> do original effort search instead\n')
            data = original_effort_search(edf)
            channels = ['effort']

        ## 2. preprocess breathing signals
        data = cut_flat_signal(data, hdr['Fs'])
        data = do_initial_preprocessing(data, hdr['newFs'], hdr['Fs'])

        # combine abd+chest if available
        if 'abd' in data.columns and 'chest' in data.columns and not 'effort' in data.columns:
            data['effort'] = data['abd'] + data['chest']
            channels += ['Sum Effort']
            print(f"->   computing 'Sum Effort'")
        hdr['columns'] = data.columns

        # clip normalize signals
        data = clip_normalize_signals(data, hdr['newFs'])

        # set main columns/channel
        main_col_chan = compute_main_channel(hdr['columns'], channels)
        assert type(main_col_chan) is not None

        ## 3. set sleep stages
        data['Stage'] = 1
        data['patient_asleep'] = np.logical_and(data.Stage>0, data.Stage<5)
        original_df = data.copy().astype(float)
        reports, events_per_channel = {}, {}

        # 4. run over all found breathing channels
        DFs = {}
        for i, (col, channel) in enumerate(zip(hdr['columns'], channels)):
            print(f"-->  assessing SS in '{channel}' [{i+1}/{len(channels)}]    ", end='\r')

            # return to original DF
            if i > 0:
                data = original_df.copy()
            
            # apply SS-algorithm
            data, hdr = run_SS_algorithm(data, hdr, col)
            events_per_channel[col] = data['flow_reductions'].values
            DFs[col] = data.copy()

            # create report
            full_report, summary_report = create_report(data, hdr)
            report_path = report_folder + f'Summary Report ({channel}).csv'
            full_report.to_csv(report_path, header=full_report.columns, index=None, mode='w+')
            reports[f'full_{col}'] = full_report
            reports[f'summary_{col}'] = summary_report
            
            # create SS recording figure
            _ = self_sim_plot(data, hdr, summary_report, channel) 
            figure_path = figure_folder + f'full recording ({channel}).pdf'
            plt.savefig(fname=figure_path, format='pdf', dpi=1200)
            plt.close()          

            # create SS% range figure
            SS_range_plot(data, hdr)  
            figure_path = figure_folder + f'SS% range ({channel}).pdf'
            plt.savefig(fname=figure_path, format='pdf', dpi=1200)
            plt.close() 
        
        # 5. Group central events across all channels
        total_array = events_per_channel[main_col_chan[0]]
        for vals in events_per_channel.values():
            for st, end in find_events(vals>0):
                # skip if there's any overlap
                start, ending = max(0, st-5*hdr['newFs']), min(end+5*hdr['newFs'], len(data))
                if any(total_array[start:ending])>0: continue
                total_array[st:end] = vals[st]

        # create final figure
        summary_report = self_sim_plot(data, hdr, reports, main_col_chan, total_array) 
        plt.savefig(fname=final_figure_path, format='pdf', dpi=1200)
        plt.close()

        # create SS% range figure
        SS_range_plot(data, hdr, DFs=DFs)  
        plt.savefig(fname=final_SS_path, format='pdf', dpi=1200)
        plt.close() 

        # create reports report
        for key in summary_report.keys():
            full_report[key] = summary_report[key]
        full_report.to_csv(final_report_path, header=full_report.columns, index=None, mode='w+')
        





        # Finish up            
        print(f'\n---> finished with Figures & Reports                        \n')
   
        # except Exception as e:
        #     print(f'Error for {edf_path}: {e}')
        #     print('Continue with next file.')




