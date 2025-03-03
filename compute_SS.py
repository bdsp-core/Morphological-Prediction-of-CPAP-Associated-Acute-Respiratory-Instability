from scipy.io import loadmat
import numpy as np
import pandas as pd
import os, glob, ray, h5py
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.signal import find_peaks
from statsmodels.robust.scale import mad
from scipy import interpolate

from .preprocessing_functions.find_events import find_events
from .preprocessing_functions.data_loading_functions import load_prepared_data_old as load_data_from_prepared_dataset
from .SS_detector.run_SS_detector import run_SS_algorithm

# loading and saving functions
def select_split_recordings(all_paths, dataset):
    # read split recordings from .csv
    split_recs_csv = f'./files/{dataset.upper()}_split_recs.csv'
    if not os.path.exists(split_recs_csv):
        skip, skip_notSplit, skip_noHour = 0, 0, 0
        split_recs =  pd.DataFrame([], columns=['FileName'])
        for p, path in enumerate(all_paths):
            print(f'finding {dataset.upper()} split recordings: #{p}/{len(all_paths)}', end='\r')            
            # init DF
            cols = ['sleep_stages', 'test_type','cpap_start', 'newFs']
            header_fields = ['patient_tag', 'newFs', 'Fs', 'test_type', 'rec_type', 'cpap_start', 'E_SS', 'T_SS', 'T_osc']
            data, hdr = pd.DataFrame([]), {}
            
            # retrieve data
            f = h5py.File(path, 'r')
            for key in cols:
                vals = f[key][:]
                if key in header_fields: 
                    hdr[key] = vals[0]
                else:
                    data.loc[:, key] = vals
            f.close()
            
            # skip non-split nights
            if not 'split-night' in str(hdr['test_type']): 
                skip_notSplit += 1
                continue

            # add patient asleep
            data['patient_asleep'] = np.logical_and(data.sleep_stages < 5, data.sleep_stages > 0)

            # skip is asleep before or after CPAP < 2hr
            hr = hdr['newFs'] * 3600 * 1
            if sum(data.loc[:hdr['cpap_start'], 'patient_asleep']) < hr or sum(data.loc[hdr['cpap_start']:, 'patient_asleep']) < hr:
                skip_noHour += 1
                continue

            split_recs.loc[len(split_recs), 'FileName'] = path.split('/')[-1].split('.hf5')[0] 

        # save split_recs
        split_recs.to_csv(split_recs_csv, index=False)
        print(f'Total recordings: {len(all_paths)}')
        print(f'Skipped not SplitNight: {skip_notSplit}')
        print(f'{skip_noHour} files skipped due to <1hr sleep before CPAP')

    # load split recs
    split_recs = pd.read_csv(split_recs_csv)['FileName'].values
    print(f'*{dataset.upper()} database contains {len(split_recs)} valid split-night recordings*')

    return split_recs

def compute_AHI_CAI(resp, stage, exclude_wake=True):
    # compute sleep time
    stage[~np.isfinite(stage)] = 0
    patient_asleep = np.logical_and(stage<5, stage>0)
    sleep_time = np.sum(patient_asleep==1) / 36000

    # compute RDI
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    metric = len(find_events(vals>0)) if exclude_wake else np.sum(vals>0)
    RDI = round(metric / sleep_time, 2)

    # compute AHI
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    vals[vals==7] = 4
    vals[vals>4] = 0
    vals[vals<0] = 0
    metric = len(find_events(vals>0)) if exclude_wake else np.sum(vals>0)
    AHI = round(metric / sleep_time, 2)

    # compute CAI
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    vals[vals==2] = 8
    vals[vals<7] = 0
    metric = len(find_events(vals>0)) if exclude_wake else np.sum(vals>0)
    CAI = round(metric / sleep_time, 2)

    return RDI, AHI, CAI, round(sleep_time, 2)

def write_to_hdf5_file(data, output_h5_path, hdr=[], default_dtype='float32', overwrite=False):
	# input:  df w/ signals, path_to_save, header dict
	# output: *saved data in .hdf5 file

	chunk_size = 64

	if not '.hf5' == output_h5_path[-4:]:
		output_h5_path = output_h5_path + '.hf5'

	if overwrite:
		if os.path.exists(output_h5_path):
			os.remove(output_h5_path)

	with h5py.File(output_h5_path, 'a') as f:

		# save signals:
		for signal_tmp in data.columns:
			if not signal_tmp in f: # first write of this signal
				if signal_tmp.lower() in ['annotation', 'test_type', 'rec_type', 'patient_tag', 'dataset']: 
					dtype1 = h5py.string_dtype(encoding='utf-8') # h5py needs to be >= 2.10
					dset_signal = f.create_dataset(signal_tmp, shape=(data.shape[0],), maxshape=(None,),
											chunks=(chunk_size,), dtype=dtype1)
					dset_signal[:] = data[signal_tmp].astype('str')
					continue # with next signal

				elif signal_tmp.lower() in ['stage', 'apnea', 'Fs', 'newFs', 'cpap_start']: 
					dtype1 = 'int32'    # for sleep lab data
					data.loc[pd.isna(data[signal_tmp]), signal_tmp] = -1

				else: dtype1 = default_dtype

				dset_signal = f.create_dataset(signal_tmp, shape=(data.shape[0],), maxshape=(None,),
												chunks=(chunk_size,), dtype=dtype1)
												
				dset_signal[:] = data[signal_tmp].astype(dtype1)

			else:                
				raise ValueError('Signal already exists in file but currently not intended to overwrite')

		# save header:
		if hdr:
			for hdr_e in hdr.keys():
				if type(hdr[hdr_e]) == type(None):
					hdr[hdr_e] = str(hdr[hdr_e])

				if type(hdr[hdr_e]) in[datetime.datetime, pd._libs.tslibs.timestamps.Timestamp]:
					hdr[hdr_e] = np.array([hdr[hdr_e].year,hdr[hdr_e].month,hdr[hdr_e].day, hdr[hdr_e].hour, 
											hdr[hdr_e].minute, hdr[hdr_e].second, hdr[hdr_e].microsecond])

				if type(hdr[hdr_e]) in [int, np.int32]:
					dset_signal = f.create_dataset(hdr_e, shape=(1,), maxshape=(1,),
													  chunks=True, dtype=np.int32)
					dset_signal[:] = np.int32(hdr[hdr_e])

				elif type(hdr[hdr_e]) == np.ndarray:
					dset_signal = f.create_dataset(hdr_e, shape=(hdr[hdr_e].shape[0],), maxshape=(hdr[hdr_e].shape[0]+10,),
													chunks=True, dtype=np.int32)
					dset_signal[:] = hdr[hdr_e].astype(np.int32)            

				elif type(hdr[hdr_e]) == str:
					dtypes1 = np.array([hdr[hdr_e]+'                     ']).astype('<S44').dtype
					dset_signal = f.create_dataset(hdr_e, shape=(1,),  maxshape=(None,),
													  chunks=True, dtype=dtypes1)
					string = hdr[hdr_e].encode('utf8')
					dset_signal[:] = string

				else:
					raise ValueError('Unexpected datatype for header.')

### E METHOD ###
def do_E_method(data, fs):
    # set data and time array
    channels = [c for c in ['abd', 'chest'] if c in data.columns]
    trace = np.squeeze(data[channels].values).T if len(channels)==1 else data[channels].mean(axis=1).values.T
    t = np.arange(len(trace))/fs 
    t = t.flatten()
    sim_thres = 0.8

    # remove outliers from signal based on signal envelope
    idx, trace = remove_outliers(t,trace,fs, min_height=0.01)

    # compute the upper and lower envelope of the respiration signal
    [up,lo] = create_env(trace, fs)
        
    # based on envelopes, central respiration signals can be detected
    _, _ = detect_central_events(lo,up,fs)

    # compute SS array
    similarity_array = compute_similarity(up, lo, fs)

    data['E_sim'] = np.array(similarity_array>sim_thres).astype(int)

    return data 

def isNaN(num):
    return num != num

def envelope(trace, fs=200, r=1000):
    '''
    fs: sampling rate
    r: min distance of peaks, in samples
    '''
    peaks = find_peaks(trace, distance=r)[0]
    troughs = find_peaks(-trace, distance=r)[0]
    
    mvg_avg_dur = 10 # in seconds
    trace_mvg_avg = pd.Series(trace)
    trace_mvg_avg = trace_mvg_avg.rolling(mvg_avg_dur*fs, center=True).mean().values
    trace_mvg_avg[np.isnan(trace_mvg_avg)]=0
    
    peaks = peaks[np.where(trace[peaks] > trace_mvg_avg[peaks]*1.1)[0]]
    troughs =  troughs[np.where(trace[troughs] < trace_mvg_avg[troughs]*0.9)[0]]

    if peaks.shape[0] == 0:
        return None, None

    try:
        f_interp = interpolate.interp1d(peaks, trace[peaks], kind='cubic', fill_value="extrapolate")
        envelope_up = f_interp(range(len(trace)))
    except:
        envelope_up = None
    try:
        f_interp = interpolate.interp1d(troughs, trace[troughs], kind='cubic', fill_value="extrapolate")
        envelope_lo = f_interp(range(len(trace)))
    except:
        envelope_lo = None

    return envelope_up, envelope_lo

def create_env(trace, fs, r = 5):
    'r: in seconds'
    
    r = r*fs; # 5 seconds distance for peaks.
    up,lo =  envelope(trace, fs, r=r)

    if np.any([up is None, lo is None]):
        return up, lo

    ind = np.where(up<lo)[0]
    up[ind] = 0
    lo[ind] = 0
    ind = np.where(up<(0.8*trace))[0]
    for i in range(4*fs,len(ind)):
        up[ind[i]] = np.mean(up[(ind[i]-3*fs):ind[i]])
    ind = np.where(lo>(0.9*trace))[0]
    for i in range(4*fs,len(ind)):
        lo[ind[i]] = np.mean(lo[(ind[i]-3*fs):ind[i]])
    return up,lo

def connect_apneas(th,d,fs):
    Nt = len(d)
    n = round(fs*5000/200); # max time before giving up search for successor

    # forward
    ct = float("inf")
    inOne = 0
    z = np.full([ len(d)], np.nan)
    for i in range(Nt):
        if d[i]<th : 
            ct=0
        if d[i]>=th:
            ct=ct+1
        if ct<n:
            z[i] = -1000

    #backward
    dd = np.fliplr([d])[0]
    ct = float("inf")
    inOne = 0
    zz=np.full([ len(dd)], np.nan)
    for i in range(Nt):
        if dd[i]<th:
            ct=0;
        if dd[i]>=th:
            ct=ct+1; 
        if ct<n:
            zz[i] = -1000  
    zz = np.fliplr([zz])[0]
    ind = np.where(~isNaN(zz))
    z[ind] = -1000;
    return z

def remove_outliers(t,trace,fs, min_height=0.01, do_plots=False):
    
    tPeaks,peaks,tTroughs,troughs = get_peaks_troughs_2(t,trace,min_height,fs)
    t1 = tPeaks.copy()
    t2 = tTroughs.copy()
    y1 = peaks.copy()
    y2 = troughs.copy()
    ThresholdFactor = 5
    tf1 = is_outlier(y1,ThresholdFactor=ThresholdFactor)
    ind1 = list(np.nonzero(tf1)[0]); 
    tf2 = is_outlier(y2,ThresholdFactor=ThresholdFactor)
    ind2 = list(np.nonzero(tf2)[0]); 

    yy1 = y1.copy()
    yy2 = y2.copy()
    yy1 = fill_outliers(yy1,ThresholdFactor=ThresholdFactor)
    yy2 = fill_outliers(yy2,ThresholdFactor=ThresholdFactor)

    #get envelope by interpolation
    f1 = interpolate.interp1d(t1,yy1,bounds_error=False)
    f2 = interpolate.interp1d(t2,yy2,bounds_error=False)
    up = f1(t)
    lo = f2(t)
    lo = [ lo[i] if (lo[i] <= up[i]) else up[i] for i in range(len(lo))]
    up = [ lo[i] if (lo[i] >= up[i]) else up[i] for i in range(len(up))]
    m = np.add(up,lo)/2

    idx = [] # keep track of points that have been "corrected"
    ind = [ i for i in range(len(trace)) if (trace[i] > up[i]) ]
    idx = idx + ind
    trace = [ up[i] if (trace[i] > up[i]) else trace[i] for i in range(len(trace))]
    ind = [ i for i in range(len(trace)) if (trace[i] < lo[i]) ]
    idx = idx + ind
    trace = [ lo[i] if (trace[i] < lo[i]) else trace[i] for i in range(len(trace))]
    idx = np.sort(idx)

    traceN = trace-m; 
    # d = np.array(up) - np.array(lo)
    # print(np.nanmedian(d))
    # traceN = 600*traceN/np.nanmedian(d)

    traceN[np.isnan(traceN)] = 0 # 500
    

    if do_plots:
        fig, ax = plt.subplots(5,1, sharex=True, sharey=True, figsize=(16,20), dpi=80)
        ax[0].plot(t, trace,linewidth=1)
        ax[0].scatter(tPeaks, peaks, c='r',s=8)
        ax[0].scatter(tTroughs, troughs, c='r',s=8)
        ax[0].set_xlim([0,1000])
        ax[0].set_ylim([-11,11])
        ax[1].scatter(t1,y1,c='r',s=8)
        ax[1].scatter(t2, y2, c='r',s=8)
        ax[1].scatter(t1[ind1],y1[ind1],c='blue',s=8)
        ax[1].scatter(t2[ind2],y2[ind2],c='blue',s=8)    

        # ax[2].scatter(t1,yy1,c='r',s=8)
        # ax[2].scatter(t2,yy2,c='r',s=8)
        # ax[2].scatter(t1[ind1],yy1[ind1],c='blue',s=8)
        # ax[2].scatter(t2[ind2],yy2[ind2],c='blue',s=8)
        ax[2].plot(t,trace,linewidth=1)
        ax[2].plot(t,up,c='peru',linewidth=1)
        ax[2].plot(t,lo,c='gold',linewidth=1)

        ax[3].plot(t,trace,linewidth=1)
        ax[3].plot(t,m,c='r',linewidth=1)

        ax[4].plot(t, traceN, linewidth=1)
    
    return idx, traceN

# replaces outliers with the previous non-outlier element
def fill_outliers(y,ThresholdFactor=5):
    outliers = is_outlier(y,ThresholdFactor=ThresholdFactor)
    outliers_indices = np.nonzero(outliers);
    for outlier_index in list(outliers_indices[0]):
        i=1
        while (outliers[outlier_index-i] == 1):
            i += 1
        y[outlier_index] = y[outlier_index-i] 
    return y

def is_outlier(x_array,ThresholdFactor=5):
    MAD = mad(x_array)
    median = np.median(x_array)
    return [1 if ((x > (median+MAD*ThresholdFactor))|(x < (median-MAD*ThresholdFactor))) else 0 for x in x_array] 

def get_peaks_troughs_4(t,trace,mph,mpd):
    i0,properties = find_peaks(trace,height=mph,distance=mpd)
    i1 = [];
    for i in range(0,len(i0)-1):
        ind = list(np.arange(i0[i],i0[i+1]))
        ii = min(trace[ind])
        jj = list(trace[ind]).index(ii) 
        i1.append(ind[jj])

    if len(i0) == 0:
        return None, None, None, None, None, None

    ind = list(np.arange(i0[-1],len(trace)))
    ii = min(trace[ind])
    jj = list(trace[ind]).index(ii) 
    i1.append(ind[jj])
    i1=np.asarray(i1)

    tPeaks =  t[i0] 
    peaks = trace[i0]; 
    tTroughs = t[i1]; 
    troughs = trace[i1]; 

    killIt = np.zeros(len(tPeaks))
    for i in range(0,len(tPeaks)):
        if (peaks[i]/(troughs[i]+.1)<3.5): 
            killIt[i]=1
        if troughs[i]>0.9: 
            killIt[i]=1
        if (abs(tPeaks[i]-tTroughs[i])>100):
            killIt[i]=1

    ind = list(np.where(killIt == 0)[0])
    peaks = peaks[ind] 
    troughs= troughs[ind]
    tPeaks=tPeaks[ind]
    tTroughs=tTroughs[ind]
    i0 = i0[ind]
    i1 = i1[ind]
    return tPeaks,peaks,tTroughs,troughs,i0,i1

def get_peaks_troughs_2(t,trace,mph,mpd):
    i0,properties = find_peaks(trace,height=mph,distance=mpd)
    i2 = [];
    for i in range(0,len(i0)-1):
        ind = list(np.arange(i0[i],i0[i+1]))
        ii = min(trace[ind]); 
        jj = list(trace[ind]).index(ii) 
        i2.append(ind[jj]); 

    ind = list(np.arange(i0[-1],len(trace)))
    ii = min(trace[ind]); 
    jj = list(trace[ind]).index(ii) 
    i2.append(ind[jj]);

    tPeaks =  t[i0].copy()
    peaks = trace[i0].copy(); 
    tTroughs = t[i2].copy(); 
    troughs = trace[i2].copy(); 
    return tPeaks,peaks,tTroughs,troughs

def load_mgh_matlab_data(data_path, fs):
    
    data = loadmat(data_path)
    channels = [data['hdr'][0][x][0][0] for x in range(len(data['hdr'][0]))]
    abd_channel_no = channels.index('ABD')
    chest_channel_no = channels.index('CHEST')
    trace = data['s'][abd_channel_no,:] + data['s'][chest_channel_no,:]
    t = np.arange(len(trace))/fs
    
    return t, trace

def clip_z_normalize(signal):
    print(np.percentile(signal,1))
    print(np.percentile(signal,4))
    print(np.percentile(signal,96))
    print(np.percentile(signal,99))
    signal_clipped = np.clip(signal, np.percentile(signal,5), np.percentile(signal,95))
    signal = (signal - np.mean(signal_clipped))/np.std(signal_clipped)

    print(np.percentile(signal,1))
    print(np.percentile(signal,4))
    print(np.percentile(signal,96))
    print(np.percentile(signal,99))

    return signal

def detect_central_events(lo, up, fs):
    d = up - lo 
    centr_detected = np.zeros(d.shape)
    peakenv = np.zeros(d.shape)
    wdw = 3*60*fs #window of 3 minutes to find value for upper 90% of the difference between upper and lower envelope 
    for i in range(wdw, len(d)-wdw, wdw):
        peakenv[i:i+wdw]=np.percentile(d[i-wdw+1:i],90)

    # central apnea when difference between upper and lower envelope is smaller than 0.2 times the 90% percentile peakenv (difference of upper and lower envelope) of the 3 minutes before
    centr_detected[np.where(d<0.2*peakenv)[0]] = 1

    # HYPOPNEAS
    centr_hypo = np.zeros(d.shape)
    centr_hypo[np.where(d<0.7*peakenv)[0]] = 1
    
    # %it is only a central apnea when it lasts 10 sec
    # %9 seconds is used now, because the envelope causes a more smooth decrease
    # %of the trace so have to give it a bit more margin
    centr_detected_final = np.zeros(centr_detected.shape)
    for i in np.arange(4.5*fs, len(centr_detected)-4.5*fs, fs):
        if sum(centr_detected[int(i-4.5*fs):int(i+4.5*fs)])>8.5*fs:
            centr_detected_final[int(i-4.5*fs):int(i+4.5*fs)] = 1


    centr_hypo_final = np.zeros(centr_hypo.shape)
    for i in np.arange(4.5*fs, len(centr_hypo)-4.5*fs, fs):
        if sum(centr_hypo[int(i-4.5*fs):int(i+4.5*fs)])>8.5*fs:
            centr_hypo_final[int(i-4.5*fs):int(i+4.5*fs)] = 1
          
    return centr_detected_final, centr_hypo_final

def generate_report(trace, centr_detected_final, centr_hypo_final, fs, save=False):
    
    state_switches = centr_detected_final[1:] - centr_detected_final[:-1]
    apnea_start = np.where(state_switches==1)[0]+1
    apnea_end = np.where(state_switches==-1)[0]+1
    state_switches = centr_hypo_final[1:] - centr_hypo_final[:-1]
    hypopnea_start = np.where(state_switches==1)[0]+1
    hypopnea_end = np.where(state_switches==-1)[0]+1
    
    hypopnea_report = pd.DataFrame(columns=['start_sample','end_sample','start_second','end_second','event'])
    hypopnea_report.start_sample = hypopnea_start
    hypopnea_report.end_sample = hypopnea_end
    hypopnea_report.start_second = np.round(hypopnea_start/fs,1)
    hypopnea_report.end_second = np.round(hypopnea_end/fs,1)
    hypopnea_report.event  = ['central hypopnea']*hypopnea_report.shape[0]
    
    apnea_report = pd.DataFrame(columns=['start_sample','end_sample','start_second','end_second','event'])
    apnea_report.start_sample = apnea_start
    apnea_report.end_sample = apnea_end
    apnea_report.start_second = np.round(apnea_start/fs,1)
    apnea_report.end_second = np.round(apnea_end/fs,1)
    apnea_report.event  = ['central apnea']*apnea_report.shape[0]
    
    report_events = pd.concat([hypopnea_report,apnea_report],ignore_index=True).sort_values(by='start_sample').reset_index()
    report_events.drop('index', axis=1,inplace=True)
    
    summary_report = pd.DataFrame([],columns=['signal duration (h)', 'detected central apnea events', 'detected central hypopnea events'])
    summary_report['signal duration (h)'] = [np.round(len(trace)/fs/3600,2)]
    summary_report['detected central apnea events'] = [len(apnea_start)]
    summary_report['detected central hypopnea events'] = [len(hypopnea_start)]
    
    full_report = pd.concat([report_events,summary_report],axis=1)
    if save:
        full_report.to_csv('report_central_resp_events.csv', index=False)

    return full_report
    
# replace all zeros by nan's
def plot_central_events(trace, centr_detected_final, centr_hypo_final, savepath = 'figure_central_resp_events'):

    central_events = np.zeros(centr_detected_final.shape)
    central_events[centr_hypo_final.astype('bool')] = 2
    central_events[centr_detected_final.astype('bool')] = 1
    central_events.astype('float')
    central_events[central_events==0] = float('nan')

    assert(central_events.shape[0] == trace.shape[0])


    # use seg_start_pos to convert to the nonoverlapping signal
    # y = ytrue                               # shape = (N, 4100)
    # yp = apnea_prediction                   # shape = (N, 4100)
    # yp_smooth = apnea_prediction_smooth     # shape = (N, 4100)

    # define the ids each row
    nrow = 10
    row_ids = np.array_split(np.arange(len(trace)), nrow)
    row_ids.reverse()

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 7
    label_color = [None, 'g', 'c']

    # here, we get clip-normalized signal. we should not need to plot >3 STD values:
    trace[np.abs(trace > 3)] = np.nan

    for ri in range(nrow):
        # plot signal
        ax.plot(trace[row_ids[ri]]+ri*row_height, c='k', lw=0.2)


        y2 = central_events         

        yi=0

        # run over each plot row
        for ri in range(nrow):
            # plot tech annonation
    #         ax.axhline(ri*row_height-3*(2**yi), c=[0.5,0.5,0.5], ls='--', lw=0.2)  # gridline
            loc = 0

            # group all labels and plot them
            for i, j in groupby(y2[row_ids[ri]]):
                len_j = len(list(j))
                if not np.isnan(i) and label_color[int(i)] is not None:
                    # i is the value of the label
                    # list(j) is the list of labels with same value
                    ax.plot([loc, loc+len_j], [ri*row_height-3*(2**yi)]*2, c=label_color[int(i)], lw=1)
                loc += len_j

    # plot layout setup
    ax.set_xlim([0, max([len(x) for x in row_ids])])
    ax.axis('off')
    plt.tight_layout()
    # plt.title(test_info[si])
    # save the figure
    plt.savefig(savepath + '.pdf')
    plt.savefig(savepath + '.png')

def compute_similarity(up, lo, fs):
    env_diff = up-lo
    similarity_array = np.zeros(env_diff.shape)

    env_diff[env_diff<0] = 0
    th = np.percentile(env_diff, 95)
    
    apneas2 = connect_apneas(th, env_diff, fs)
    apneas2[np.where(np.isnan(apneas2))] = 0
    apneas2[np.where(apneas2==-1000)] = 2

    env_diff = env_diff-np.percentile(env_diff,5)
    # upper and lower envelopes of envelope differences
    [tmp,lo2] = create_env(env_diff, fs=fs, r=25) # 25 8000/fs
    if lo2 is None:
        return similarity_array

    # subtract baseline
    env_diff = env_diff - lo2 
    [up2,tmp] = create_env(env_diff, fs=fs, r=25) # 25 8000/fs
    if up2 is None:
        return similarity_array

    env_diff = env_diff/(up2+0.00001)
    t = np.arange(len(env_diff))/fs
    
    [tePeaks,epeaks,teTroughs,etroughs,i0,i1] = get_peaks_troughs_4(t,abs(up-lo),2.51,10*fs) # get_peaks_troughs_4(t,abs(up-lo),3.51,5000)

    if tePeaks is None:
        return similarity_array

    # get similarities
    # get time of crescendo-diminuendo patterns

    # peakTimes are start/end points of waves/segments of signals of interest.
    # peakTimes are simply mean points between two peaks found in envelope differnece.
    # for first peak, it's -15/+15 seconds of 
    # new implemention, with samples:
    pattern_boundary = np.zeros((len(i0),2), dtype=np.int64) # this is called peakTimes.
    if pattern_boundary.shape[0] > 0:
        pattern_boundary[0,:] = [max(0,i0[0]-15*fs), i0[0]+15*fs]
        for i in range(len(tePeaks)-1):
            pattern_boundary[i+1,:] = [np.mean(i0[i:i+2]), np.mean(i0[i+1:i+3])]
        if len(tePeaks)-1 > 0:
            pattern_boundary[i+1,1] = min(pattern_boundary[i+1,1]+15*fs, len(env_diff))    

        # do convolution:
        for [pattern_start, pattern_end] in pattern_boundary:
            if pattern_start == 0: continue
            # get first pattern (wave) and normalize it:
            s1 = env_diff[pattern_start:pattern_end].copy()

            s1 = (s1-np.nanmean(s1)) / (np.nanstd(s1)+0.00001)
            len_pattern = pattern_end-pattern_start

            # get wave ahead (wave-1)|
            s0 = env_diff[max(1,pattern_start-len_pattern):pattern_start].copy()
            s0 = (s0-np.nanmean(s0)) / (np.nanstd(s0)+0.00001)

            # get wave behind (wave+1)
            s2 = env_diff[pattern_end: min(pattern_end+len_pattern,len(env_diff))]
            s2 = (s2-np.nanmean(s2)) / (np.nanstd(s2)+0.00001)
            # convolution
            if len(s2)==0:
                # s2 = np.zeros(s1.shape)
                average_conv = max(np.convolve(s1,s0,'same'))/len_pattern
            else:
                conv1 = np.convolve(s1,s0,'same')
                conv2 = np.convolve(s1,s2,'same')
                # average_conv = (conv1 + conv2)/2/len_pattern
                average_conv = (np.nanpercentile(conv1,99)/len_pattern +  np.nanpercentile(conv2,99)/len_pattern)/2

            similarity_array[pattern_start:pattern_end] = average_conv

    return similarity_array
    
def post_processing_detections(apneas, hypopneas, similarity_array, fs, sim_thres=0.5):
    # if apnea in hypopnea, keep only apnea.
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    for [hypo_on_tmp, hypo_off_tmp] in list(zip(hypo_on, hypo_off)):
        if any(apneas[hypo_on_tmp:hypo_off_tmp]==1):
            hypopneas[hypo_on_tmp:hypo_off_tmp+1] = 0

    # remove detections with similarity less than `sim_thres`
    apneas[similarity_array<sim_thres] = 0
    hypopneas[similarity_array<sim_thres] = 0
    
    # need to last more than 9 seconds:
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    diff=[]
    for [hypo_on_tmp, hypo_off_tmp] in list(zip(hypo_on, hypo_off)):
        if hypo_off_tmp - hypo_on_tmp < 9*fs:
            hypopneas[hypo_on_tmp:hypo_off_tmp+1] = 0
    
    # if less than 5 seconds gap between two detections, connect them
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    apnea_on = np.where(np.diff(apneas)==1)[0]
    apnea_off = np.where(np.diff(apneas)==-1)[0]
        
    for [hypo_on_tmp, hypo_off_previous] in list(zip(hypo_on[1:], hypo_off[:-1])):
        if hypo_on_tmp - hypo_off_previous < 5*fs:
            hypopneas[hypo_off_previous:hypo_on_tmp+1] = 1

    for [apnea_on_tmp, apnea_off_previous] in list(zip(apnea_on[1:], apnea_off[:-1])):
        if apnea_on_tmp - apnea_off_previous < 5*fs:
            apneas[apnea_off_previous:apnea_on_tmp+1] = 1

    return apneas, hypopneas


### T METHOD ###
def do_T_method(data, hdr, channels):
    # convert central hypopneas to normal hyponeas
    # data.loc[data.Apnea==7, 'Apnea'] = 4

    # apply SS algorithm
    tag = 'abd_chest/' if all([True if c in data.columns else False for c in ['abd', 'chest']]) else 'abd/'
    data = run_SS_algorithm(data, hdr, tag, plot_version='None')


    self_sim = data['self similarity'].values
    self_sim[self_sim!=2] = 0
    self_sim[self_sim==2] = 1
    data['T_sim'] = self_sim

    # save tagged locations with positive E_sim
    data['E_tagged'] = np.logical_and(data.TAGGED, data.E_sim)
    
    return data


# saving functions 
def save_output(data, hdr, out_file, dataset):
    # put data in DataFrame
    df = pd.DataFrame([]) 
    df['breathing'] = data.Ventilation_combined.values
    df['abd'] = data.abd.values
    df['spo2'] = data.spo2.values
    df['arousal'] = data.arousal.values
    df['apnea'] = data.Apnea.values
    df['flow_reductions'] = data.flow_reductions.values
    df['sleep_stages'] = data.Stage.values
    df['E_sim'] = data.E_sim.values
    df['T_sim'] = data.T_sim.values
    df['T_tagged'] = data.TAGGED.values
    df['self similarity'] = data['self similarity'].values
    df['ss_conv_score'] = data.ss_conv_score.values
    # df['bad_signal'] = data.bad_signal

    # add individual scorer info
    if dataset == 'robert':
        for label in ['stage_0', 'resp_0', 'stage_1', 'resp_1', 'stage_2', 'resp_2']:
            df[label] = data[label].values

    # add header info
    for key in hdr.keys():
        df[key] = hdr[key]

    write_to_hdf5_file(df, out_file, overwrite=True)

def both_methods_self_sim_plot(data, hdr, out_path=''):
    # take middle 5hr segment --> // 10 rows == 30 min per row
    fs = hdr['newFs']
    block_size = fs * 3600 * 5
    
    # set signal variables
    signal = data.Ventilation_combined.values
    sleep_stages = data.Stage.values
    y_tech = data.Apnea.values
    y_algo = data.flow_reductions.values
    E_sim = data['E_sim'].values
    T_sim = data['T_sim'].values
    tagged_breaths = data.TAGGED.values
    Etagged_breaths = data.E_tagged.values
    ss_conv_score = data.ss_conv_score.values

    # define the ids each row
    nrow = 10
    row_ids = np.array_split(np.arange(len(signal)), nrow)
    row_ids.reverse()

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 30

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
        a = 1 #if ri == 7 else 0
        # plot signal
        ax.plot(sleep[row_ids[ri]]+ ri*row_height, c='k', lw=.3, alpha=a)
        ax.plot(wake[row_ids[ri]] + ri*row_height, c='r', lw=.3, alpha=a)
        ax.plot(rem[row_ids[ri]] + ri*row_height, c='b', lw=.3, alpha=a)
        # plot envelopes + baseline
        # ax.plot(env_pos[row_ids[ri]]+ri*row_height, c='b', lw=0.5)
        # ax.plot(env_neg[row_ids[ri]]+ri*row_height, c='b', lw=0.5)
        # ax.plot(baseline[row_ids[ri]]+ri*row_height, c='g', lw=0.8)

        # plot split line for PTAF <--> CPAP
        if hdr['cpap_start'] in row_ids[ri]:
            loc = np.where(row_ids[ri]==hdr['cpap_start'])[0]
            min_ = -15 + ri*row_height
            max_ =  15 + ri*row_height
            ax.plot([loc, loc], [min_, max_], c='r', linestyle='dashed', zorder=10, lw=4)
            ax.text(loc-20*fs, min_, 'Diagnostic', fontsize=8, ha='right', va='bottom')
            ax.text(loc+20*fs, max_, 'Titration', fontsize=8, ha='left', va='top')

        # set max_y
        if ri == nrow-1:
            max_y = np.nanmax([sleep[row_ids[ri]], wake[row_ids[ri]], rem[row_ids[ri]]]) + ri*row_height

    # PLOT LABELS
    for yi in range(6):
        if yi==0:
            labels = y_tech                 # plot tech label
            label_color = [None, 'k', 'b', 'k', 'm', None, None]
        elif yi==1:
            labels = y_algo                 # plot tech label
            label_color = [None, 'b', 'g', 'c', 'm', 'r', None, 'g']
        if yi == 2:
            labels = E_sim               # shaded red area for self-sim spots
            label_color = [None, 'b'] 
        if yi == 3:
            labels = T_sim               # shaded red area for self-sim spots
            label_color = [None, 'b']
        if yi == 4:
            labels = tagged_breaths         # '*' for HLG breathing oscillations
            label_color = [None, 'k', 'r']
        if yi == 5:
            labels = Etagged_breaths         # '*' for HLG breathing oscillations
            label_color = [None, 'k', 'r']

        # run over each plot row
        for ri in range(nrow):
            # if yi == 0:
            #     # plot tech annonation
            #     ax.axhline(ri*row_height-3*(2**yi), c=[0.5,0.5,0.5], ls='--', lw=0.2)  # gridline
            
            # group all labels and plot them
            loc = 0
            for i, j in groupby(labels[row_ids[ri]]):
                len_j = len(list(j))
               
                if not np.isnan(i) and label_color[int(i)] is not None:
                    # i is the value of the label
                    # list(j) is the list of labels with same value
                    if yi < 1:
                        # add scored events
                        sub = 0 if int(i) < 7 else 2
                        minus = 3 if not 'fc811d4b' in out_path else 4
                        ax.plot([loc, loc+len_j], [ri*row_height-minus*(2**yi)]*2, c=label_color[int(i)], lw=1)
                        if int(i) == 7:
                            ax.plot([loc, loc+len_j], [ri*row_height-minus*(2**1)]*2, c='m', lw=2)
                    # if yi == 2:
                    #     # create red area self_sim events
                    #     ax.plot([loc, loc+len_j], [ri*row_height+12]*2, c=label_color[int(i)], lw=3, alpha=1)
                    # if yi == 3:
                        # create red area self_sim events
                        # ax.plot([loc, loc+len_j], [ri*row_height+10]*2, c=label_color[int(i)], lw=3, alpha=1)
                    if yi == 4:
                        tag = '*' if i == 1 else '\''
                        c_score = np.round(ss_conv_score[row_ids[ri]][loc], 2)
                        color = 'b' if c_score > 0.8 else 'k'
                        # if not np.isnan(c_score):
                        if c_score>=0:
                            ax.text(loc, ri*row_height+14, str(c_score), ha='center', fontsize=3)
                            ax.text(loc, ri*row_height+8, tag, c=color, ha='center')

                loc += len_j
                
    # plot layout setup
    ax.set_xlim([0, max([len(x) for x in row_ids])])
    ax.axis('off')

    ### construct legend box ###
    # split line
    len_x = len(row_ids[0])
    fz = 11
    # ax.plot([0, len_x], [-20]*2, c='k', lw=1)

    # add 10 sec marking
    ax.plot([len_x-60*fs*5, len_x], [max_y+5]*2, color='k', lw=1) # 10sec
    ax.text(len_x-60*fs*2.5, max_y+7, '5 min', color='k', fontsize=fz-2, ha='center', va='bottom', weight='bold')

    # event types
    y = -20
    
    line_types = ['Wake', 'NREM', 'REM']
    line_colors = ['r', 'k', 'b']
    for i, (color, e_type) in enumerate(zip(line_colors, line_types)):
        x = 200*fs + 200*fs*i
        ax.plot([x, x+50*fs], [y]*2, c=color, lw=0.8)
        ax.text(x+25*fs, y-3, e_type, fontsize=fz, c=color, ha='center', va='top')

    # lines types
    event_types = ['Hypopnea', 'Central\napnea', 'Obstructive\napnea']
    label_colors = ['m', 'b', 'k']
    for i, (color, e_type) in enumerate(zip(label_colors, event_types)):
        x = len_x -200*fs -250*fs*i
        ax.plot([x, x-100*fs], [y]*2, c=color, lw=3)
        ax.text(x-50*fs, y-3, e_type, fontsize=fz, ha='center', va='top')

    ax.text(len_x//2, y, '*', c='b', fontsize=fz+3, ha='center')
    ax.text(len_x//2, y-3, 'Detected HLG\nbreathing oscillation', fontsize=fz, ha='center', va='top')

    plt.tight_layout()
    # save the figure
    if len(out_path) > 0:
        plt.savefig(out_path)
    plt.close()


##################################################################################

@ray.remote
def run_algo_and_plot_multiprocess(path, static_vars):
    from .preprocessing_functions.data_loading_functions import load_prepared_data_old as load_data_from_prepared_dataset
    dataset, out_file_folder, figure_folder = static_vars

    # show patientID
    patient_ID = path.split('/')[-1].split('.')[0]
    tag = '*loading %s recording: %s (%s)'%(dataset.upper(), patient_ID[:30], len(os.listdir(figure_folder))+1)
    print('\n\n' + '='*len(tag), '\n%s'%tag)

    # skip if output exists
    out_file = out_file_folder + patient_ID + '.hf5'

    # load data
    try:
        data, hdr = load_data_from_prepared_dataset(path, 'mgh_v3')
        data['patient_asleep'] = 1
        old_names = ['resp', 'stage']
        new_names = ['Apnea', 'Stage']
        for old_name, new_name in zip(old_names, new_names):
            try: 
                data = data.rename(columns={old_name: new_name})
            except:
                data[new_name] = data[old_name]
                data = data.drop(columns=old_name)
        hdr['patient_tag'] = patient_ID
        assert 'Apnea' in data.columns and 'Stage' in data.columns, 'no labels found'
    except Exception as error: print('Loading error: ', error); return  
    
    # compute sleep metrics
    RDI, AHI, CAI, sleep_time = compute_AHI_CAI(data.Apnea.values, data.Stage.values)
    hdr['RDI'], hdr['AHI'], hdr['CAI'], hdr['sleep_time'] = RDI, AHI, CAI, sleep_time

    #### APPLY E METHOD ####
    try:
        data = do_E_method(data, hdr['newFs'])
        hdr['E_SS'] = np.round((np.sum(data['E_sim']==1) / (np.sum(data.patient_asleep==1))) * 100, 1)
    except Exception as error: print('Error applying E method: ', error); return 

    #### APPLY T METHOD #######
    try:
        data = do_T_method(data, hdr, channels)
        hdr['T_SS'] = np.round((np.sum(data['T_sim']==1) / (np.sum(data.patient_asleep==1))) * 100, 1)    
        hdr['T_osc'] = round(len(find_events(data.TAGGED)) / (np.sum(data.patient_asleep==1) / hdr['newFs'] / 3600), 1)
    except Exception as error: 
        print('Error applying T method: ', error); return 

    # save output
    save_output(data, hdr, out_file, dataset)

    # plot signal + both outputs
    both_methods_self_sim_plot(data, hdr, out_path='')

def run_algo_and_plot_single_run(path, dataset, out_file_folder):
    # show patientID
    patient_ID = path.split('/')[-1].split('.')[0]

    tag = '*loading %s recording: %s'%(dataset.upper(), patient_ID[:30])
    print('\n\n' + '='*len(tag), '\n%s'%tag)

    # skip if output exists
    out_file = out_file_folder + patient_ID + '.hf5'
    
    # load data
    try:
        data, hdr = load_data_from_prepared_dataset(path, 'mgh_v3')
        data['patient_asleep'] = 1
        old_names = ['resp', 'stage']
        new_names = ['Apnea', 'Stage']
        for old_name, new_name in zip(old_names, new_names):
            try: 
                data = data.rename(columns={old_name: new_name})
            except:
                data[new_name] = data[old_name]
                data = data.drop(columns=old_name)
        hdr['patient_tag'] = patient_ID
        assert 'Apnea' in data.columns and 'Stage' in data.columns, 'no labels found'
    except Exception as error: print('Loading error: ', error); return 

    
    
    # compute sleep metrics
    RDI, AHI, CAI, sleep_time = compute_AHI_CAI(data.Apnea.values, data.Stage.values)
    hdr['RDI'], hdr['AHI'], hdr['CAI'], hdr['sleep_time'] = RDI, AHI, CAI, sleep_time

    #### APPLY E METHOD ####
    try:
        data = do_E_method(data, hdr['newFs'])
        hdr['E_SS'] = np.round((np.sum(data['E_sim']==1) / (np.sum(data.patient_asleep==1))) * 100, 1)
    except Exception as error: print('Error applying E method: ', error); return 

    #### APPLY T METHOD #######
    try:
        data = do_T_method(data, hdr, channels)
        hdr['T_SS'] = np.round((np.sum(data['T_sim']==1) / (np.sum(data.patient_asleep==1))) * 100, 1)    
        hdr['T_osc'] = round(len(find_events(data.TAGGED)) / (np.sum(data.patient_asleep==1) / hdr['newFs'] / 3600), 1)
    except Exception as error: 
        print('Error applying T method: ', error); return 

    # save output
    save_output(data, hdr, out_file, dataset)

    # plot signal + both outputs
    both_methods_self_sim_plot(data, hdr, out_path='')



if __name__ == '__main__':
    # set paths and 'datset-to-predict'
    dataset = 'mgh'
    channels = ['abd', 'spo2']  # <---- input signals 
    multiprocess = False

    # get all edf paths in this folder
    if dataset == 'mgh':
        input_folder = './input_data/mgh_DeiD/' #TODO 
        all_paths = glob.glob(input_folder + '*.h5')
        split_recs = select_split_recordings(all_paths, dataset)
        run_paths = [p for p in all_paths if p.split('/')[-1].split('.h5')[0] in split_recs]
            
    # if output folder does not exist, create
    date = datetime.now().strftime('%m_%d_%Y')
    out_file_folder = f'./hf5data/{dataset}_{date}/'
    os.makedirs(out_file_folder, exist_ok=True)

    # run parallel processing using ray
    if multiprocess:
        ray.init(num_cpus=3)

        # create futures
        static_vars = dataset, out_file_folder
        futures = [run_algo_and_plot_multiprocess.remote(p, static_vars) for p in run_paths]

        # compute results
        result = ray.get(futures)

        # close ray
        ray.shutdown()

    # use a loop
    else:
        for path in run_paths:
            # create futures
            run_algo_and_plot_single_run(path, dataset, out_file_folder)
        


        
        
