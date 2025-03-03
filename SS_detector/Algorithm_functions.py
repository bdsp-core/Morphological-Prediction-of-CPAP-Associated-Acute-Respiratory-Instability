import numpy as np
import pandas as pd
import glob
import h5py
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, convolve, resample
from scipy.stats import variation

from get_dbx import get_dbx
dbx_pfx, _, _ = get_dbx()
from Class_Preprocessing import * 

event_tags = [  'None', 'Obstructive apneas', 'Central apneas', \
                'Mixed apneas', 'Hypopneas', 'Partial apneas', 'RERAs']


# setup patient info:

def get_patient_info(txt_file, path, nr, total_paths, hdr, output_file=[]):
    file_info = pd.read_csv(txt_file, sep='\t')

    # identify patient from path
    patient_tag = path.split('/')[-1].split('.')[:-1]
    patient_tag = '.'.join(patient_tag)
    tag = patient_tag.split('~')
    name = tag[0]
    visit_date = tag[1].replace('/', '_').split('$')[0]
    hdr['patient_tag'] = patient_tag

    # print progress
    # percent = round((nr+1)*100/total_paths,2)
    # print('Completed %s%% --> patient %s/%s   '%(percent,nr+1,total_paths), end="\r")

    # skip if already processed
    # if output_file:
    #     skip_already_processed(output_file, patient_tag)

    # open data overview txt file
    rec = np.array(file_info['tag'] == name) * np.array(file_info['visit_date'] == visit_date)
    rec = np.where(rec)

    # raise error if multiple recordings are linked to a patient or removed from research
    if len(rec) > 1:
        raise Exception('Duplicate file found for %s, %s'%(name, visit_date))
    if file_info['research_inclusion'][rec[0][0]] == 'R':
        raise Exception('Recording manually removed from reserach %s, %s'%(name, visit_date))

    # set patient info
    rec = rec[0][0]
    recording_type = file_info['rec_type'][rec]
    test_type = file_info['test_type'][rec]
    if type(hdr['cpap_start']) == type(None): hdr['cpap_start'] = -1
    # cpap_start = file_info['CPAP_tag'][rec]
    # if cpap_start != '-1':
    #     t_new = cpap_start.split('~')
    #     t_old = hdr['cpap_start']
    #     hdr['cpap_start'] = pd.to_datetime(f'{t_old.year}-{t_old.month}-{t_new[0]}\
    #          {t_new[1]}:{t_new[2]}:{t_new[3]}', infer_datetime_format = True) 
    hdr['rec_type'] = recording_type
    if test_type == 'titration': test_type = 'full_cpap'
    hdr['test_type'] = test_type

    return hdr

def skip_already_processed(output_file, patient_tag):
    data = pd.read_csv(output_file, sep='\t',header=0)
    if patient_tag in data['patient_tag'].values:
        raise Exception ('File is already processed :)')


# Creating kernels for RERA's

def create_kernals(all_paths, txt_file, plot=False):
    # run over all test_patients
    print('\nCreate kernels from these recordings:\n')
    run = 0
    for n, file_path in enumerate(all_paths):
        if run == 4:
            break
        if file_path == '/media/cdac/hdd/Auto_respiration_Nassi/default/default_input_data/Hagan_Gregory Neil~02_17_2015.hf5': 
            rec = 'gregory'
        elif file_path == '/media/cdac/hdd/Auto_respiration_Nassi/default/default_input_data/Vanness_Patricia~12_01_2008.hf5':
            rec = 'patricia'
        elif file_path == '/media/cdac/hdd/Auto_respiration_Nassi/default/default_input_data/Gagnier_Roland~2019_05_28.hf5':
            rec = 'roland'
        elif file_path == '/media/cdac/hdd/Auto_respiration_Nassi/default/default_input_data/Sullivan_Scott B~12_03_2017.hf5':
            rec = 'scott'
        else:
            continue

        # load in patient data
        data, hdr = load_sleep_data(file_path, load_all_signals=1)
        hdr = get_patient_info(txt_file, file_path, n, 4, hdr)
        data, _ = create_ventilation_combined(data, hdr, hdr['test_type'])

        # create kernels
        flat_data, non_flat_data = format_manual_kernels(data, rec)

        # save to HF5 files
        output_h5_path1 = '/media/cdac/hdd/Auto_respiration_Nassi/default/Kernels/flat_kernels1.hf5'
        output_h5_path2 = '/media/cdac/hdd/Auto_respiration_Nassi/default/Kernels/non_flat_kernels1.hf5'
        save_kernels(flat_data, output_h5_path1, rec, run=run)
        save_kernels(non_flat_data, output_h5_path2, rec, run=run)
        run +=1

    print('\n**Kernels created**\n')

def format_manual_kernels(data, rec):
    # select according recording
    if rec == 'gregory':
        flat_regions = [[120725,120765],[215760,215820],[215820,215880],[244445,244490]] # [5240,5280],
        non_flat_regions = [[47610,47660],[186545,186590],[266850,266905],[285500,285550]]
    elif rec == 'patricia':
        flat_regions = [[139713,139763]] # [202815,202895]
        non_flat_regions = [[128540,128580],[216925,216970]]
    elif rec == 'roland':
        flat_regions = [[117790,117835],[183315,183345]]
        non_flat_regions = [[5324,5361],[164265,164300]]
    elif rec == 'scott':
        flat_regions = [[238435,238470],[242190,242220],[119660,119690]]
        non_flat_regions = [[164865,164885],[91817,91847]] # ,[228505,228545]
    
    flat_data = []
    non_flat_data = []

    data_lists = [flat_data, non_flat_data]

    regions = [flat_regions, non_flat_regions]
    for i, data_list in enumerate(data_lists):
        # run over flat regions
        for beg, end in regions[i]:
            region = list(range(beg,end))
            kernel, short, extend = regions_to_kernels(data, region)
            # add kernels to dataframe
            data_list.append(kernel)
            # data_list.append(half)
            data_list.append(short)
            data_list.append(extend)
            # data_list.append(double)

    return data_lists[0], data_lists[1]

def regions_to_kernels(data, region):

    # define kernel
    kernel = data.Ventilation_combined[region].values

    # normalize kernels!!
    kernel = (kernel - np.mean(kernel)) / (np.std(kernel) + 0.000001)

    # create 0.75 and 1.25 versions
    # half = resample(kernel, int(0.5*len(kernel)))
    # short = resample(kernel, int(0.75*len(kernel)))
    # extend = resample(kernel, int(1.25*len(kernel)))
    # double = resample(kernel, int(1.5*len(kernel)))

    short = resample(kernel, int(0.75*len(kernel)))
    extend = resample(kernel, int(1.3*len(kernel)))
    
        
    return kernel, short, extend

def save_kernels(data, output_h5_path, rec, default_dtype='float32', run=0):
    chunk_size = 64

    if run==0 and os.path.exists(output_h5_path):
        os.remove(output_h5_path)
    elif os.path.exists(output_h5_path):
        old_data = load_kernels(output_h5_path)
        for d in data:
            old_data.append(d)
        data = old_data
        os.remove(output_h5_path)

    with h5py.File(output_h5_path, 'a') as f:
        # save signals:
        for i, kernel in enumerate(data):
            name = rec + '_' + str(i)
            dset_signal = f.create_dataset(name, shape=(len(kernel),), maxshape=(None,),
                                            chunks=(chunk_size,), dtype=default_dtype)
            dset_signal[:] = kernel.astype(default_dtype)

def load_kernels(path):
    # load file object
    ff = h5py.File(path, 'r')

    data = []
    signals_to_load = list(ff.keys())
    for sig in signals_to_load:
        kernel = ff[sig][:]
        data.append(kernel)

    return data


# Retrieving EEG arousals

def retrieve_EEG_arousals(base_path, data, hdr, region=[], PhysioNet_th=0.3, plot=False):
    Fs = hdr['newFs']
    patient_tag = hdr['patient_tag']

    # add original lables from annotation file
    data = add_original_EEG_arousals(data)

    # add Physionet labels
    # data = add_PhysioNet_labels(base_path, patient_tag, data, region, Fs, PhysioNet_th)

    if plot:
        test_type = hdr['test_type']
        rec_type = hdr['rec_type']
        patient_tag = patient_tag.split('~')[0].replace('_',' ')

        fontdict = {'fontsize':8}
        fig = plt.figure(figsize=(9.6,6))
        
        ax1 = fig.add_subplot(211)
        # ax1.set_title('Respiration trace - %s - %s - %s'%(patient_tag,rec_type,test_type), \
        #                                                     fontdict=fontdict, pad=-1)

        ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep==0), 'k')
        ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep==1), 'r')
        ax1.plot(data.EEG_arousals.mask(data.EEG_arousals==0)*6.1, 'c')


        # plot apnea labels by the experts
        # ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=1)), 'b')
        # ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=2)), 'g')
        # ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=3)), 'c')
        # ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=4)), 'm')
        # ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=5)), 'b')
        # ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=6)), 'r')

        ax1.set_ylim([-7.1,8])
        ax1.set_ylim([-15,15])
        ax1.xaxis.set_visible(False)
        
        ######
        ax2 = fig.add_subplot(212, sharex=ax1)
        # ax2.set_title('Sleep staging - PhysioNet', fontdict=fontdict, pad=-1)
        ax2.xaxis.set_visible(False)
        
        ax2.plot(data.PhysioNet.mask(data.patient_asleep==0),'g')
        data['trash'] = 0.3
        ax2.plot(data.trash,'r')
        ax2.set_ylim([-0.5,1.5])
        # ax2.set_ylabel('PhysioNet ratio')

        # add_ax = ax2.twinx()
        # add_ax.plot(data.Stage,'y')
        # add_ax.set_ylabel('Sleep Staging')

        save_folder = '/media/cdac/hdd/Auto_respiration_Nassi/default/figures/%s'%hdr['patient_tag']
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(save_folder + '/arousals.pdf')

        # ax3 = fig.add_subplot(313, sharex=ax1)
        # ax3.set_title('Various smoothed OSD options', fontdict=fontdict, pad=-1)
        # ax3.plot(data.OSD_smoothed10,'y')
        # ax3.plot(data.OSD_smoothed20,'b')
        # ax3.plot(data.OSD_smoothed30,'g')
        # ax3.set_ylabel('OSD ratio')

        # fig = plt.figure(figsize=(9.6,6))
        # ax1 = fig.add_subplot(711)
        # ax1.plot(data['F3_M2'].mask(data.patient_asleep==0),'b')
        # ax1.plot(data['F3_M2'].mask(data.patient_asleep==1),'r')
        # ax1.set_title('F3_M2', fontdict=fontdict,pad=-1)
        # ax1.set_ylim([-300,300])
        # ax1.xaxis.set_visible(False)

        # channels = ['F4_M1', 'C3_M2', 'C4_M1', 'O1_M2', 'O2_M1']
        # s=2
        # for p, c in enumerate(channels):
        # 	if c in data.columns:
        # 		ax = fig.add_subplot(700+10+s, sharex=ax1)
        # 		ax.plot(data[c].mask(data.patient_asleep==0),'b')
        # 		ax.plot(data[c].mask(data.patient_asleep==1),'r')
        # 		ax.set_title(c, fontdict=fontdict,pad=-1)
        # 		ax.xaxis.set_visible(False) if not p == len(channels)-1 else ax.xaxis.set_visible(True)
        # 		ax.set_ylim([-300,300])
        # 		s += 1

        plt.show()
    
    return data

def add_original_EEG_arousals(data):
    # add arousals from annotations to dataframe
    data['EEG_arousals'] = data.EEG_events_anno.mask(data.patient_asleep==0).fillna(0)

    # remove NaN values
    selected_columns = data[['Ventilation_combined', 'EEG_arousals']]
    new_df = selected_columns.copy()
    new_df = remove_nans(new_df, add_rem=0)
    data['EEG_arousals'] = new_df['EEG_arousals']

    # remove excessive df columns
    data = data.drop(columns=['EEG_events_anno'])

    return data

def add_OSD_data(base_path, patient_tag, data, Fs):
    # define current patient and list all possible OSD files
    input_folder = base_path + '/OSD/OSD_output_data/'
    all_paths = glob.glob(input_folder + '*')
    current_patient = input_folder + patient_tag + '.hf5'

    # add data to dataframe
    if current_patient in all_paths:
        # load data from file
        data_OSD, _ = load_sleep_data(current_patient, load_all_signals=1)
        # smooth
        data_OSD['OSD_smoothed10'] = data_OSD['OSD'].rolling(10*Fs).mean()
        data_OSD['OSD_smoothed20'] = data_OSD['OSD'].rolling(20*Fs).mean()
        data_OSD['OSD_smoothed30'] = data_OSD['OSD'].rolling(30*Fs).mean()

        # add OSD data to dataframe
        data = data.join(data_OSD)
    else:
        data['OSD'] = np.nan
        print('No OSD file found for %s' %(patient_tag))
        
    return data

def add_PhysioNet_labels(base_path, patient_tag, data, region, Fs, PhysioNet_th):
    # define current patient and list all possible PhysioNet files
    input_folder = base_path + '/PhysioNet/PhysioNet_output_data/'
    all_paths = glob.glob(input_folder + '*')
    current_patient = input_folder + patient_tag + '.vec'

    # add data to dataframe
    data['PhysioNet'] = np.nan
    if current_patient in all_paths:
        with open(current_patient,"r") as f:
            array = []
            for line in f:
                array.append(float(line))
        for i, val in enumerate(region[:len(array)]):
            data.loc[i, 'PhysioNet'] = array[val]
        # data.loc[list(range(0,len(array))),'PhysioNet'] = [array[i] for i in region[:len(array)]]
    else:
        print('No PhysioNet file found for %s' %(patient_tag))

    data = create_PhysioNet_labels(data, Fs, PhysioNet_th)
    
    return data

def create_PhysioNet_labels(data, Fs, PhysioNet_th):
    threshold = PhysioNet_th
    data['PhysioNet_labels'] = 0

    # find locations where PhysioNet output > threshold 
    smooth_signal = data['PhysioNet'].rolling(Fs, center=True).mean()
    exceed =  (smooth_signal > threshold) * smooth_signal
    
    # determine peaks and create 3 sec labels
    pos_peaks, _ = find_peaks(exceed, distance=int(Fs*15), width=int(20*Fs), rel_height=1)
    EEG = np.array(smooth_signal[pos_peaks].index)
    for e in EEG:
        e_min = int(e-(1.5*Fs))
        e_max = int(e+(1.5*Fs))
        # skip event if it starts during wake
        if data.patient_asleep[e_min] == 0:
            continue
        # add to dataframe
        region = list(range(e_min, e_max))
        data.loc[region, 'PhysioNet_labels'] = 1

    # apply window correction
    # EEG_labels = exceed.rolling(3*Fs, center=True).mean() == 1
    # EEG_labels = np.array(EEG_labels.mask(data.patient_asleep==0).fillna(0))
    # data['PhysioNet_labels'] = window_correction(EEG_labels, window_size=3*Fs)

    # # connect multiple labels when arousability does not decrease sufficiently
    # minimal_theshold = 0.1
    # eegs = find_events(data['PhysioNet_labels'])
    # for i, eeg in enumerate(eegs[:-1]):
    #     region = list(range(eeg[0],eegs[i+1][1]))
    #     if np.all(smooth_signal[region] > minimal_theshold):
    #         if np.nanmax(smooth_signal[region]) < 2*np.nanmin(smooth_signal[region]):
    #             import pdb; pdb.set_trace()
    #             data.loc[region, 'PhysioNet_labels'] = 1

    return data

def filter_paths_with_PhysioNet(all_paths, base_path):
    Physionet_folder = base_path + '/PhysioNet/PhysioNet_output_data/'
    all_Physionet_paths = glob.glob(Physionet_folder + '*')

    Physionets = [f.split('/')[-1].split('.vec')[0] for f in all_Physionet_paths]
    filtered_paths = []
    counts = 0
    for f in all_paths:
        file = f.split('/')[-1].split('.hf5')[0]
        if np.any([p==file for p in Physionets]) == 0:
            counts += 1
        else:
            filtered_paths.append(f)

    return filtered_paths, counts


# Ventilation Analysis functions:

def set_ventilation_trace(data, hdr):
    Fs = hdr['newFs']
    test_type = hdr['test_type']

    # compute ventilation combined
    data, flowchannels = create_ventilation_combined(data, hdr, test_type)
    for s, signal in enumerate(flowchannels):
        region = list(range(0,len(signal))) if s == 0 else list(range(len(data)-len(signal),len(data)))
        if len(region) > 0:
            # compute envelope and baseline
            new_df = compute_envelope(signal, Fs)
            data.loc[region, 'Ventilation_pos_envelope'] = new_df['pos_envelope'].values
            data.loc[region, 'Ventilation_neg_envelope'] = new_df['neg_envelope'].values
            data.loc[region, 'Ventilation_default_baseline'] = new_df['baseline'].values
            data.loc[region, 'Ventilation_baseline'] = new_df['correction_baseline'].values
            data.loc[region, 'baseline2'] = new_df['baseline2'].values

    return data

def remove_erroneous_ventilation(data, hdr, mask_window=3, signals=[], plot=False):
    Fs = hdr['newFs']
    data['flat_respiration'] = 0
    data['noise_respiration'] = 0

    # check full signal for noise
    full_range = list(range(0,len(data)))
    if check_for_noise(data, Fs, full_range):
        plt.figure(figsize=(9.6,6))
        plt.plot(data['Ventilation_combined'], 'y')
        plt.suptitle('REMOVED DUE TO FULL SIGNAL NOISE DETECTION')
        raise Exception('Full signal is removed due to noise detection!')
    
    std = data['Ventilation_combined'].rolling(30*Fs, center=True).std().fillna(0)

    # set up boundaries for flat signal
    limit = np.nanmax(data['Ventilation_combined']) * .01
    up = limit
    low = -limit
    # determine flat regions
    flat = (std < up) & (std > low)

    # mask regions with duration > 3 min
    flat = flat.astype(int)
    flat_segments = find_events(flat)
    for st, end in flat_segments:
        time = (end - st)
        if time > mask_window*60*Fs:
            st = st - 5*Fs if st - 5*Fs > 0 else 0
            end = end + 5*Fs if end + 5*Fs < len(data) else len(data)
            region = list(range(st, end))
            if not signals:
                signals = [col for col in data.columns if col != 'Apnea']
            data.loc[region, signals] = np.nan
            data.loc[region, 'flat_respiration'] = 1

    if plot:
        plt.figure(figsize=(9.6,6))
        plt.plot(data['Ventilation_combined'].mask(data['flat_respiration']==1), 'y')
        plt.plot(data['Ventilation_combined'].mask(data['flat_respiration']==0), 'b')
        plt.plot(data.flat_respiration*10,'r')
        plt.plot(std, 'k')

        x = list(range(0, len(data)))
        y1 = np.repeat(up, len(data))
        y2 = np.repeat(low, len(data))

        plt.plot(x,y1, 'r')
        plt.plot(x,y2, 'r')
        plt.suptitle('%s trace'%(hdr['patient_tag']))

    return data

def check_for_noise(data, Fs, region):
    is_bool = False
    sig = data.loc[region, 'Ventilation_combined'].fillna(0)

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
    peak = round(np.abs(f_range[np.argmax(power)]),2)
    if peak < 0.1:
        is_bool = True


    # plot power spectrum
    # fig = plt.figure(figsize=(9.6,6))
    
    # ax1 = fig.add_subplot(2,1,1)
    # ax1.set_title('Ventilation trace', fontsize=8, pad=-1)
    # ax1.plot(sig, 'y')

    # ax2 = fig.add_subplot(2,1,2)
    # ax2.set_title('Power spectrum with peak at %s'%(peak), fontsize=8, pad=-1)
    # ax2.plot(f_range, power)
    # ax2.set_xlabel('Freq (Hz)')

    return is_bool

def assess_ventilation(data, hdr, drop_hyp=0.4, drop_apnea=0.9, drop_duration=10, pos_neg=True, plot=False):
    # compute dynamic excursion threshold, both for apnea and hypopneas.
    Fs = hdr['newFs']
    excursion_duration = 20     # the larger the interval, the less dynamic a baseline gets computed.
    excursion_q = 0.90         # ventilation envelope quantile
    pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2, center=True).quantile(excursion_q, interpolation='lower')
    neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2, center=True).quantile(1-excursion_q, interpolation='lower')

    ###############
    # use lagging moving windonw, events are found based on future recovery breaths / eupnea
    pos_excursion1 = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
    pos_excursion1[:-excursion_duration*Fs*2] = pos_excursion1[excursion_duration*Fs*2:]
    pos_excursion1[-excursion_duration*Fs*2:] = np.nan
    neg_excursion1 = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values
    neg_excursion1[:-excursion_duration*Fs*2] = neg_excursion1[excursion_duration*Fs*2:]
    neg_excursion1[-excursion_duration*Fs*2:] = np.nan
    pos_excursion = pos_excursion1
    neg_excursion = neg_excursion1
    ###############

    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_baseline'])

    # set excursion into DF
    data['pos_excursion_apnea'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_apnea))
    data['pos_excursion_hyp'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_hyp))
    data['neg_excursion_apnea'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_apnea))
    data['neg_excursion_hyp'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_hyp))
   
    # find drops in ventilation signal for apneas and hypopneas
    data = locate_ventilation_drops(data, hdr, drop_duration=drop_duration)

    # combine positive and negative excursion flow limitations
    if pos_neg:
        # print(len(find_events(data.pos_Ventilation_drop_hypopnea)))
        data = pos_neg_excursion_combinations(data, hdr)
        # print(len(find_events(data.Ventilation_drop_hypopnea)))
    else:
        data['Ventilation_drop_apnea'] = data.pos_Ventilation_drop_apnea
        data['Ventilation_drop_hypopnea'] = data.pos_Ventilation_drop_hypopnea
        data['soft_ventilation_drop_apnea'] = data.pos_soft_ventilation_drop_apnea

    if plot:
        plt.figure(figsize=(9.5,6)) 

        # airflow signal and baseline
        plt.plot(data.Ventilation_combined.mask(data.patient_asleep==0),'y', lw=0.5)
        plt.plot(data.Ventilation_combined.mask(data.patient_asleep==1),'r', lw=0.5)
        plt.plot(data.Ventilation_baseline.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'k', lw=0.5)

        # other baselines
        # plt.plot(data.Ventilation_default_baseline.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'b')
        # plt.plot(data.baseline2.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'g')
        # plt.plot(data.corrected_baseline.mask(np.isnan(data.Ventilation_combined)),'g')
        
        # airflow envelopes
        plt.plot(pos_excursion,'b--')
        plt.plot(neg_excursion,'b--')

        # apnea / hypopnea threshold lines
        plt.plot(data.pos_excursion_apnea.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'r', lw=0.8)
        plt.plot(data.pos_excursion_hyp.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'m--', lw=0.5)

        plt.plot(data.neg_excursion_apnea.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'r', lw=0.8)
        plt.plot(data.neg_excursion_hyp.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'m--', lw=0.5)

        # original labels
        plt.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=1)), 'b')
        plt.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=4)), 'm')


        #### 
        # plt.plot(-0.5*data.pos_Ventilation_drop_apnea.mask(data.pos_Ventilation_drop_apnea!=1), 'k')
        # plt.plot(-1.5*data.neg_Ventilation_drop_apnea.mask(data.neg_Ventilation_drop_apnea!=1), 'k')
        plt.plot(-data.Ventilation_drop_apnea.mask(data.Ventilation_drop_apnea!=1), 'r')
        plt.plot(-1.25*data.soft_ventilation_drop_apnea.mask(data.soft_ventilation_drop_apnea!=1), 'k')


        plt.plot(-4*data.Ventilation_drop_hypopnea.mask(data.Ventilation_drop_hypopnea!=1), 'm')
        plt.plot(-3.5*data.pos_Ventilation_drop_hypopnea.mask(data.pos_Ventilation_drop_hypopnea!=1), 'k')
        plt.plot(-4.5*data.neg_Ventilation_drop_hypopnea.mask(data.neg_Ventilation_drop_hypopnea!=1), 'k')

        # plt.plot(-6.5*data['just_pos_hypes'].mask(data['just_pos_hypes'] != 1), 'm')
        # plt.plot(-7*data['test_either_hypes'].mask(data['test_either_hypes'] != 1), 'c')

        plt.show()
        import pdb; pdb.set_trace()

    # remove excessive df columns
    # data.drop(columns=['pos_excursion_apnea','pos_excursion_hyp'], inplace=True)  
    # data.drop(columns=['neg_excursion_apnea','neg_excursion_hyp'], inplace=True)  

    return data

def locate_ventilation_drops(data, hdr, drop_duration=10):
    Fs = hdr['newFs']
    # remove NaN values only for selected channels -->
    selected_columns = ['pos_excursion_apnea', 'pos_excursion_hyp', 'Ventilation_baseline',
                        'neg_excursion_apnea', 'neg_excursion_hyp']
    new_df = data[selected_columns + ['Ventilation_combined']].copy()
    new_df['Ventilation_combined'] = new_df['Ventilation_combined'].mask(data.patient_asleep==0)
    new_df = remove_nans(new_df, add_rem=0)

    # put selected columns in original dataframe
    for col in selected_columns:
        data[col] = new_df[col]

    try:
        # find areas with potential apnea / hypopnea flow limitations
        data['pos_Ventilation_drop_apnea'] = (data.Ventilation_combined < data.pos_excursion_apnea) 
        data['pos_Ventilation_drop_hypopnea'] = (data.Ventilation_combined < data.pos_excursion_hyp) 
        data['neg_Ventilation_drop_apnea'] = (data.Ventilation_combined > data.neg_excursion_apnea) 
        data['neg_Ventilation_drop_hypopnea'] = (data.Ventilation_combined > data.neg_excursion_hyp) 
    except: import pdb; pdb.set_trace()
    # run over the various ventilation drop options, and find flow limitations
    tag_window = [drop_duration//2, drop_duration, drop_duration]
    for ex in ['pos', 'neg']:
        tag_list = ['%s_soft_ventilation_drop_apnea'%ex, 
                    '%s_Ventilation_drop_apnea'%ex, 
                    '%s_Ventilation_drop_hypopnea'%ex]
        data = find_flow_limitations(data, tag_list, tag_window, Fs)

    return data
    
def find_flow_limitations(data, tag_list, tag_window, Fs):
    for t, tag in enumerate(tag_list):
        win = int(tag_window[t]*Fs)
        # find events with duration <win>
        if t==0:
            col = [c for c in tag_list if 'Ventilation_drop_apnea' in c]
            data[tag] = data[col].rolling(win, center=True).mean() == 1
        else:
            data[tag] = data[tag].rolling(win, center=True).mean() == 1

        # apply window correction
        data[tag] = np.array(data[tag].mask(data.patient_asleep==0).fillna(0))
        data[tag] = window_correction(data[tag], window_size=win)

        # remove all apnea events with a duration > .. sec
        data = remove_long_events(data, tag, Fs, max_duration=60)

    return data

def pos_neg_excursion_combinations(data, hdr):
    # run over apnea options
    for col in ['Ventilation_drop_apnea', 'soft_ventilation_drop_apnea', 'Ventilation_drop_hypopnea']:
        # for soft apneas, pos and neg flow limitation has to occur simultaniously
        if 'soft' in col:
            data[col] = (data['pos_%s'%col] * data['neg_%s'%col]) > 0
            # remove events < 4 sec
            data[col] = remove_short_events(data[col], 4*hdr['newFs'])
        elif 'hypopnea' in col:
            data[col] = 0
            data = hyp_flow_limitations(data, col)
        # for apneas / hypopneas include both pos or neg flow limitations 
        else:
            data[col] = (data['pos_%s'%col] + data['neg_%s'%col]) > 0
        data[col] = data[col].astype(int)

        # connect apneas, within 5 sec (only if event < 20sec)
        events = find_events(data[col].fillna(0))
        if len(events) == 0: continue
        events = connect_events(events, 5, hdr['newFs'], max_dur=20)
        data[col] = events_to_array(events, len(data))

    return data

def hyp_flow_limitations(data, col):
    both = (data['pos_%s'%col] * data['neg_%s'%col]) > 0
    either = (data['pos_%s'%col] + data['neg_%s'%col]) > 0

    # run over flow limitation regions
    for st, end in find_events(either.fillna(0)):
        region = list(range(st, end))
        
        # if both a pos and negative excursion was found... 
        if np.any(both.loc[region] > 0):
            pos_events = find_events(data.loc[region, 'pos_%s'%col])
            if len(pos_events) > 1: 
                neg_events = find_events(data.loc[region, 'neg_%s'%col])
                
                #.. save middle are if, if possible
                if len(neg_events) > 1:
                    boths = find_events(both.loc[region])
                    loc = (boths[0][0], boths[-1][1])
                
                # .. or save negative region
                else:
                    loc = neg_events[0]
            
            # .. or save positive region
            else:
                loc = pos_events[0]
            loc = list(range(loc[0], loc[1]))
            if loc[-1] == len(region): loc = loc[:-1]
            data.loc[np.array(region)[loc], col] = 1
        
        # if only a pos or neg excursion was found, save that region
        else:
            data.loc[region, col] = 1
        
    return data

def connect_events(events, win, Fs, max_dur=False):
    new_events = []
    cnt = 0
    while cnt < len(events)-1:
        st = events[cnt][0]
        end = events[cnt][1]
        dist = events[cnt+1][0] - end 
        condition = dist < win if max_dur == False else (dist<win) * ((end-st)*Fs > max_dur*Fs)
        if condition:
            new_events.append((st, events[cnt+1][1]))
            cnt +=2
        else:
            new_events.append((st, end))
            cnt +=1 
    new_events.append((events[-1]))

    return new_events

def compute_envelope(signal, Fs, base_win=30, env_smooth=10):
    new_df = pd.DataFrame()
    new_df['x'] = signal

    # determine peaks of signal
    x = new_df['x']
    pos_peaks, _ = find_peaks(x, distance=int(Fs*1.5), width=int(0.4*Fs), rel_height=1)
    neg_peaks, _ = find_peaks(-x, distance=int(Fs*1.5), width=int(0.4*Fs), rel_height=1)

    new_df['pos_envelope'] = x[x.index[0] + pos_peaks]
    new_df['neg_envelope'] = x[x.index[0] + neg_peaks]

    # compute envelope of signal
    new_df['pos_envelope'] = new_df['pos_envelope'].interpolate(method='cubic', order=2, limit_area='inside')
    new_df['neg_envelope'] = new_df['neg_envelope'].interpolate(method='cubic', order=2, limit_area='inside')        
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
    
    # compute correction ratio according to difference between envelope and baseline
    # new_df['pos_distance'] = np.abs(pos - base)
    # new_df['neg_distance'] = np.abs(neg - base)
    # pos_distance = new_df['pos_distance'].rolling(30*Fs, center=True).median()
    # neg_distance = new_df['neg_distance'].rolling(30*Fs, center=True).median()

    # p_shifts = pos_distance < neg_distance
    # n_shifts = pos_distance > neg_distance

    # correction_ratio[p_shifts] = neg_distance[p_shifts] / pos_distance[p_shifts]
    # correction_ratio[n_shifts] = pos_distance[n_shifts] / neg_distance[n_shifts]

    # new_df.loc[p_shifts,'correction_baseline'] = base[p_shifts] + (base[p_shifts] * correction_ratio[p_shifts] / 2)
    # new_df.loc[n_shifts,'correction_baseline'] = base[n_shifts] + (base[n_shifts] * correction_ratio[n_shifts] / 2)



    return base1, base2, base_corr


# Hypopnea Analysis functions:

def determine_saturation_drops(data, hdr, sat_drop):
    Fs = hdr['newFs']
    ranges = np.array([5, 10, 20, 30])*Fs
    data['saturation_drop'] = 0
    data['saturation_return'] = 0
    # find saturation drops using various windows in sequence
    for ran in ranges:
        # set thresholds 
        thresh = round(Fs/ran, 3)
        sat_min = data['SpO2'].rolling(ran, center=True).quantile(thresh, interpolation='lower')
        sat_max = data['SpO2'].rolling(ran, center=True).quantile(1-thresh, interpolation='higher')

        # determine drops
        data['potential_saturation_drop']  = np.array(((sat_max - sat_min) >= sat_drop)) * 1

        # run over all drops
        drops = find_events(data['potential_saturation_drop'])

        for drop in drops:
            region = list(range(drop[0], drop[1]))
            # skip when drop includes wake time
            # if np.any(data.patient_asleep[region]==0):
            #     continue
            # skip when drop is already found using a smaller window --> prev iteration
            if not np.all(data.saturation_drop[region] == 0):
                beg = np.where(data.saturation_drop[region] == 1)[0][0]
                end = np.where(data.saturation_drop[region] == 1)[0][-1]
                if beg == 0:
                    area = list(range(drop[0]+end+1, region[-1]))
                else:
                    area = list(range(drop[0], drop[0]+beg))
                # skip invalid area's
                if (np.max(data['SpO2'][area]) - np.min(data['SpO2'][area]) < sat_drop) or len(area)==0:
                    continue
            else:
                h_w = ran//2
                area = list(range(drop[0]-h_w, drop[1]+h_w))
            # cut area if it exceeds signal length
            if area[-1] > len(data)-1:
                area = list(range(area[0], len(data)))

            # define search area 
            f_area = area[:len(area)//2]
            l_area = area[len(area)//2:]
            # compute smooth sautration
            smooth_saturation = data['SpO2'].rolling(Fs, center=True).median()
            # locate 's'tart and 'e'nd location of drop
            s = np.where(smooth_saturation[f_area] == np.max(smooth_saturation[f_area]))[0][-1]
            if s > (len(f_area)-Fs):
                s = np.where(smooth_saturation[area] == np.max(smooth_saturation[area]))[0][-1]
            e = np.where(smooth_saturation[l_area] == np.min(smooth_saturation[l_area]))[0][-1]
            fill = list(range(area[0]+s, len(area)//2+area[0]+e))
            
            # remove drop if length or thresholds are incorrect
            if (np.max(data['SpO2'][fill[:Fs]]) - np.min(data['SpO2'][fill[-Fs:]]) < sat_drop) or len(fill) < 2*Fs:
                data.loc[fill, 'saturation_return'] = 1 
                continue
            else:
                # skip if saturation drop includes wake time
                # if np.any(data['patient_asleep'][fill]==0):
                #     continue
                data.loc[fill, 'saturation_drop'] = 1              

    # remove excessive df columns
    data.drop(columns='potential_saturation_drop', inplace=True)

    return data

def match_saturation_and_ventilation_drops(data, hdr, hyp_range=30):
    Fs = hdr['newFs']
    data['potential_saturation_drops'] = data['saturation_drop']
    data['accepted_saturation_hypopneas'] = 0
    data['rejected_saturation_hypopneas'] = 0
    potential_hyp = find_events(data['Ventilation_drop_hypopnea'].fillna(0))

    # run backwards through all events
    potential_hyp = potential_hyp
    hyp_range = hyp_range*Fs
    
    for p_h in potential_hyp:
        # define ventilation drop as potential hypopnea
        hyp = list(range(p_h[0], p_h[1])) 

        # specify region to find desaturation drops
        region = define_saturation_search_region(p_h, hyp_range, data, Fs)
       
        # match potential event with potential saturation drop
        tag = 'potential_saturation_drops'
        found = hypopnea_match_making(data, region, tag)

        # save potential drop in according df column
        if found == True:
            data.loc[hyp, 'accepted_saturation_hypopneas'] = 1
        else:
            data.loc[hyp, 'rejected_saturation_hypopneas'] = 1

    # remove excessive df columns
    data = data.drop(columns='potential_saturation_drops')

    return data

def match_EEG_with_ventilation_drops(data, hdr, hyp_range=20):
    data['potential_EEG_arousals'] = data['EEG_arousals']
    data['accepted_EEG_hypopneas'] = 0
    data['rejected_EEG_hypopneas'] = 0
    potential_hyp = find_events(data['Ventilation_drop_hypopnea'].fillna(0))
    hyp_range = int(hyp_range*hdr['newFs'])
    
    for p_h in potential_hyp:
        hyp = list(range(p_h[0], p_h[1])) 

        # specify region to find desaturation drops
        region = define_EEG_search_region(p_h, hyp_range, data)
        if len(region) == 0:
            continue
            
        # match potential event with potential arousal
        tag = 'potential_EEG_arousals'
        found = hypopnea_match_making(data, region, tag)

        # save potential drop in according df column
        if found == True:
            data.loc[hyp, 'accepted_EEG_hypopneas'] = 1
        else:
            data.loc[hyp, 'rejected_EEG_hypopneas'] = 1
            
    # remove excessive df columns
    data = data.drop(columns='potential_EEG_arousals')

    return data

def define_saturation_search_region(p_h, hyp_range, data, Fs):
    # define ventilation drop as potential hypopnea
    hyp = list(range(p_h[0], p_h[1])) 
    # specify optional region end to search for saturation drop
    end = (p_h[0]+p_h[1])//2 + hyp_range if (p_h[0]+p_h[1])//2 + hyp_range < data.shape[0] else data.shape[0]
    region = list(range(p_h[0], int(end))) 
    # increase range search range if hypopnea > 45sec
    if len(hyp) > len(region):
        region = hyp

    # find and group possible upcoming events
    all_apneas = data.loc[region, 'Ventilation_drop_apnea'].mask(data.loc[region, 'Ventilation_drop_apnea'] == 0).fillna(0)
    hypopneas = data.loc[region, 'Ventilation_drop_hypopnea'].mask(data.loc[region, 'Ventilation_drop_hypopnea'] == 0).fillna(0)
    events = all_apneas + hypopneas
    events[events>0] = 1
    events = find_events(events)

    # if an event is found, shorten search region upto start of next event
    if len(events) > 1:
        region = list(range(p_h[0], p_h[0]+events[1][0]+5*Fs))

    return region

def define_EEG_search_region(p_h, hyp_range, data):
    # define ventilation drop as potential hypopnea
    hyp = list(range(p_h[0], p_h[1]))
    # specify region to search for saturation drop
    end = p_h[1] + hyp_range if p_h[1] + hyp_range < data.shape[0] else data.shape[0]
    region = list(range(p_h[1], end)) 

    # limit search region upto next already verified saturation hypopnea
    # if np.any(data.loc[region,'accepted_saturation_hypopneas'].fillna(0) != 0):
    #     verified_loc = np.where(data.loc[region,'accepted_saturation_hypopneas'].fillna(0) != 0)[0][0]
    #     if verified_loc == 0:
    #         return []
    #     region = list(range(p_h[1], p_h[1]+verified_loc)) 
    
    # increase range search range if hypopnea > 45sec
    if len(hyp) > len(region):
        region = hyp 

    return region

def hypopnea_match_making(data, region, tag):
    # find associated saturation drops
    segment = data[tag][region]
    desaturations = find_events(segment.fillna(0))
    
    # run over found desaturation drops
    found = False
    for desat in desaturations:
        if desat[0] == 0:
            continue
        # only use desats that initiated after flow limitation
        found = True 

    return found


# RERA analysis functions:

def RERA_detection(data, hdr, quantile, plot=False):
    Fs = hdr['newFs']
    test_type = hdr['test_type']
    rec_type = hdr['rec_type']
    patient_tag = hdr['patient_tag'].split('~')[0].replace('_',' ')

    data['algo_reras'] = 0
    data['RERA_morphology'] = np.nan
    data['RERA_morphology_score_flat'] = np.nan
    data['RERA_morphology_score_nonflat'] = np.nan

    # return to code if no EEG arousals are found
    eeg_arousals = find_events(data['PhysioNet_labels'].fillna(0))
    if len(eeg_arousals) == 0:
        return data

    # load all flat kernels
    kernel_paths = glob.glob('/media/cdac/hdd/Auto_respiration_Nassi/default/Kernels/*.hf5')
    for path in kernel_paths:
        kernels = load_kernels(path)
        if "non_flat_kernels" in path:
            kernels_nonflat = kernels
        else:
            kernels_flat = kernels

    # locate inspiratory flattening using pre-defined kernels
    data = find_RERA_morphology(data, eeg_arousals, kernels_flat, kernels_nonflat, Fs, quantile=quantile)

    # locate effort increase in effort belts
    data = find_increased_inspiratory_effort(data, hdr)

    # create labels
    data = create_RERA_labels(data, eeg_arousals, Fs)

    if plot:
        # plot the first 9 kernels
        plot_kernels(kernels_flat, kernels_nonflat)

        plt.show()

        fig = plt.figure(figsize=(9.6,6))
        fontdict = {'fontsize':9}

        #################
        ax1 = fig.add_subplot(411)
        ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep==0), 'y')
        ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep==1), 'r')
        ax1.plot(data.EEG_arousals.mask(data.EEG_arousals==0)*6.1, 'c')


        # plot apnea labels by the experts
        ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=1)), 'b')
        ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=2)), 'g')
        ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=3)), 'c')
        ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=4)), 'm')
        ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=5)), 'b')
        ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=6)), 'r')

        ax1.set_ylim([-7.1,8])
        ax1.set_title('Respiration trace - %s - %s - %s'%(patient_tag,rec_type,test_type), \
                                                            fontdict=fontdict, pad=-1)
        
        #################
        ax2 = fig.add_subplot(412, sharex=ax1)
        ax2.plot(data['ABD'].mask(data.patient_asleep==0),'b')
        ax2.plot(data['ABD'].mask(data.patient_asleep==1),'r')
        ax2.plot(data['CHEST'].mask(data.patient_asleep==0),'g')
        ax2.plot(data['CHEST'].mask(data.patient_asleep==1),'r')

        ax2.set_ylim([-10,10])

        #################
        ax3 = fig.add_subplot(413, sharex=ax1)

        smooth_PhysioNet= data['PhysioNet'].rolling(Fs, center=True).mean()
        ax3.plot(smooth_PhysioNet.mask(data.patient_asleep==0),'b')
        ax3.plot(smooth_PhysioNet.mask(data.patient_asleep==1),'r')

        ax3.plot(data.PhysioNet_labels.mask((data.patient_asleep==0) | (data.PhysioNet_labels!=1)), 'c')
        ax3.plot(data.algo_reras.mask(data.algo_reras!=6)/6*1.1, 'r')

        # ax3.plot(data.test.mask(data.test==0)*0.8, 'k')

        # ax3.scatter(data.index, data['RERA_morphology']*1.1, c='g', s=4)
        ax3.plot(data['RERA_morphology_smooth'].mask(data.RERA_morphology_smooth==0)*1.2, 'g')
        ax3.plot(data['RERA_increased_effort'].mask(data['RERA_increased_effort']==0)*1.3, 'm')

        # ax3.plot(data.EEG_arousals.mask(data.EEG_arousals==0)*1.7, 'b')
        ax3.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=6))/6*1.5, 'r')
        ax3.set_ylabel('arousability index')

        add_ax = ax3.twinx()
        add_ax.set_ylim([0,5])
        add_ax.plot(data.Stage.mask(data.patient_asleep==0),'y')
        add_ax.set_ylabel('sleep staging')

        ax3.set_ylim([-0.1,1.7])
        ax3.set_title('PhysioNet - sleep staging', fontdict=fontdict, pad=-1)

        #################
        ax4 = fig.add_subplot(414, sharex=ax1)
        
        ax4.scatter(data.index, data['RERA_morphology_score_flat'].values, c='r', s=4)
        ax4.scatter(data.index, data['RERA_morphology_score_nonflat'].values, c='k', s=4)

        ax4.set_title('Best scores, flat (r) vs non-flat (k)', fontdict=fontdict, pad=-1)

        import pdb; pdb.set_trace()

    return data

def find_RERA_morphology(data, locs, kernels_flat, kernels_nonflat, Fs, quantile=0.8, rera=0):
    # run over all EEG_spontaneous arousals
    for st, end in locs:
        # specify region of interest
        candidate = data.Ventilation_combined[st-10*Fs : end+5*Fs]
        if np.all(np.isnan(candidate)):
            continue

        # apply kernels to data
        _, conv_score_flat_kernels = apply_kernels(candidate.values, kernels_flat, Fs)
        _, conv_score_nonflat_kernels = apply_kernels(candidate.values, kernels_nonflat, Fs)

        # compute mean kernal scores
        score_flat = np.quantile(conv_score_flat_kernels, quantile, axis=0)
        score_nonflat = np.quantile(conv_score_nonflat_kernels, quantile, axis=0)
        
        # compare best flat vs best non-flat convolutions
        is_flat = score_flat > score_nonflat
        # also remove results with general low convolution score, they are probably not related to a breath 
        is_flat[score_flat < 0.65] = 0  
        is_flat = is_flat.astype(float)

        # replace zeros with NaN's
        score_flat[score_flat==0] = np.nan      
        score_nonflat[score_nonflat==0] = np.nan  
        is_flat[is_flat==0] = np.nan 

        data.loc[candidate.index, 'RERA_morphology'] = is_flat

        data.loc[candidate.index, 'RERA_morphology_score_flat'] = score_flat
        data.loc[candidate.index, 'RERA_morphology_score_nonflat'] = score_nonflat

    return data

def apply_kernels(candidate, kernels, Fs):
    # initialize empty variables
    best_scores = np.zeros(candidate.shape)
    individual_scores = []

    # run through all kernels
    for kernel in kernels: 
        # run through the entire candidate before an arousal annotation 
        ind_conv_score = np.zeros(candidate.shape)
        for i_start in range(0, len(candidate) - len(kernel), Fs//2): 

            # select segment and normalize!
            candidate_seg = candidate[i_start : i_start + len(kernel)]
            candidate_seg = (candidate_seg - np.mean(candidate_seg)) / (np.std(candidate_seg) + 0.000001)
            if np.all(np.isnan(candidate_seg)):
                continue
            # apply convolution
            convolution_score = np.nanmax(convolve(candidate_seg, kernel, mode='same')) / len(kernel)
            
            # save individual kernel scores (only positive peaks)
            center_index = i_start + len(kernel)   #//2
            if convolution_score > ind_conv_score[center_index]:
                # update, this kernel yields a high score on this position:
                ind_conv_score[center_index] = convolution_score         

            # save highest covultion score among all kernels
            if convolution_score > best_scores[center_index]:
                best_scores[center_index] = convolution_score

        # append individual kernel scores to list
        individual_scores.append(ind_conv_score)

    return best_scores, individual_scores

def find_increased_inspiratory_effort(data, hdr, insp_effort_thresh=1.5, plot=False):
    
    Fs = hdr['newFs']

    # compute envelope and baseline for effort traces
    effort = data[['ABD','CHEST']].mean(axis=1).rolling(Fs, center=True).median()
    new_df = compute_envelope(effort, Fs, env_smooth=20)
    baseline = new_df['baseline']
    pos_envelope = new_df['pos_envelope']

    # find sudden high peak locations
    dist_to_base = pos_envelope - baseline
    effort[effort<insp_effort_thresh] = 0
    data['RERA_increased_effort'] = effort > (2*dist_to_base)

    if plot:
        # plot inspiratory effort increase
        fig = plt.figure(figsize=(9.6,6))
        fontdict = {'fontsize':9}
        ax1 = fig.add_subplot(211)
        # abd
        ax1.set_title('Abdominal trace', fontdict=fontdict, pad=-1)
        ax1.plot(data['ABD'].mask(data.patient_asleep==0),'b')
        ax1.plot(data['ABD'].mask(data.patient_asleep==1),'r')
        # chest
        ax1.plot(data['CHEST'].mask(data.patient_asleep==0),'g')
        ax1.plot(data['CHEST'].mask(data.patient_asleep==1),'r')
        # RERA and PhysioNet labels
        apneas = data.Apnea.mask((data.patient_asleep==0) | (data.Apnea==6))
        apneas[apneas>0] = 5
        ax1.plot(apneas, 'k')
        ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=6)), 'r')
        ax1.plot(data.PhysioNet_labels.mask(data.PhysioNet_labels==0)*7, 'm')
        
        # plot effort sum and detected increase
        ax2 = fig.add_subplot(212,sharex=ax1)
        ax2.set_title('effort analysis trace', fontdict=fontdict, pad=-1)
        ax2.plot(effort,'y')
        ax2.plot(baseline.mask(data.patient_asleep==0),'k')
        ax2.plot(pos_envelope.mask(data.patient_asleep==0),'b')

        ax2.plot(data['RERA_increased_effort'].mask(data['RERA_increased_effort']==0)*5, 'r')

    return data 

def create_RERA_labels(data, eeg_arousals, Fs):
    # smooth morphology output
    morphology_labels = data['RERA_morphology'].fillna(0).values.astype(int)
    data['RERA_morphology_smooth'] = smooth(morphology_labels, win=Fs, zeros=int(0.9*Fs))
    
    # run over all events 
    for start, end in eeg_arousals:
        start -= 10*Fs 
        end += 10*Fs
        # search region for morphology reras
        st = start if start-5*Fs >= 0 else 0
        e = end + 1 if end + 1 <= len(data) else len(data)
        region1 = list(range(st, e))
        morph_events = find_events(data.loc[region1, 'RERA_morphology_smooth'].fillna(0)) 
        # search region for effort reras
        st = start if start >= 0 else 0
        e = end+5*Fs if end+5*Fs <= len(data) else len(data)
        region2 = list(range(st, e))
        effort_segment = data.loc[region2, 'RERA_increased_effort'].astype(int)

        if len(morph_events) > 0:
            try:
                label_range = list(range(region1[morph_events[0][0]], region1[morph_events[-1][1]]))
            except:
                label_range = list(range(region1[morph_events[0][0]], region1[morph_events[-1][1]-1]))
            # include all events with a duration > 10 sec
            if len(label_range) >= 10*Fs:
                data.loc[label_range, 'algo_reras'] = 6
                continue

        if np.any(effort_segment == 1):
            effort_segment.iloc[-1] = 0
            events = find_events(effort_segment.fillna(0))
            for i, ev in enumerate(events): 
                region = list(range(start, start+ev[1]))
                if len(region) < 10*Fs and i < len(events)-1:
                    continue
                else:
                    data.loc[region, 'algo_reras'] = 6

    return data


# Apnea Comparison functions: 

def assess_apnea_type(data, hdr, plot=False, apnea_hyp=[]):
    Fs = hdr['newFs']   
    if not 'algo_reras' in data.columns: data['algo_reras'] = 0

    # remove all hypopneas that overlap with an apnea
    data = overlapping_apneas_check(data, hdr, Fs)
        
    # group all apneas
    all_apneas = data['Ventilation_drop_apnea'].fillna(0)
    grouped_apneas = find_events(all_apneas)

    # run over all found apnea events
    for st, end in grouped_apneas:
        # specify region of interest
        region = list(range(st, end))
        middle = (region[0]+region[-1])//2
        s = middle - 40*Fs; e = middle + 40*Fs
        complete_window = list(range(s,e))
        # correct for beginnnig and end of recording
        if s<0 or e>=len(data):
            dist = np.min([middle, len(data) - middle])
            complete_window = list(range(middle - dist, middle + dist)) 
            if len(complete_window) < 10*Fs:
                continue
        first_half = list(range(middle-10*Fs, middle))
        second_half = list(range(middle, middle+10*Fs))
 
        # run over all signal segments of interest
        window_list = [complete_window, first_half, second_half]
        inspiratory_ratios = []; abd_base = 0; chest_base = 0
        for i, window in enumerate(window_list):
            # compute smoothed inspiratory effort belt signals
            abd = data.loc[window,'ABD'].rolling(int(0.5*Fs), center=True).median().fillna(0)
            chest = data.loc[window,'CHEST'].rolling(int(0.5*Fs), center=True).median().fillna(0) 
            
            # compute inspiratory effort coefficient
            abd_ratio, abd_base = apnea_effort_ratio(abd, Fs, i=i, base=abd_base)
            chest_ratio, chest_base = apnea_effort_ratio(chest, Fs, i=i, base=chest_base)
            inspiratory_ratios.append(np.mean([abd_ratio, chest_ratio]))
        
        if not apnea_hyp:
            insp_ratio_thresh = 10      #20
            mixed_factor_thresh = 8     # 3
        else:
            insp_ratio_thresh = apnea_hyp[0]
            mixed_factor_thresh = apnea_hyp[1]

        # all events with large decrease in inspiratory effort are set to central apnea
        if inspiratory_ratios[0] > insp_ratio_thresh:
            # all events with increased effort during latter half of the event are set to mixed apnea
            if mixed_factor_thresh*inspiratory_ratios[1] < inspiratory_ratios[2]:
                data.loc[region, 'Ventilation_drop_apnea'] = 3
            else:
                data.loc[region, 'Ventilation_drop_apnea'] = 2
        
        if plot:
            fontdict = {'fontsize':8}
            plt.figure(figsize=(7.6,6))

            # create datetime index for dataframe
            # data.index = pd.date_range(hdr['start_time'], periods=data.shape[0], freq=str(1/Fs)+'S')
            ###

            # plot signal
            plt.plot(data.loc[complete_window,'ABD'].rolling(int(0.5*Fs), center=True).median(), 'b')
            plt.plot(data.loc[complete_window,'CHEST'].rolling(int(0.5*Fs), center=True).median(), 'g') 

            print(inspiratory_ratios[0])

            # plot apneas found by algorithm
            # plt.plot(-data['Ventilation_drop_apnea'][region].mask(data['Ventilation_drop_apnea'] != 1),'b')
            # plt.plot(-data['Ventilation_drop_apnea'][region].mask(data['Ventilation_drop_apnea'] != 2),'g')
            # plt.plot(-data['Ventilation_drop_apnea'][region].mask(data['Ventilation_drop_apnea'] != 3),'c')
            # plt.plot(-data['algo_hypopneas_three'][region].mask(data['algo_hypopneas_three'] != 4),'m')

            # plot original apnea labels
            # plt.plot(data.Apnea[region].mask((data.patient_asleep==0) | (data.Apnea!=1)), 'b')
            # plt.plot(data.Apnea[region].mask((data.patient_asleep==0) | (data.Apnea!=2)), 'g')
            # plt.plot(data.Apnea[region].mask((data.patient_asleep==0) | (data.Apnea!=3)), 'b')
            # plt.plot(data.Apnea[region].mask((data.patient_asleep==0) | (data.Apnea!=4)), 'm')
            # plt.plot(data.Apnea[region].mask((data.patient_asleep==0) | (data.Apnea!=5)), 'k')
            # plt.plot(data.Apnea[region].mask((data.patient_asleep==0) | (data.Apnea!=6)), 'r')

            # plt.title('Segment with apnea (found by algorithm)', fontdict=fontdict,pad=-1)

            if np.any(data.Apnea[region]== 3):
                plt.show()
                import pdb; pdb.set_trace()

            
  
    if 'algo_hypopneas_four' in data.columns:
        # prioritize 4% over 3% events
        data = four_over_three_hypopnea_priority(data)

        # create algo output column
        data['algo_apneas'] = data['Ventilation_drop_apnea'] + data['algo_hypopneas_three'] + \
                                data['algo_hypopneas_four'] + data['algo_reras']
        # data = data.drop(columns=['algo_hypopneas','Ventilation_drop_apnea'])

        if np.any(data['algo_apneas'] > 6):
            print('3%% - 4%% discrimination problem!!')
            import pdb; pdb.set_trace()
    else:
        # create algo output column
        data['algo_apneas'] = data['Ventilation_drop_apnea'] + data['algo_hypopneas_three'] + \
                                data['algo_reras']

    return data

def overlapping_apneas_check(data, hdr, Fs):

    # define apnea location arrays
    all_apneas = data['Ventilation_drop_apnea'].fillna(0)
    soft_aneas = data['soft_ventilation_drop_apnea'].fillna(0)
    threes = data['algo_hypopneas_three'].fillna(0)
    hyps = ['algo_hypopneas_three', 'algo_hypopneas_four'] if 'algo_hypopneas_four' in data.columns else ['algo_hypopneas_three']
    data['hypopneas'] = data[hyps].any(axis=1, skipna=False).astype(int)
    hypopneas = data['hypopneas'].fillna(0)

    reras = data['algo_reras'].fillna(0)
    data['3%_reras'] = 0
    # remove all reras that overlap with an apnea
    grouped_reras = find_events(reras)
    for st, end in grouped_reras:
        region = list(range(st, end))
        if np.any(threes[region] > 0):
            data.loc[region, '3%_reras'] = 1
        if np.any(all_apneas[region] == 1) or np.any(hypopneas[region] > 0):
            data.loc[region, 'algo_reras'] = 0

    # remove all hypopneas that overlap with an apnea
    grouped_hypopneas = find_events(hypopneas)
    for st, end in grouped_hypopneas:
        region = list(range(st, end))
        if np.any(all_apneas[region] == 1):
            data.loc[region, hyps] = 0
            data.loc[region, 'Ventilation_drop_apnea'] = 1
            
            # remove to make plot less crowded
            data.loc[region, 'accepted_saturation_hypopneas'] = 0
            data.loc[region, 'accepted_EEG_hypopneas'] = 0

        # convert soft apneas (>5sec of 90% excursion decrease) into apneas @ locations where hypopnea is found
        elif np.any(soft_aneas[region] == 1) and np.sum(soft_aneas[region]) > 0.5*len(region):
            data.loc[region, hyps] = 0
            data.loc[region, 'Ventilation_drop_apnea'] = 1  

            # remove to make plot less crowded
            data.loc[region, 'accepted_saturation_hypopneas'] = 0
            data.loc[region, 'accepted_EEG_hypopneas'] = 0

    data.drop(columns=['hypopneas'], inplace=True)

    return data

def apnea_effort_ratio(signal, Fs, i=0, base=0):
    if i==0:    
        # compute envelope and baseline
        envelope_data = compute_envelope(signal, Fs, base_win=5, env_smooth=5)
        pos_envelope = envelope_data['pos_envelope']
        baseline = envelope_data['correction_baseline']
        # compute min and max
        distance_to_baseline = pos_envelope - baseline
        max_val = np.nanquantile(distance_to_baseline,0.8) if np.nanquantile(distance_to_baseline,0.8) < 100 else 100
        min_val = np.nanquantile(distance_to_baseline,0.1) if np.nanquantile(distance_to_baseline,0.1) > 0 else 0.01
        # save base for next runs
        base = baseline[(baseline.index[0]+baseline.index[-1])//2]
    else:
        signal[signal<base] = base
        distance_to_baseline = signal - base
        max_val = np.nanquantile(distance_to_baseline,0.9) if np.nanquantile(distance_to_baseline,0.9) < 100 else 100
        min_val = np.nanquantile(distance_to_baseline,0.1) if np.nanquantile(distance_to_baseline,0.1) > 0 else 0.01

    # determine ratio around apnea vs during apnea

    ratio = max_val / min_val

    return ratio, base

def four_over_three_hypopnea_priority(data):
    # 4% gets priority over 3% rule
    three = 'algo_hypopneas_three'
    four = 'algo_hypopneas_four'
    threes = find_events(data[three])
    # run over all 3% hypopneas
    for st, end in threes:
        region = list(range(st, end))
        # remove 3% event if 4% event is found
        if np.any(data.loc[region, four] == 4):
            data.loc[region, three] = 0
    
    # reset value of 4% to '5'
    data.loc[data[four]==4, four] = 5

    return data


# Post-Processing functions:

def post_processing(data, hdr, plot=False):
    # Fs = hdr['newFs']


    return data

def remove_long_events(data, tag, Fs, max_duration=60):
    data['too_long_events'] = 0
    # find and remove events with duration > 'max_duration'
    events = find_events(data[tag])
    for st, end in events: 
        region = list(range(st, end))
        if len(region) > max_duration*Fs:
            data.loc[region, tag] = 0
            data.loc[region, 'too_long_events'] = 1

    return data

def remove_short_events(array, duration):
    # find and remove events with duration < 'duration'
    events = find_events(array)
    array = array.values
    for st, end in events: 
        region = list(range(st, end))
        if len(region) < duration:
            array[region] = 0

    return array

def smooth(y, win=10, zeros=3):
    ## apply smoothning for windows of length <win>
    seg_ids = np.arange(0,y.shape[0]-win+1,1)
    label_segs = y[[np.arange(x,x+win) for x in seg_ids]]
    # run over all windows
    for s, seg in enumerate(label_segs):
        lab, cn = np.unique(seg,return_counts=True)
        # make all 0's if window contains more than <zeros> 0's.
        if lab[0]==0 and cn[0]>=zeros:
            label_segs[s] = 0
        # otherwise take the most occuring value
        else:
            label_segs[s] = lab[1+np.argmax(cn[1:], axis=0)]

    # save the smooth labels
    ys = label_segs[:,0]
    y_smooth = np.array(ys)
    # determine shift
    shift = zeros-1
    half_win = win//2

    #binarize for shift correction
    ys[ys>0] = 1
    y_diff = np.concatenate([[0],np.diff(ys)])
    beg = np.argwhere(y_diff>0)[:,0]
    end = np.argwhere(y_diff<0)[:,0]

    # verify label size matches
    if len(beg)!=0:
        if len(beg) == 0 or len(end) == 0:
            return y_smooth
        if beg[0] > end[0]:
            beg = np.concatenate([[0],beg])
        if beg[-1] > end[-1]:
            end = np.concatenate([end,[len(y_diff)-shift-half_win]])
        
    # Make sure only one label is assigned and shift all events by <zeros-1>, back to orignal
    for x in range(len(beg)):
        # determine segment start & end
        s = beg[x]
        e = end[x]
        # determine including labels
        lab, cn = np.unique(y_smooth[s:e],return_counts=True)
        try:
            # assign most occuring one to the whole segment + correction for the sliding window
            ce = e+half_win # correction
            y_smooth[s:ce] = lab[np.argmax(cn,axis=0)]
            # and apply shift
            y_smooth[s+half_win:ce+half_win] = y_smooth[s:ce]
            y_smooth[s:s+half_win] = 0
        except:
            try:
                print('Tried to fill the event till the end')
                # assign most occuring one to the whole segment + correction for the sliding window
                ce = -1 # correction
                y_smooth[s:ce] = lab[np.argmax(cn,axis=0)]
                # and apply shift
                y_smooth[s+half_win:ce] = int(y_smooth[s])
                y_smooth[s:s+half_win] = 0
            except:
                print('But it didn\'t work, somthing went wrong with beginning (%s) and end (%s)'%(s,e))
    
    # correct array for the edges 
    y_smooth = np.concatenate([y_smooth, y[:win-1]])
    return(y_smooth)


# Plotting functions:

def do_plot(data, hdr, algo_version, event_tag=''):
    Fs = hdr['newFs']
    cpap_start_ind = hdr['cpap_start']
    patient_tag = hdr['patient_tag'].split('~')[0]
    test_type = hdr['test_type']
    rec_type = hdr['rec_type']
    fontdict = {'fontsize':8}

    ##########################################################################################################
    # SUBPLOT 1: Respiration trace with events
    fig = plt.figure(figsize=(9.5,6))
    ax1 = fig.add_subplot(311)
    if algo_version == 'full':
        plt.suptitle('%s - %s - %s'%(rec_type, test_type, patient_tag), fontsize=10)
    elif algo_version == 'events':
        tag = event_tag.replace('neas', 'nea').replace('As', 'A')
        plt.suptitle('%s - %s - %s'%(rec_type, patient_tag, tag), fontsize=10)
    
    # plot signal 
    ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep==0), 'y')
    ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep==1), 'r')
    ax1.plot(data['Ventilation_baseline'].mask(np.isnan(data.Ventilation_combined)),'k')

    if algo_version == 'full':  
        # plot split line for PTAF <--> CPAP
        if hdr['cpap_start'] != None:
            index = pd.date_range(hdr['start_time'], periods=data.shape[0], freq=str(1/Fs)+'S')
            loc = np.where(index==cpap_start_ind)[0][0]
            ax1.plot([loc, loc], [np.min(data.Ventilation_combined), np.max(data.Ventilation_combined)], \
                                                                        c='r',linestyle='dashed', zorder=10)

    # plot apnea labels by the experts
    ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=1)), 'b')
    ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=2)), 'g')
    ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=3)), 'c')
    ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=4)), 'm')
    ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=5)), 'b')
    ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea!=6)), 'r')


    # plot events found by algorithm
    ax1.plot(-data['algo_apneas'].mask(data['algo_apneas'] != 1), 'b')
    ax1.plot(-data['algo_apneas'].mask(data['algo_apneas'] != 2), 'g')
    ax1.plot(-data['algo_apneas'].mask(data['algo_apneas'] != 3), 'c') ########
    ax1.plot(-data['algo_apneas'].mask(data['algo_apneas'] != 4), 'm')
    ax1.plot(-data['algo_apneas'].mask(data['algo_apneas'] != 5), 'm')
    # ax1.plot(-data['algo_hypopneas_four'].mask(data['algo_hypopneas_four'] != 4)-0.25, 'k')

    ax1.plot(-data['algo_apneas'].mask(data['algo_apneas'] != 6), 'r')

    # ax1.plot(-data['too_long_events'].mask(data['too_long_events']==0)*1.5, 'k') ###########

    # events = find_events(data['algo_apneas'])
    # for i, e in enumerate(events):
    #         plt.text(e[0], -5, str(i+1), fontsize=6)


    # plot additional 3% desaturation rule EEG hypopneas
    # if hyp_rule == 3:
    #     ax1.plot(-4.5*data['accepted_saturation_hypopneas'].mask(data['accepted_saturation_hypopneas'] != 1), 'b')
    #     ax1.plot(-4.5*data['accepted_EEG_hypopneas'].mask(data['accepted_EEG_hypopneas'] != 1), 'c')

    # # plot REJECTED hypopnea ventilation drops
    # ax1.plot(-5*data['rejected_saturation_hypopneas'].mask(data['rejected_saturation_hypopneas'] != 1), 'k')
    # if hyp_rule == 3:
    #     ax1.plot(-5*data['rejected_EEG_hypopneas'].mask(data['rejected_EEG_hypopneas'] != 1), 'k')

    # plot excursion lines
    # ax1.plot(data.Ventilation_drop_apnea, 'r')
    # ax1.plot(data.Ventilation_drop_hypopnea, 'g')

    # subplot layout
    ax1.set_title('Respiration trace', fontdict=fontdict, pad=-1)
    ax1.set_ylim([-7.1,8])
    # ax1.xaxis.set_visible(False)


    ##########################################################################################################
    # SUBPLOT 2: Respiratory effort belts
    abd = data.ABD.rolling(int(0.5*Fs), center=True).median().mask(data.patient_asleep==0)
    abd_s = data.ABD.rolling(int(0.5*Fs), center=True).median().mask(data.patient_asleep==1)
    chest = data.CHEST.rolling(int(0.5*Fs), center=True).median().mask(data.patient_asleep==0)
    chest_s = data.CHEST.rolling(int(0.5*Fs), center=True).median().mask(data.patient_asleep==1)

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(abd, 'b')
    ax2.plot(abd_s, 'r')
    ax2.plot(chest, 'g')
    ax2.plot(chest_s, 'r')

    # subplot layout
    ax2.set_title('Effort Belts', fontdict=fontdict ,pad=-1)
    ax2.set_ylim([-10,10])
    # ax2.xaxis.set_visible(False)


    ##########################################################################################################
    # SUBPLOT 3: Saturation and EEG arousals
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(data['SpO2'].mask(data.patient_asleep==0),'b')
    ax3.plot(data['SpO2'].mask(data.patient_asleep==1),'r')
    ax3.plot(data['saturation_drop'].mask(data['saturation_drop'] == 0)+99, 'm')
    # ax2.plot(data['saturation_return'].mask(data['saturation_return'] == 0)+98, 'r')
    ax3.plot(data.EEG_arousals.mask(data.EEG_arousals==0)+100, 'g')

        # ax3.plot(data['potential_RERAs'].mask(data.potential_RERAs==0)+101,'k')
    
    # subplot layout
    ax3.set_title('Saturation (m) and EEG arousals (g)', fontdict=fontdict, pad=-1)
    ax3.set_ylim([90,103])
    ax3.set_ylabel('%')

    ##########################################################################################################
    # show location of event in all plots
    if algo_version == 'events':
        loc = int(len(data) / 3 * 2)
        for ax in [ax1, ax2, ax3]:
            ax.plot([loc, loc], [-200, 200], c='r',linestyle='dashed', zorder=10)

    ##########################################################################################################
    # save_folder = '/media/cdac/hdd/Auto_respiration_Nassi/default/figures/%s'%patient_tag
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # plt.savefig(save_folder + '/results.pdf')

def plot_kernels(flat_kernels, non_flat_kernels):
    # plot some flat and non-flat kerels 
    fontdict = {'fontsize':8}
    kernel_options = [flat_kernels, non_flat_kernels]
    for i, kernels in enumerate(kernel_options):
        name = '"Flat kernels"' if i == 0 else '"Non-flat kernels"'
        fig = plt.figure(figsize=(9.5,6))
        plt.suptitle(name)
        for n in range(len(kernels)):
            ax = fig.add_subplot(6,5,n+1)
            ax.plot(kernels[n])
            # ax.set_title('kernel %s'%(n+1), fontdict=fontdict, pad=-1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])


# Performance Analysis functions:

def compute_full_performance(data, hdr, output_file='', base_path='', save=False):
    # set all 4% hypopneas to #4
    data.loc[np.where(data['algo_apneas']==5)[0], 'algo_apneas'] = 4

    # compute and show events per hour
    perf_dic = determine_events_per_hour(data, hdr, show=False)

    if save:
        # save event info
        save_full_performance(perf_dic, [], hdr, output_file, base_path)

    #####

    # compute and show binary confusion matrix
    ytrue = data.Apnea.fillna(0)
    ytrue[ytrue>0] = 1
    ypred = data.algo_apneas.fillna(0)
    ypred[ypred>0] = 1
    ss = data.patient_asleep.fillna(0)
    b_cmt = custom_confusion_matrix(ytrue, ypred, ss, binary=True)
    # print()
    # print(b_cmt)

    # compute and show multiclass confusion matrix
    ytrue = data.Apnea.fillna(0)
    ytrue[ytrue==6] = 5
    ypred = data.algo_apneas.fillna(0)
    ypred[ypred==6] = 5
    ss = data.patient_asleep.fillna(0)
    m_cmt = custom_confusion_matrix(ytrue, ypred, ss, binary=False)
    # print()
    # print(m_cmt)

    return b_cmt, m_cmt, perf_dic
  
def compute_event_performance(data, hdr, tag):
    Fs = hdr['newFs']

    # define event location (red dotted line )
    loc = int(len(data) / 3 * 2)
    region = list(range(loc-5*Fs, loc+5*Fs))

    # define most occuring label > 0
    model_output = data.loc[region, tag].values
    labels, counts = np.unique(model_output,return_counts=True)
    if len(labels) == 1 and labels[0] == 0:
        output = 0
    else:
        if labels[0] == 0:
            labels = labels[1:]
            counts = counts[1:]
        output = labels[np.argmax(counts)]

    return output

def cmt_tech_events(custom_cmt, wakes, ytrue, ypred):
    # segment all events in tech array
    exp_events = find_events(ytrue)
    
    for event in exp_events:
        # determine segment start & end
        s = event[0]
        e = event[1]
        # dont use segment if it includes > 50% wake time
        if np.sum(np.in1d(wakes, range(s,e+1))) > np.floor(0.5*len(range(s,e+1))): 
            continue
        # find according label in both arrays
        tL, tcn = np.unique(ytrue[s:e],return_counts=True)
        pL, pcn = np.unique(ypred[s:e],return_counts=True)
        try:
            tV = tL[np.argmax(tcn,axis=0)]
            # pV = pL[np.argmax(pcn,axis=0)]
            if len(pL)==1:
                pV = pL[0]
            else:
                if np.any(pL == 0):
                    pV = pL[np.argmax(pcn[1:],axis=0)+1]
                else:
                    pV = pL[np.argmax(pcn,axis=0)]
            # count true positives and false negatives (when no label found only pV == 0)
            if tV == pV or pV == 0:
                custom_cmt[tV,pV] += 1
            else:
                continue
        except:
            print('true, oh ooh\n %s \n %s'%(ytrue[s:e],ypred[s:e]))
            continue

    return custom_cmt   

def custom_confusion_matrix(ytrue, ypred, ss, binary=False, apnea_hyp = []):
    # make sure arrays are integers
    ytrue = ytrue.astype(int)
    ypred = ypred.astype(int)

    # define confusion matrix dimensions
    n_cl = 2 if binary else 6
    if apnea_hyp: n_cl = 4  ################
    custom_cmt = np.zeros([n_cl,n_cl],dtype=int)

    # determine wake indices
    wakes = np.where(ss==0)[0]

    # determine true negatives, total time of no events - wake time
    median_event_size = 180 
    custom_cmt[0,0] = int((sum((ypred+ytrue)==0)-len(wakes)) / median_event_size) 

    # run over all tech event segments
    custom_cmt = cmt_tech_events(custom_cmt, wakes, ytrue, ypred)
   
    # segment all events in prediction array
    algo_events = find_events(ypred)

    # run over all pred event segments
    for event in algo_events:
        # determine segment start & end
        s = event[0]
        e = event[1]
        # dont use segment if it includes > 50% wake time
        if np.sum(np.in1d(wakes, range(s,e+1))) > np.floor(0.5*len(range(s,e+1))): 
            continue
        # find according label in both arrays
        tL, tcn = np.unique(ytrue[s:e],return_counts=True)
        pL, pcn = np.unique(ypred[s:e],return_counts=True)
        try:
            # tV = tL[np.argmax(tcn,axis=0)]
            if len(tL)==1:
                tV = tL[0]
            else:
                if np.any(tL == 0):
                    tV = tL[np.argmax(tcn[1:],axis=0)+1]
                else:
                    tV = tL[np.argmax(tcn,axis=0)]
            pV = pL[np.argmax(pcn,axis=0)]
            
            # count all false positives, skip all true positives
            if tV != pV:
                custom_cmt[tV,pV] += 1
        except:
            print('pred, oh ooh\n %s \n %s'%(ytrue[s:e],ypred[s:e]))
            continue

    return(custom_cmt)
  
def determine_events_per_hour(data, hdr, show=True):
    # set info
    algo_apneas = data.algo_apneas.mask(data.patient_asleep == 0).fillna(0)
    exp_apneas = data.Apnea.mask(data.patient_asleep==0).fillna(0)
    rec_time = data.shape[0]/(3600*hdr['newFs'])

    # run over all events
    perf_dic = {}
    AHI_a = 0
    AHI_e = 0
    for i in range(1, len(event_tags)): 
        if i == 5:
            continue
        tag = event_tags[i]
        
        # determine number of event
        n_algo = len(find_events(algo_apneas.mask(data.algo_apneas != i).fillna(0)))
        n_exp = len(find_events(exp_apneas.mask(data.Apnea != i).fillna(0)))

        # save number of events in dictionary
        perf_dic = dict_event_per_hour(perf_dic, n_algo, n_exp, tag, rec_time)

        # print number of event
        if show:
            if n_exp == 0 and n_algo == 0:
                print("no %s present."%(tag))
            else:
                print("%s: algo: %s experts: %s." %(tag, n_algo, n_exp))

        # compute AHI and RDI
        if i in [1,2,3,4]:
            AHI_a += n_algo
            AHI_e += n_exp
        elif i == 6:            
            RDI_a = AHI_a + n_algo
            RDI_e = AHI_e + n_exp

    # save and show AHI and RDI values 
    perf_dic = show_AHI_RDI(data, hdr, perf_dic, AHI_a, AHI_e, RDI_a, RDI_e, show)

    return perf_dic

def show_AHI_RDI(data, hdr, dic, AHI_a, AHI_e, RDI_a, RDI_e, show):
    # set info
    rec_time = data.shape[0]/(3600*hdr['newFs'])

    tags = ['AHI', 'RDI']
    values = [[AHI_a, AHI_e], [RDI_a, RDI_e]]
    for i in range(len(tags)):
        # save number of events in dictionary
        dic = dict_event_per_hour(dic, values[i][0], values[i][1], tags[i], rec_time)

        # print number of event
        a_val = round(values[i][0]/rec_time, 1)
        e_val = round(values[i][1]/rec_time, 1)
        if show:
            print("%s (algo / exp): %s / %s "%(tags[i], a_val, e_val))

    return dic

def dict_event_per_hour(dic, n_algo, n_exp, tag, rec_time):
    # save events found by algo
    name = 'algo_' + tag
    dic[name] = round(n_algo/rec_time, 1)

    # save events found by experts
    name = 'exp_' + tag
    dic[name] = round(n_exp/rec_time, 1)

    return dic

def add_all_events(data, hdr):
    event_dic = {}

    # find all events
    no_sleep = data['patient_asleep']==0
    exp_events = find_events(data['Apnea'].mask(no_sleep).fillna(0))
    algo_events = find_events(data['algo_apneas'].mask(no_sleep).fillna(0))
    # and add to dictionary
    for e, events in enumerate([exp_events, algo_events]):
        tag = 'exp_event' if e == 0 else 'algo_event' 
        event_dic[tag+'_starts'] = np.array([ev[0] for ev in events])
        event_dic[tag+'_ends'] = np.array([ev[1] for ev in events])

    # determine according labels
    exp_labels = data.loc[[ev[0] for ev in exp_events], 'Apnea'].values.astype(int)
    algo_labels = data.loc[[ev[0] for ev in algo_events], 'algo_apneas'].values.astype(int)
    # and add to dictionary
    event_dic['exp_labels'] = exp_labels
    event_dic['algo_labels'] = algo_labels
    
    return event_dic

def save_full_performance(info_dic, all_data, hdr, output_file, base_path):
    dic = {}

    # store patient info
    info_hdr = ['patient_tag', 'test_type', 'rec_type']
    for i in info_hdr:
        dic[i] = hdr[i]
    # store AHI and RDI
    for d in info_dic.keys():
        dic[d] = info_dic[d]

    # retrieve already stored data
    d = append_new_and_old_data(output_file, dic)

    # save the performance
    d.to_csv(output_file, header=d.columns, index=None, sep='\t', mode='w+')

    if all_data:
        # setup hf5 file to save the events
        output_hf = base_path + 'performance/events/' + hdr['patient_tag']
        if not os.path.exists(output_hf): 
            os.makedirs(output_hf)
        version = output_file.split('/')[-1].split('.')[0]
        output_hf = output_hf + '/%s'%version

        # save algo apneas to .hf5 file
        d = pd.DataFrame([])
        no_sleep = all_data['patient_asleep']==0
        d['Apnea'] = all_data.Apnea.mask(no_sleep).fillna(0).values
        d['algo_apneas'] = all_data.algo_apneas.mask(no_sleep).fillna(0).values
        d['hyp_sat_three'] = all_data.accepted_saturation_hypopneas.mask(no_sleep).fillna(0).values
        d['hyp_eeg_three'] = all_data.accepted_EEG_hypopneas.mask(no_sleep).fillna(0).values
        d['hyp_three'] = all_data.algo_hypopneas_three.mask(no_sleep).fillna(0).values
        d['hyp_four'] = all_data.algo_hypopneas_four.mask(no_sleep).fillna(0).values
        d['PhysioNet'] = all_data.PhysioNet_labels.mask(no_sleep).fillna(0).values
        write_to_hdf5_file(d, output_hf, default_dtype='int', overwrite=True)

def save_event_performance(event_dic, output_file, base_path, tags=[]):
    # put option in dictionary
    dic = {}
    if not tags:
        dic['model_output'] = event_dic[0]
        dic['patient_tag'] = event_dic[1]
        dic['event_tag'] = event_dic[2]
        dic['time_stamp'] = event_dic[3]
    else:
        dic['patient_tag'] = tags[0]
        dic['event_tag'] = tags[1]
        dic['output'] = event_dic
    
    # append to already stored data
    data = append_new_and_old_data(output_file, dic)

    # save the performance
    data.to_csv(output_file, header=data.columns, index=None, sep='\t', mode='w+')

def append_new_and_old_data(output_file, dic):
    # put it all in a dataframe
    if os.path.exists(output_file):
        data = pd.read_csv(output_file, sep='\t')
        add_data = pd.DataFrame(dic, index=[len(data)])
        data = data.append(add_data)
    else:
        data = pd.DataFrame(dic, index=[0])

    return data

def get_original_event_labels(all_paths, txt_file, research_csv, signals, base_path, output_folder):
    print('* compute original labels *')
    output_file = output_folder + 'original_labels.csv'

    # run over all files
    for n, file_path in enumerate(all_paths):  
        # load in patient data
        data, hdr = load_sleep_data(file_path, load_all_signals=0, signals_to_load=signals)

        # find according patient info in txt file
        hdr = get_patient_info(txt_file, file_path, n, len(all_paths), hdr)
        
        # define event location
        event_df = research_event_to_df(research_csv, hdr['patient_tag'])
        for (_, event) in event_df.iterrows():
            e_seg, _ = define_tool_event_region(event, data, hdr)
            segment = e_seg.reset_index(drop=True) 
            # determine original label
            output = compute_event_performance(segment, hdr, 'Apnea')

            # save label
            tags = [hdr['patient_tag'], event['type'] + ' bin' + event['bin']]
            save_event_performance(output, output_file, base_path, tags=tags)


# Save output functions:

def save_full_output(data, hdr, out_path):
    if np.sum(data.patient_asleep.isna()) + np.sum(data.patient_asleep < 1) > 0.9*len(data):
        data['algo_apneas'] = 0
    else:
        # drop some columns 
        cols =  [   'rejected_saturation_hypopneas', 'soft_ventilation_drop_apnea', 'too_long_events', 'baseline2',
                    'Ventilation_neg_envelope', 'Ventilation_default_baseline', 'saturation_return' , \
                    'Ventilation_drop_apnea', 'Ventilation_drop_hypopnea', 'algo_hypopneas_four', \
                    'algo_hypopneas_three' ]
        data.drop(columns=cols, inplace=True)
        
    # save data
    write_to_hdf5_file(data, out_path, hdr=hdr, overwrite=True)


# Hyper-parameter functions:

def save_hyp_tuning_variables(data, output_file, nr, apnea_hyp=[]):

    if apnea_hyp:
        # compute confusion matrix
        ytrue = data.Apnea.fillna(0)
        ytrue[ytrue==6] = 0
        ytrue[ytrue==5] = 0
        ytrue[ytrue==4] = 0
        ypred = data.algo_apneas.fillna(0)
        ypred[ypred==6] = 0
        ypred[ypred==5] = 0
        ypred[ypred==4] = 0
        ss = data.patient_asleep.fillna(0)

        # compute normalized confusion matrix
        custom_cmt = custom_confusion_matrix(ytrue, ypred, ss, binary=False, apnea_hyp=apnea_hyp)
        nmz_cmt = custom_cmt.astype('float') / custom_cmt.sum(axis=1)[:, np.newaxis]
        nmz_cmt = nmz_cmt.round(decimals=2)

        # compute mean of confusion matrix diagional
        performance = []
        for i in range(len(nmz_cmt)):
            performance.append(nmz_cmt[i,i])
        performance = np.nanmean(performance)

        # initialize cmt value lists for each event
        tnr, pnr, tns, tps, fps, fns, sens, spec = [], [], [], [], [], [], [], []
        # run over all events 
        for e in range(1,custom_cmt.shape[0]):
            # compute tp, fp, tn, fn
            tnr.append(np.nansum(custom_cmt[e,:]))
            pnr.append(np.nansum(custom_cmt[:,e]))
            tps.append(custom_cmt[e,e])
            tns.append(np.sum(np.concatenate([custom_cmt[:e,0], custom_cmt[e+1:,0]]),axis=0))
            fns.append(np.sum(np.concatenate([custom_cmt[e,:e], custom_cmt[e,e+1:]]),axis=0))
            fps.append(np.sum(np.concatenate([custom_cmt[:e,e], custom_cmt[e+1:,e]]),axis=0))
            # compute sensitivity
            try:
                sens.append(int(np.round(tps[-1] / (tps[-1]+fns[-1]),decimals=2)*100))
            except:
                sens.append(np.nan)
            # compute specificity
            try:
                spec.append(int(np.round(tns[-1] / (fps[-1]+tns[-1]),decimals=2)*100))
            except:
                spec.append(np.nan)
        
        # store performance in df
        df = {}
        types = ['obstructive', 'central', 'mixed']
        df['performance'] = performance
        for i, t in enumerate(types):
            df[t+'_sens'] = sens[i]
            df[t+'_spec'] = spec[i]
            df['true_nr_'+t] = tnr[i]
            df['pred_nr_'+t] = pnr[i]


    else:
        # compute confusion matrix
        ytrue = data.Apnea.fillna(0)
        ytrue[ytrue==6] = 5
        ypred = data.algo_apneas.fillna(0)
        ypred[ypred==5] = 4
        ypred[ypred==6] = 5
        ss = data.patient_asleep.fillna(0)

        # compute normalized confusion matrix
        custom_cmt = custom_confusion_matrix(ytrue, ypred, ss, binary=False)
        nmz_cmt = custom_cmt.astype('float') / custom_cmt.sum(axis=1)[:, np.newaxis]
        nmz_cmt = nmz_cmt.round(decimals=2)
        # print(nmz_cmt)

        # compute mean of confusion matrix diagional
        performance1 = []
        for i in range(len(nmz_cmt)):
            performance1.append(nmz_cmt[i,i])
        performance1 = np.nanmean(performance1)

        #############
        # compute confusion matrix
        ytrue = data.Apnea.fillna(0)
        ytrue[ytrue>0] = 1
        ypred = data.algo_apneas.fillna(0)
        ypred[ypred>0] = 1
        ss = data.patient_asleep.fillna(0)

        # compute normalized binary confusion matrix
        binary_cmt = custom_confusion_matrix(ytrue, ypred, ss, binary=True)
        nmz_bin_cmt = binary_cmt.astype('float') / binary_cmt.sum(axis=1)[:, np.newaxis]
        nmz_bin_cmt = nmz_bin_cmt.round(decimals=2)
        # print(nmz_bin_cmt)

        # compute mean of confusion matrix diagional
        performance2 = []
        for i in range(len(nmz_bin_cmt)):
            performance2.append(nmz_bin_cmt[i,i])
        performance2 = np.nanmean(performance2)
        #############


        # put performance metrics in df
        performance = np.nanmean([performance1, performance2])
        binary_sensitivity = nmz_bin_cmt[1,1]*100
        binary_specificity = nmz_bin_cmt[0,0]*100

        # initialize cmt value lists for each event
        tnr, pnr, tns, tps, fps, fns, sens, spec = [], [], [], [], [], [], [], []
        # run over all events 
        for e in range(1,custom_cmt.shape[0]):
            # compute tp, fp, tn, fn
            tnr.append(np.nansum(custom_cmt[e,:]))
            pnr.append(np.nansum(custom_cmt[:,e]))
            tps.append(custom_cmt[e,e])
            tns.append(np.sum(np.concatenate([custom_cmt[:e,0], custom_cmt[e+1:,0]]),axis=0))
            fns.append(np.sum(np.concatenate([custom_cmt[e,:e], custom_cmt[e,e+1:]]),axis=0))
            fps.append(np.sum(np.concatenate([custom_cmt[:e,e], custom_cmt[e+1:,e]]),axis=0))
            # compute sensitivity
            try:
                sens.append(int(np.round(tps[-1] / (tps[-1]+fns[-1]),decimals=2)*100))
            except:
                sens.append(np.nan)
            # compute specificity
            try:
                spec.append(int(np.round(tns[-1] / (fps[-1]+tns[-1]),decimals=2)*100))
            except:
                spec.append(np.nan)
        
        # store performance in df
        df = {}
        types = ['obstructive', 'central', 'mixed', 'hypopnea', 'RERA']
        df['performance'] = performance
        df['perf_bin'] = performance2
        df['perf_multi'] = performance1
        df['binary sens'] = binary_sensitivity
        df['binary spec'] = binary_specificity
        for i, t in enumerate(types):
            df[t+'_sens'] = sens[i]
            df[t+'_spec'] = spec[i]
            df['true_nr_'+t] = tnr[i]
            df['pred_nr_'+t] = pnr[i]


    df = pd.DataFrame.from_dict(df, orient='index').T
    if os.path.exists(output_file):
        saved_df = pd.read_csv(output_file, sep='\t')
        df = saved_df.append(df)

    # save performance in .csv file
    df.to_csv(output_file, header=df.columns, index=None, sep='\t', mode='w+')

    return df, custom_cmt


