import os, glob, datetime, random, h5py, hdf5storage
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from matplotlib.dates import DateFormatter

from .mgh_sleeplab import *


### CLASS OBJECT AND METHODS ###

class Preprocessing_file:

	def __init__(self):
		self.nr = -1
		self.signal_file = ''
		self.label_file = ''
		self.annotation_file = ''
		self.tag = ''
		self.patient_ID = ''
		self.patient_tag = ''
		self.Anno = None
		self.rec_type = ''
		self.test_type = ''
		self.visit_date = ''
		self.preproccess_version = 'default'
		self.data = []
		self.hdr = {}
		self.include_in_research = ''

		self.stored_channels = np.array([]).astype('<U32')
		self.stored_signals = np.array([])

	def set_version(self, file_info, nr, version=''):
		# define preprocess version
		if version != '':
			self.preproccess_version = version

		# set file specifics
		# self.include_in_research = file_info['research_inclusion'][nr]
		self.nr = nr
		self.signal_file = file_info['signal_file'][nr]
		self.label_file = file_info['label_file'][nr]
		self.tag = file_info['tag'][nr]
		self.patient_ID = file_info['ID'][nr]
		self.visit_date = file_info['visit_date'][nr].replace('/','_')
		self.patient_tag = self.tag + '~' + self.visit_date
		if file_info['annotation_file'][nr] != 'None':
			self.annotation_file = pd.read_csv(file_info['annotation_file'][nr])
			self.Anno = True
		self.rec_type = 'Grass' if file_info['rec_type'][nr].lower() == 'grass' else 'Natus'
		self.test_type = file_info['test_type'][nr]
		if 'diagnostic' in self.test_type.lower():
			self.test_type = 'diagnostic'
		elif 'split' in self.test_type.lower():
			self.test_type = 'split_night'
		elif 'titration' in self.test_type.lower(): 
			self.test_type = 'full_cpap'
		else:
			raise Exception('Unknown recording test type (%s)'%self.test_type)
			
	def get_data(self, sleeplab_table=''):
		# open signal and label file
		if len(sleeplab_table) == 0:
			signals, channels, apnea, stage, Fs, start_time_anno, start_time_lab, cpap_start = \
				retrieve_signals_and_labels(self.signal_file, self.label_file, self.annotation_file,\
												self.nr, self.preproccess_version, self.rec_type, self.Anno,\
													self.stored_channels, self.stored_signals)

			# remove leading and trailing NaN values in sleep stage at all channels.		
			signals, apnea, stage = adjust_no_recorded_sleep(signals, apnea, stage)

			# check if data is valid
			start_time = valid_signal_and_label_check(self.nr, signals, apnea, Fs, start_time_anno,\
														start_time_lab)

		else:
			signals, channels, apnea, stage, arousal, Fs, cpap_start, start_time = \
				retrieve_signals_and_labels_Wolfgang(self.signal_file, self.annotation_file, sleeplab_table, self.nr)
			self.arousal = arousal

		# save for recurrent preprocessing data use
		# self.stored_signals = signals
		# self.stored_channels = channels


		# create dataframe with all data
		self.data, self.hdr = setup_dataframe(signals, Fs, channels, apnea, stage, start_time,\
												cpap_start, self.preproccess_version)

		self.data['EEG_events_anno'] = arousal

		# verify if signals for accordign test type is correct
		verify_signals_with_test_type(self.data, self.test_type, self.hdr, self.preproccess_version)

	def perform_preprocessing(self, tool_events=[]):
		# set respiration and eeg channel names
		if self.test_type == 'diagnostic':
			resp_channels = ['PTAF','ABD','CHEST']
		elif self.test_type == 'split_night':
			resp_channels = ['PTAF','CFlow','ABD','CHEST']
		elif self.test_type == 'full_cpap':
			resp_channels = ['CFlow','ABD','CHEST']
		eeg_channels = ['F3_M2', 'F4_M1', 'C3_M2', 'C4_M1', 'O1_M2', 'O2_M1']

		# remove all saturation values according to pulse quality or when below 50 
		self.data = correct_pulse_quality(self.data)

		# filter the respiration signals; and optionally the EEG channels
		self.data = apply_filtering(self.data, self.hdr, self.preproccess_version, resp_channels, eeg_channels)

		# resample all data
		self.data = resample_data(self.data, self.hdr, self.preproccess_version)

		# remove NaN values in every row +- 5sec
		self.data = remove_nans(self.data, add_rem=(5*self.hdr['newFs']))

		# clip normalize respiration signals
		for eb in range(len(resp_channels)):
			self.data[resp_channels[eb]] = clip_normalize(self.data[resp_channels[eb]], np.array(self.data['Stage']))

		# add annotations if availible
		# if self.preproccess_version == 'default' and self.Anno == True:
		# 	self.data = add_annotations(self.data, self.annotation_file, self.rec_type, self.Anno)
	
		# find tool segments and create according spectrograms for EEG channels
		if self.preproccess_version == 'Tool':
			self.tool_event_dic, self.spectrograms, self.spectrogram_full = find_tool_samples(self.data, \
												self.hdr, tool_events, self.patient_tag, self.test_type)

	def save_data(self, output_file):
		if self.preproccess_version == 'Tool':
			# run over all tool events and save in separate .mat files
			for i in self.tool_event_dic.keys():
				data = self.tool_event_dic[i]
				e_spec = self.spectrograms[i]
				file_name = output_file + '%s %s %s'%(self.patient_tag, e_spec['event_type'], e_spec['event_bin'])
				write_to_mat_file(data, file_name, self.preproccess_version, self.test_type, hdr=self.hdr, \
										spectrogram=e_spec, full_spectrogram=self.spectrogram_full)
		elif self.preproccess_version == 'PhysioNet':
			write_to_mat_file(self.data, output_file, self.preproccess_version, self.test_type, hdr=self.hdr)
		else:
			write_to_hdf5_file(self.data, output_file, hdr=self.hdr, overwrite=True)
			




### STATIC METHODS ###

# Data loading functions

def load_sleep_data(filepath, load_all_signals=1, idx_to_datetime=0, signals_to_load=[]):
	'''
	filepath: full filepath of sleep data (icu sleep or sleeplab .h5) file to be loaded
	load_all_signals = 1: boolean if all the data contained in file should be loaded.
	idx_to_datetime = 0: boolean if index should be transformed to datetime. expected: hdr in correct format, see 'write_to_hdf5_file' function.
	signals_to_load = []: non-default only needed if load_all_signals ==0, list of signals that should be loaded
	'''
	# try:
	# load file object
	ff = h5py.File(filepath, 'r')

	if load_all_signals:
		signals_to_load = list(ff.keys())
	# load the data:
	data = pd.DataFrame([])
	hdr = {}

	# header:
	header_fields = ['Fs', 'newFs', 'start_time', 'cpap_start', 'rec_type', 'test_type', 'patient_tag']
	header_fields = [x for x in header_fields if x in list(ff.keys())]
	# load all header variables
	for signal_to_load_tmp in header_fields:
		if signal_to_load_tmp in ['Fs', 'newFs']:
			hdr[signal_to_load_tmp] = ff[signal_to_load_tmp][:]
			if hdr[signal_to_load_tmp].shape[0]:
				hdr[signal_to_load_tmp] = hdr[signal_to_load_tmp][0]
		elif signal_to_load_tmp in ['cpap_start', 'start_time']:
			t = ff[signal_to_load_tmp][:]
			hdr[signal_to_load_tmp] = pd.to_datetime(f'{t[0]}-{t[1]}-{t[2]} {t[3]}:{t[4]}:{t[5]}.{t[6]}', infer_datetime_format = True) if len(t) > 1 else None  
		elif signal_to_load_tmp in ['rec_type', 'test_type', 'patient_tag']:
			t = ff[signal_to_load_tmp][:]
			hdr[signal_to_load_tmp] = str(t[0]).split('\'')[1]

	# load all signals
	for signal_to_load_tmp in signals_to_load:
		if signal_to_load_tmp in header_fields: continue       
		else:
			try:
				data[signal_to_load_tmp] = ff[signal_to_load_tmp][:]
				if signal_to_load_tmp == 'SpO2':
					data['SpO2'] = np.round(data['SpO2']).astype('int')

				if signal_to_load_tmp.lower() in ['stage', 'apnea']: 
						data.loc[(data[signal_to_load_tmp]==-1), signal_to_load_tmp] 
						data[signal_to_load_tmp] = data[signal_to_load_tmp].astype('int')
		
				if type(data[signal_to_load_tmp].iloc[0]) == np.float16:
					data[signal_to_load_tmp] = data[signal_to_load_tmp].astype('float32')
			except:
				continue

	ff.close()
	return data, hdr

def channel_selection(ff_s, version):
	# select channels according to file type
	eeg_channels = np.array(['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']) 	
	option_ch =	np.array([	'AIRFLOW','PTAF',\
							'CFLOW','CPAP',\
							'ABDOMEN','ABD','CHEST',\
							'SaO2','SpO2','PulseQuality',\
							'HR','PR','EKG',\
							'ECG-V1','Pleth',\
							'E1-M2','Chin1-Chin2'	])
		
	# define all signal tags
	header = ff_s['hdr']
	signal_labels = header['signal_labels'][:]
	channel_names = [ ''.join(list(map(chr, ff_s[signal_labels[i,0]][:].flatten()))) for i in range(signal_labels.shape[0]) ]
	channel_names = [channel.upper() for channel in channel_names]

	# add optional channels
	channels = np.array([])
	for opt in option_ch:
		opt = opt.upper()
		if opt in channel_names:
			channels = np.concatenate((channels, np.array([opt])))
	channels = np.concatenate((channels, eeg_channels))
	

	# find all signals of interest
	signal_channel_ids = []
	for i in range(len(channels)):
		found = False
		for j in range(len(channel_names)):
			if channel_names[j]==channels[i].upper():
				signal_channel_ids.append(j)
				found = True
				break
		if not found:
			if channels[i] in eeg_channels and version != 'default':
				raise Exception('Channel %s is not found.'%channels[i])		
	
	return signal_channel_ids, channels

def retrieve_signals_and_labels(signal_file, label_file, annotation_file, nr, version, rec_type, Anno, stored_channels, stored_signals):

	# open singal file
	with h5py.File(signal_file,'r') as ff_s:

		# find all signals in HF5 file
		channel_ids, channels = channel_selection(ff_s, version)
	
		# sort tags and retrieve signals
		tag_sorted = np.argsort(channel_ids)
		channels = channels[tag_sorted]

		# preallocate signal and fill with stored or from mad3 file
		signal = np.zeros((len(ff_s['s']), len(channels)))
		for i, ch in enumerate(channels):
			if ch in stored_channels:
				sig = np.where(stored_channels==ch)[0][0]
				signal[:,i] = stored_signals[:, sig]
			else:
				signal[:,i] = ff_s['s'][:, np.sort(channel_ids)[i]]
		

	# retrieve apnea and sleep stages
	apnea, stage = retrieve_apnea_and_stage_labels(label_file, nr)

	# retrieve Natus info
	if rec_type == 'Natus':
		Fs, start_time_anno, start_time_lab, cpap_start = retrieve_natus_info(signal_file, annotation_file, nr, rec_type, Anno)
	# retrieve Grass info
	else:
		Fs, start_time_anno, start_time_lab, cpap_start = retrieve_grass_info(label_file, annotation_file, nr, rec_type, Anno)

	return signal, channels, apnea, stage, Fs, start_time_anno, start_time_lab, cpap_start

def retrieve_apnea_and_stage_labels(label_file, nr):
	# retrieve apnea labels and sleep stages
	with h5py.File(label_file,'r') as ff_l:	
		apnea = ff_l['apnea'][:]
		stage = ff_l['stage'][:]
		apnea = apnea.astype(int)
			
		# verify if patiets had proper sleep
		stages = np.unique(stage[np.logical_not(np.isnan(stage))]).astype(int).tolist()
		if len(stages)<2:
			raise Exception('No sleep detected in patient: %s'%(nr+1))

	return apnea, stage

def retrieve_natus_info(signal_file, annotation_file, nr, rec_type, Anno):
	# open signal file
	with h5py.File(signal_file,'r') as ff_s:
		Fs = int(ff_s['recording']['samplingrate'][0][0])
		year = int(ff_s['recording']['year'][:][0][0])
		month = int(ff_s['recording']['month'][:][0][0])
		day = int(ff_s['recording']['day'][:][0][0])
		hour = int(ff_s['recording']['hour'][:][0][0])
		minute = int(ff_s['recording']['minute'][:][0][0])
		second = int(ff_s['recording']['second'][:][0][0])

		start_time_lab = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
	
	# retrieve cpap start for Natus
	start_time_anno, cpap_start = get_cpap_and_start_time(nr, annotation_file, rec_type, Anno, year=year, month=month, day=day)

	return Fs, start_time_anno, start_time_lab, cpap_start

def retrieve_grass_info(label_file, annotation_file, nr, rec_type, Anno):	
	# retrieve apnea labels and sleep stages
	with h5py.File(label_file,'r') as ff_l:	
		# retrieve start time for Grass from label file
		time_str_elements = ff_l['features']['StartTime'][()].flatten()
		st = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
		st = st.split(':')
		se = st[-1].split('.')
		s_hour = int(float(st[0]))
		start_time_lab = datetime.datetime(1990,1,1,s_hour, minute=int(float(st[1])),
				second=int(float(se[0])), microsecond=int(float('0.' + se[1])*1000000))

		# fixed sample freq for Grass data
		Fs = 200	
		# retrieve cpap start for Grass
		start_time_anno, cpap_start = get_cpap_and_start_time(nr, annotation_file, rec_type, Anno)

	return Fs, start_time_anno, start_time_lab, cpap_start


# Data saving functions

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

def write_to_mat_file(data, output_file, version, test_type, hdr=[], spectrogram=[], full_spectrogram=[], default_dtype='float32', overwrite=False):
	# input:  df w/ signals, path_to_save, header dict, spectrograms dict
	# output: *saved data in .mat file
	eeg_channels = ['F3_M2', 'F4_M1', 'C3_M2', 'C4_M1', 'O1_M2', 'O2_M1']
	tool_channels = ['Ventilation_combined', 'CHEST', 'ABD', 'SpO2', 'HR', 'ECG', 'Pleth', 'Stage']
	
	# create dictionary with all signals
	all_data_dict = {}
	
	if version == 'Tool':
		for s_tmp in data.columns:
			if s_tmp not in eeg_channels + tool_channels:
				continue
			all_data_dict[s_tmp] = np.array(data[s_tmp])

		# add spectrograms
		all_data_dict.update(spectrogram)
		all_data_dict.update(full_spectrogram)
		
	elif version == 'PhysioNet':
		# add ventilation combined
		data, _ = create_ventilation_combined(data, hdr, test_type)

		# create 12xdata.shape matrix
		s_list = eeg_channels + ['E1_M2', 'CHIN1_CHIN2', 'ABD', 'CHEST', 'Ventilation_combined', 'SpO2']
		all_data = np.zeros((len(s_list),data.shape[0]))
		for n, s_tmp in enumerate(s_list):
			all_data[n,:] = np.array(data[s_tmp])
		all_data_dict['val'] = all_data

		if not os.path.exists(output_file):
			os.makedirs(output_file)

		output_file = output_file + '/signals.mat'

		# overwrite file if required
		if overwrite:
			if os.path.exists(output_file):
				os.remove(output_file)

	else:
		raise Exception('Error due to wrong version saving in .mat file')
	
	# write all data to .mat file
	if not '.mat' == output_file[-4:]:
		output_file = output_file + '.mat'
	hdf5storage.savemat(output_file, all_data_dict)

def append_to_hdf5_file(data, output_h5_path, default_dtype='float32', run=0):
	# input:  df w/ signals, path_to_save, header dict
	# output: *saved data in .hdf5 file
	chunk_size = 64

	if run==0 and os.path.exists(output_h5_path):
		os.remove(output_h5_path)
	elif os.path.exists(output_h5_path):
		old_data, _ = load_sleep_data(output_h5_path)
		if not old_data.empty:
			data = old_data.join(data)
		os.remove(output_h5_path)

	with h5py.File(output_h5_path, 'a') as f:
		# save signals:
		for signal_tmp in data.columns:
			if not signal_tmp in f: # first write of this signal
				if signal_tmp.lower() in ['annotation', 'test_type', 'rec_type', 'patient_tag']: 
					dtype1 = h5py.string_dtype(encoding='utf-8') # h5py needs to be >= 2.10
					dset_signal = f.create_dataset(signal_tmp, shape=(data.shape[0],), maxshape=(None,),
											chunks=(chunk_size,), dtype=dtype1)
					dset_signal[:] = data[signal_tmp].astype('str')
					continue # with next signal

				elif signal_tmp.lower() in ['stage', 'apnea', 'Fs', 'newFs', 'cpap_start']: 
					dtype1 = 'int8'    # for sleep lab data
					data.loc[pd.isna(data[signal_tmp]), signal_tmp] = -1
				else: dtype1 = default_dtype

				dset_signal = f.create_dataset(signal_tmp, shape=(data.shape[0],), maxshape=(None,),
												chunks=(chunk_size,), dtype=dtype1)
				dset_signal[:] = data[signal_tmp].astype(dtype1)

			else:                
				raise ValueError('Signal already exists in file but currently not intended to overwrite')


# Data formatting functions

def get_cpap_and_start_time(nr, annotation_file, rec_type, Anno, year=1990, month=1, day=1):
	# input: patient_iteration, df w/ file paths, annotation_tag, (opt: recording year, month and day)
	# ouptut: recording start, cpap start

	# When no annotation file is present, return NaN's
	if not Anno:
		return None, -1
	
	# cpap should be mentioned in annotations:
	anno = annotation_file
	cpap_annotations = [x for x in anno["event"] if 'cmH2O' in x]
	
	# define start time
	if rec_type == "Grass": 
		start_annotations = [x for x in anno["event"] if 'Start Recording' in x]
	else:
		start_annotations = [x for x in anno["event"] if 'Video_Recording_ON' in x]
	if not start_annotations:
		start_annotations = anno["time"][0]
	else:
		start_annotations = anno["time"].iloc[np.in1d(anno["event"].values, start_annotations)].values[0]
	# convert into datetime
	st = start_annotations.split(":")
	se = st[-1].split('.')
	# specify if start time is after midnight
	hour=int(float(st[0]))
	if hour < 12:
		day += 1 
	microsecond = 0 if len(se) == 1 else int(float('0.' + se[1])*1000000)
	start_time = datetime.datetime(year,month,day, hour=hour, minute=int(float(st[1])),
					second=int(float(se[0])), microsecond=microsecond)
	
	# save cpap_tag > 0.0 pressure
	found = False
	for i, tag in enumerate(cpap_annotations):
		if not '0.0 ' in tag:
			found = True
			f = i
			break
	# return if no good cpap tag is found
	if not found:
		return start_time, -1

	# define cpap start time
	cpap_annotations = anno["time"].iloc[np.in1d(anno["event"].values, cpap_annotations)].values[f] ############JOW
	# convert into datetime
	st = cpap_annotations.split(":")
	se = st[-1].split('.') 
	# specify if cpap_start is after midnight
	hour=int(float(st[0]))
	if hour < 12:
		day += 1 
	if len(se) == 1:
		try:
			start_cpap = datetime.datetime(year,month,day,hour=hour, minute=int(float(st[1])),
							second=int(float(se[0])))
		except:
			start_cpap = datetime.datetime(year,month,day-1,hour=hour, minute=int(float(st[1])),
							second=int(float(se[0])))
	else:
		try:
			start_cpap = datetime.datetime(year,month,day,hour=hour, minute=int(float(st[1])),
							second=int(float(se[0])), microsecond=int(float('0.' + se[1])*1000000))
		except:
			start_cpap = datetime.datetime(year,month,day-1,hour=hour, minute=int(float(st[1])),
							second=int(float(se[0])), microsecond=int(float('0.' + se[1])*1000000))

	return start_time, start_cpap

def valid_signal_and_label_check(nr, signal, apnea, Fs, start_time_anno, start_time_lab):
	# check for faulty data!
	if signal.shape[0] != apnea.shape[0]:
		raise Exception('Inconsistant apnea legnth and signal length for patient: %s'%(nr+1))
	if Fs!=512 and Fs!=200 and Fs!=250:
		raise Exception('Spurious sampling freqency found for patient: %s'%(nr+1))
	if start_time_anno != None:
		start_time = start_time_anno
		if abs(start_time_anno.day - start_time_lab.day) == 1:
			start_time_anno = start_time_anno.replace(day=1)
			start_time_lab = start_time_lab.replace(day=1)
			start_time = start_time.replace(day=1)
			if abs(start_time_anno - start_time_lab) > datetime.timedelta(seconds=1):
				raise Exception('Start times error for patient: %s. %s and %s' %(nr+1, start_time_anno, start_time_lab))
		elif abs(start_time_anno - start_time_lab) > datetime.timedelta(seconds=5):
			raise Exception('Start times error for patient: %s. %s and %s' %(nr+1, start_time_anno, start_time_lab))
	else:
		start_time = start_time_lab
	
	return start_time

def setup_dataframe(signal, Fs, channels, apnea, stage, start_time, cpap_start, version):
	data = pd.DataFrame(signal, columns=channels)
	data = data.loc[:,~data.columns.duplicated()]

	# name all signal columns accordingly
	data = rename_dataframe_columns(data)

	# add labels and sleep staging
	data['Apnea'] = apnea
	data['Stage'] = stage
	data['patient_asleep'] = data.Stage < 5
	
	# add header values
	hdr = {}
	hdr['Fs'] = Fs
	if version == 'Tool':
		hdr['newFs'] = 200 
	elif version == 'PhysioNet':
		hdr['newFs'] = 50
	else:
		hdr['newFs'] = 10
	hdr['start_time'] = start_time
	hdr['cpap_start'] = cpap_start
	
	# create datetime index for dataframe
	data.index = pd.date_range(start_time, periods=data.shape[0], freq=str(1/Fs)+'S')

	return data, hdr

def rename_dataframe_columns(data):
	if not 'CFlow' in data.columns:
		if 'CFLOW' in data.columns:
			data = data.rename(columns={"CFLOW": "CFlow"})
	if not 'SpO2' in data.columns:
		if 'SAO2' in data.columns:
			data = data.rename(columns={"SAO2": "SpO2"})
		elif 'SPO2' in data.columns:
			data = data.rename(columns={"SPO2": "SpO2"})
	if not 'HR' in data.columns:
		if 'PR' in data.columns:
			data = data.rename(columns={"PR": "HR"})
	if not 'ABD' in data.columns:
		if 'ABDOMEN' in data.columns:
			data = data.rename(columns={"ABDOMEN": "ABD"})
	if not 'ECG' in data.columns:
		if 'EKG' in data.columns:
			data = data.rename(columns={"EKG": "ECG"})
		elif 'ECG-V1' in data.columns:
			data = data.rename(columns={"ECG-V1": "ECG"})
	if not 'Pleth' in data.columns:
		if 'PLETH' in data.columns:
			data = data.rename(columns={"PLETH": "Pleth"})
		else:
			data['Pleth'] = np.nan

	# remove '-' for error in hdf5storate.savemat
	for name in data.columns:
		if name in ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'CHIN1-CHIN2']:
			data = data.rename(columns={name: name.replace('-', '_')})

	return data

def verify_signals_with_test_type(data, test_type, hdr, version):
	cols = data.columns
	if test_type == 'diagnostic':
		if not 'PTAF' in cols:
			raise Exception('Diagnostic test type but no PTAF.')

	elif test_type == 'split_night':
		if (not 'PTAF' in cols) and (not 'CFlow' in cols):
			raise Exception('Split night test without PTAF or CPAP.')
		if (not 'PTAF' in cols) and (hdr['cpap_start'] != hdr['start_time']):
			raise Exception('Split night test without PTAT and cpap does not start at beginning.')
		if (not 'CFlow' in cols) and (hdr['cpap_start'] != -1):
			raise Exception('Split night test with cpap but no CFlow.')

	elif test_type == 'full_cpap':
		if not 'CFlow' in cols:
			raise Exception('Full cpap test type but no CFlow.')

	else:
		raise Exception('Unknown recording test type (%s)'%test_type)

def adjust_no_recorded_sleep(signal, apnea, stage):
	# remove leading and trailing NaN values in sleep stage at all channels.
	start_stage_id, end_stage_id = 0, len(stage)+1
	if np.isnan(stage[0][0]):
		start_stage_id = np.argmax(np.isfinite(stage))
		end_stage_id = -np.argmax(np.isfinite(np.flip(stage)))
		signal = signal[start_stage_id:end_stage_id,:]
		apnea = apnea[start_stage_id:end_stage_id]
		stage = stage[start_stage_id:end_stage_id]

	return signal, apnea, stage

def correct_pulse_quality(data):
	if 'PulseQuality' in data.columns:
		data.loc[data.PulseQuality < 20, 'SpO2'] = None
		data.loc[data.PulseQuality < 0.5*np.median(data.PulseQuality), 'SpO2'] = None 
		data.loc[data.SpO2 < 50, 'SpO2'] = None
		data.loc[data.HR < 20, 'HR'] = None
		data.loc[data.HR > 250, 'HR'] = None

	return data

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

def add_annotations(data, anno, rec_type, Anno):
	# input:  df w/a data, patient_iteration, recording tag
	# ouptut: df with added annotations
		
	# specify time format
	forms = [' %H:%M:%S.%f', '%H:%M:%S']
	for i, form in enumerate(forms):
		try:
			time1 = pd.to_datetime(anno.time, format=form)
		except:
			if i < len(forms)-1:
				continue
			else:
				raise Exception('Multiple annotation files were found!')
				return data
	
	# find midnight location in annotation file
	before_midnight = np.where(time1.apply(lambda x: x.hour>12))[0]
	assert([0] in before_midnight)
	
	# create new time array according to midnight location
	before_midnight = before_midnight.shape[0]
	after_midnight = time1.shape[0]-before_midnight
	before_midnight = [str(data.iloc[0].name.date())]*before_midnight
	after_midnight = [str(data.iloc[-1].name.date())]*after_midnight
	date_array = before_midnight + after_midnight

	# convert into datetime and round to .1 sec
	anno.time = pd.to_datetime([x[0] +' ' + x[1] for x in list(zip(date_array, anno.time))], format='%Y-%m-%d '+ form)
	anno.time = [x.round('100ms') for x in anno.time]    
	
	# remove all duplicates
	for iPotentialDuplicateRound in range(1,9):
		if sum(anno.time.duplicated()) == 0: break
		new_time = np.array([x.replace(microsecond=iPotentialDuplicateRound*100000) for x in anno.time[anno.time.duplicated()]])
		anno.loc[anno.time.duplicated(),'time'] = new_time[0] if new_time.shape==1 else new_time

	assert sum(anno.time.duplicated()) == 0, "Not all duplicates solved."

	# add EEG events from annotation file to data
	data = add_EEG_arousals(data, rec_type, Anno, anno=anno)

	# add to dataframe
	anno.set_index('time', inplace=True)

	# drop epch and duration columns
	anno.drop('epoch', axis=1,inplace=True)
	if 'duration' in anno.columns:
		anno.drop('duration', axis=1, inplace=True) 

	anno.columns = ['Annotation']
	data = data.join(anno)
		
	return data

def add_EEG_arousals(data, Grass, Anno, anno=False):
	if Anno:
		# specify EEG_events from annotation file
		eeg_durations = []
		try:
			eeg_arousals = anno.event.str.contains('Spontaneous')
			# determine all event durations
			for ed in anno.event[eeg_arousals]:
				duration = ed.split(':')[1].split('sec')[0].strip()
				try: 
					duration = str(duration).split('.')
					eeg_durations.append(datetime.timedelta(seconds=float(duration[0]),microseconds=float('0.'+duration[1])*1000000))
				except:
					duration = str(ed)
					eeg_durations.append(datetime.timedelta(seconds=float(duration[0])))

		except:
			eeg_arousals = anno.event.str.contains('EEG_arousal')
			# determine all event durations
			for ed in anno.duration[eeg_arousals]:
				try: 
					duration = str(ed).split('.')
					eeg_durations.append(datetime.timedelta(seconds=float(duration[0]),microseconds=float('0.'+duration[1])*1000000))
				except:
					duration = str(ed)
					eeg_durations.append(datetime.timedelta(seconds=float(duration[0])))
		
		eeg_starts = anno.time[eeg_arousals]  
		# determine ends of events
		eeg_ends = eeg_starts + eeg_durations 
		eeg_ends = [x.round('100ms') for x in eeg_ends]
		eeg_starts = [x.round('100ms') for x in eeg_starts]
		eeg_events = [x for x in list(zip(eeg_starts, eeg_ends))]
		# add all EEG events to dataframe
		data['EEG_events_anno'] = np.nan
		for x in eeg_events:
			data.loc[x[0]:x[1],'EEG_events_anno'] = 1 

	else:
		data['EEG_events_anno'] = np.nan

	return data

def create_ventilation_combined(data, hdr, test_type):
	# select channels according to recording test type
	# for diagnostic just take PTAF signal
	if test_type == 'diagnostic':
		data['Ventilation_combined'] = data['PTAF']	
		flowchannels = [data['Ventilation_combined']]
		cpap_index = 0
	# for full cpap recording start from cpap start
	elif test_type == 'full_cpap':
		if hdr['cpap_start'] == -1:
			cpap_index = 0
		else:
			cpap_index = time_stamp_datetime_to_index(hdr['cpap_start'], hdr)
		data['Ventilation_combined'] = np.nan
		data['Ventilation_combined'].values[cpap_index:] = data['CFlow'].values[cpap_index:]
		flowchannels = [data['Ventilation_combined']]
		data['Apnea'].values[:cpap_index+3*hdr['Fs']] = 0
	elif test_type == 'Robert':
		if 'Ventilation_combined' in data.columns: 
			flowchannels = [data['Ventilation_combined']]
		else:
			if hdr['cpap_start'] == -1:
				flowchannels = [data['Ventilation1']]
				cpap_index = data.shape[0]
			else:
				cpap_index = time_stamp_datetime_to_index(hdr['cpap_start'], hdr)
				flowchannels = [data['Ventilation1'].values[:cpap_index], data['Ventilation2'].values[cpap_index:]]
			
			# initialize new dataframe column to be filled each run
			data['Ventilation_combined'] = np.nan

			# run over the ventilation signal(s)
			for n, flowch in enumerate(flowchannels):
				# Add signal part to new dataframe column
				if n == 0:
					data['Ventilation_combined'].values[:cpap_index] = flowch
				else:
					data['Ventilation_combined'].values[cpap_index:] = flowch
	else:
		# for split night, split Nasal Pressure and CPAP signal if cpap tag is provided
		if hdr['cpap_start'] == -1:
			flowchannels = [data['PTAF']]
			cpap_index = data.shape[0]
		else:
			cpap_index = time_stamp_datetime_to_index(hdr['cpap_start'], hdr)
			flowchannels = [data['PTAF'].values[:cpap_index], data['CFlow'].values[cpap_index:]]

		# initialize new dataframe column to be filled each run
		data['Ventilation_combined'] = np.nan

		# run over the ventilation signal(s)
		for n, flowch in enumerate(flowchannels):
			# Add signal part to new dataframe column
			if n == 0:
				data['Ventilation_combined'].values[:cpap_index] = flowch
			else:
				data['Ventilation_combined'].values[cpap_index:] = flowch

	return data, flowchannels

def time_stamp_datetime_to_index(time_stamp, hdr):
	td = time_stamp - hdr['start_time']
	time_stamp_ind = int(td.seconds*hdr['newFs'] + (td.microseconds / 100000))

	return time_stamp_ind

def info_to_dic(add_list, cols):
    dic = {}
    for i, c in enumerate(cols):
        dic[c] = add_list[i]

    return dic


# Preprocessing functions

def clip_normalize(signal, sleep_stages):
	# input:  1D signal, 1D sleep stage array
	# ouptut: normalized_clipped_signal

	signal = np.array(signal)

	# determine all sleep indices
	if sleep_stages is not None:
		sleep_stages[np.isnan(sleep_stages)] = 6
		sleep = np.where(sleep_stages<5)[0]
	else:
		sleep = np.array(range(len(signal)))

	# clip signal based on sleep
	signal_clipped = np.clip(signal, np.nanpercentile(signal[sleep],1), np.nanpercentile(signal[sleep],99))
	signal = np.array((signal - np.nanmean(signal_clipped)) / np.nanstd(signal_clipped))

	return signal

def apply_filtering(data, hdr, version, resp_channels, eeg_channels):
	for ch in range(len(resp_channels)):
		data[resp_channels[ch]] = filter_signal(data[resp_channels[ch]], hdr['Fs']) 
	if version == 'Tool':
		for ch in range(len(eeg_channels)):
			data[eeg_channels[ch]] = filter_signal(data[eeg_channels[ch]], hdr['Fs']) 
	if 'ECG' in data.columns:
		data['ECG'] = filter_signal(data['ECG'], hdr['Fs']) 

	return data

def filter_signal(signal, Fs):
	# input:  1D signal, Sampling freq
	# ouptut: filtered signal
	from mne.filter import notch_filter, filter_data
	n_jobs = -1

	if signal.name in ['PTAF','CFlow','ABD','CHEST', 'Ventilation_combined', 'CPAP', 'THERM']:
		bandpass_frq = [None, 10]
		notch_freq = [50, 60]

		# apply preprocessing to effort belts
		signal = notch_filter(signal, Fs, notch_freq, n_jobs=n_jobs, verbose='ERROR')
		signal = filter_data(signal, Fs, bandpass_frq[0], bandpass_frq[1], n_jobs=n_jobs, verbose='ERROR')

	elif signal.name in ['F3_M2', 'F4_M1', 'C3_M2', 'C4_M1', 'O1_M2', 'O2_M1']:
		bandpass_frq = [0, 20]
		notch_freq =  [50]
		
		# apply preprocessing to effort belts
		signal = notch_filter(signal, Fs, notch_freq, n_jobs=n_jobs, verbose='ERROR')
		signal = filter_data(signal, Fs, bandpass_frq[0], bandpass_frq[1], n_jobs=n_jobs, verbose='ERROR')

	elif signal.name == 'ECG':
		# apply preprocessing to ECG signal
		bandpass_frq = [10, None]
		signal = filter_data(signal, Fs, bandpass_frq[0], bandpass_frq[1], n_jobs=n_jobs, verbose='ERROR')
	
	else:
		raise Exception('Filtering error!')

	return signal

def resample_data(data, hdr, version):
	# input:  df w/a data, header dict
	# ouptut: dataframe with resampled data
	#  
	# run over all signals in sequence
	Fs = hdr['Fs']
	labels = ['Stage','Apnea', 'EEG_events_anno']
	for column in data.columns:
		# check for NaN values in apnea labels
		if column in labels:
			if np.any(np.isnan(data[column])):
				locs = np.where(np.isnan(data[column]))[0]
				data.iloc[locs, np.where(column==data.columns)[0]] = -1
				# raise Exception('Found Nan in %s labels for patient!'%column)
		# fill NaN values upto 20 sec
		if column == ['SpO2','HR']:
			data[column] = data[column].interpolate(method='pchip', order=3, limit_area='inside', limit=Fs*20)
		# fill NaN values upto 5 sec
		else:
			data[column] = data[column].interpolate(method='pchip', order=3, limit_area='inside', limit=Fs*5)
	
	# resample all data to 10Hz, 50Hz for PhysioNet, or 200Hz for the Tool
	signals = list(set(data.columns) - set(labels))
	aggregation_dict = {}
	for signal in signals:
		aggregation_dict[signal] = 'mean'
	for label in labels:
		aggregation_dict[label] = 'max' # usually one would like to use 'nearest' or sth. but max works too, i.e. just do not interpolate mean in between labels, use either first or second element.
	
	if version == 'PhysioNet':
		data = data.resample('0.02S').agg(aggregation_dict)			# 50 Hz for Physionet model
	elif version == 'Tool':
		data = data.resample('0.005S').agg(aggregation_dict)		# 200 Hz for Tool examples
	elif version in ['default', 'Robert']:
		data = data.resample('0.1S').agg(aggregation_dict)		# 10 Hz for normal analysis
	for column in labels:
		locs = np.where(data[column] == -1)[0]
		data.iloc[locs, np.where(column==data.columns)[0]] = np.nan

	return data

def window_correction(array, window_size):
	half_window = int(window_size//2)
	events = find_events(array)
	corr_array = np.array(array)
	
	# run over all events in array
	for st, end in events:
		label = array[st]
		corr_array[st-half_window : end+half_window] = label

	return corr_array.astype(int) 

def create_spectrogram(data, hdr, region=[], plot=False):
	from mne.time_frequency import psd_array_multitaper
	Fs = hdr['newFs']
	
	if len(region) == 0:
		# segment full recording in 30s epochs
		full_eeg = data.values
		epoch_size = int(30*Fs)
		epoch_step = int(30*Fs)
		start_ids = np.arange(0, len(full_eeg)-epoch_size+1, epoch_step)
		seg_ids = list(map(lambda x:np.arange(x,x+epoch_size), start_ids))
		eeg_segs = full_eeg[np.array(seg_ids)]  
		eeg_segs = np.expand_dims(eeg_segs, axis=1) # eeg_segs.shape=(#epoch, #channel, Tepoch)
		# set spectrogram parameters
		NW = 10.
		BW = NW*2./30	

	else:
		# segment 30 sec window into 3sec epochs
		eeg = data.iloc[region].values
		epoch_size = int(3*Fs)
		epoch_step = Fs//10
		start_ids = np.arange(0, len(eeg)-epoch_size+1, epoch_step)
		seg_ids = list(map(lambda x:np.arange(x,x+epoch_size), start_ids))
		eeg_segs = eeg[np.array(seg_ids)]
		eeg_segs = np.expand_dims(eeg_segs, axis=1) # eeg_segs.shape=(#epoch, #channel, Tepoch)
		# set spectrogram parameters
		NW = 10.
		BW = NW*2./3

	# compute spectrogram
	specs, freqs = psd_array_multitaper(eeg_segs, Fs, fmin=0.1, fmax=20, adaptive=False,low_bias=True, verbose='ERROR', bandwidth=BW, normalization='full')
	specs = np.squeeze(10*np.log10(specs))
	time = np.arange(0, 30, 1/specs.shape[0]*30)

	if plot:
		spec_db_vmax = np.quantile(specs, .95)
		spec_db_vmin = np.quantile(specs, .05)

		plt.figure(figsize=(7.6,6))
		plt.imshow(specs.T, cmap='jet', origin='lower', aspect='auto', vmax=spec_db_vmax ,vmin=spec_db_vmin)
		plt.show()

	return specs, freqs, time


# Ananlysis functions

def find_events(signal, Fs=10):
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

	if len(ends) != len(starts):
		# print('Try label correction for channel ..')
		starts, ends = label_correction(starts, ends, signal, Fs)
		
	assert len(ends) == len(starts)

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
		else:
			print('ERROR!!')
			import pdb; pdb.set_trace()

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

def search_for_merged_labels(signal):		
	# find locations where events are merged
	locs = []
	diff_drops = pd.DataFrame(signal).diff().fillna(0)
	diff_drops.drop(np.where(diff_drops==0)[0], inplace=True)
	for i in range(len(diff_drops)-1):
		try:
			current_val = diff_drops.iloc[i].values[0]
			next_val = diff_drops.iloc[i+1].values[0]
		except:
			current_val = diff_drops.iloc[i]
			next_val = diff_drops.iloc[i+1]
		if current_val > 0 and next_val > 0:
			locs.append((diff_drops.index[i], diff_drops.index[i+1], diff_drops.index[i+2]))
		elif current_val < 0 and next_val < 0:
			locs.append((diff_drops.index[i-1], diff_drops.index[i], diff_drops.index[i+1]))
	return locs

def find_tool_samples(data, hdr, tool_events, patient_tag, test_type):
	# create combined central EEG channel
	eegs = ['C3_M2', 'C4_M1']
	data['EEG_Tool'] = data[eegs].mean(axis=1, skipna=True)
	# add ventilation combined
	data, _ = create_ventilation_combined(data, hdr, test_type)

	# find samples
	event_df = tool_events_to_df(tool_events, patient_tag)

	# run over all events
	tool_event_dic = {}
	spectrograms = []
	for (e, event) in event_df.iterrows():
		# define event regions
		spectrogram_dic = {}
		e_seg, spectrogram_region = define_tool_event_region(event, data, hdr)

		# compute spectrogram
		specs, freqs, time = create_spectrogram(data['EEG_Tool'], hdr, region=spectrogram_region)

		# add to dictionaries
		tool_event_dic[e] = e_seg.reset_index(drop=True)
		spectrogram_dic['event_spectrogram'] = specs
		spectrogram_dic['event_freqs'] = freqs
		spectrogram_dic['event_time'] = time[:specs.shape[0]]
		spectrogram_dic['event_type'] = event.type
		spectrogram_dic['event_bin'] = event.bin
		spectrograms.append(spectrogram_dic)

	# compute spectrogram of full recording, and add to dictionary
	specs, freqs, time = create_spectrogram(data['EEG_Tool'], hdr)
	spectrogram_full = {}
	spectrogram_full['full_spectrogram'] = specs
	spectrogram_full['full_freqs'] = freqs
	spectrogram_full['full_time'] = time[:specs.shape[0]]

	return tool_event_dic, spectrograms, spectrogram_full

def tool_events_to_df(tool_events, patient_tag):
	# create df with all event for this patient
	cols = ['type', 'bin', 'time']
	event_df = pd.DataFrame([], columns=cols)
	for bins in tool_events:
		# set event and info
		t, b = bins.split('_')
		b = b.split('bin')[-1]
		dic = tool_events[bins]
		events = dic['loc_datetime'][dic['patient_tag'] == patient_tag].values

		# append events to df
		for event in events:
			time_stamp = str_to_time_stamp(event)
			df_row = info_to_dic([t, b, time_stamp], cols)
			event_df = event_df.append(df_row, ignore_index=True)

	return event_df

def define_tool_event_region(event, data, hdr):
	Fs = hdr['newFs']

	# convert timestamp to index
	event_index = time_stamp_datetime_to_index(event['time'], hdr)

	# define region
	st = event_index - 120*Fs
	st = st if st > 0 else 0
	end = event_index + 60*Fs
	end = end if end < len(data) else len(data)
	region = list(range(st, end))
	
	# cut region from data
	segment = data.iloc[region, :]

	if len(segment) != 3*60*Fs:
		print('** Wrong segment cut is found! **')
		print('Check duration and aftermath tool selection algo')
		import pdb; pdb.set_trace()

	# define spectrogram region
	ran = 15*Fs
	st = event_index - ran
	end = event_index + ran
	reg = list(range(st, end))
	spectrogram_region = reg
	
	return segment, spectrogram_region

def str_to_time_stamp(string):
	try:
		time_stamp = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S.%f')
	except:
		time_stamp = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
	
	return time_stamp

def events_to_array(events, len_array, labels=[]):
	array = np.zeros(len_array)
	if len(labels)==0: labels = [1]*len(events)
	for i, (st, end) in enumerate(events):
		array[st:end] = labels[i]
		
	return array


# Plotting functions

def do_plot(file_info, nr, data, hdr, spectrogram_dict, version):
	# global variables
	mpl.rcParams['font.size'] = 10
	plt.rcParams["font.family"] = "Arial"
	# seconds_roi = 10
	# flow_seconds_drop = 15
	fontdict = {'fontsize':8}
	ff = 900 
	s = 3
	# create figure
	fig = plt.figure(figsize=(7.6,6))
	fig.suptitle('Patient %s (%s)'%(nr+1, file_info['rec_type'][nr]))
	# create axis with Apnea labels
	ax1 = fig.add_subplot(ff+11)
	ax1.plot(data.Apnea.mask((data.patient_asleep==0) | (data.Apnea<1)),'k')
	ax1.plot(data.Apnea.mask((data.patient_asleep==1) | (data.Apnea<1)),'r')
	ax1.set_title('Apnea labels', fontdict=fontdict,pad=-1)
	ax1.set_ylim([-1,7])
	ax1.set_xticklabels([])
	ax1.xaxis.set_visible(False)
	
	# create axis with sleep staging
	ax = fig.add_subplot(ff+12, sharex=ax1)
	ax.plot(data.Stage,'b')
	ax.set_title('Sleep staging + EEG_events_anno', fontdict=fontdict,pad=-1)
	# create axis with EEG events
	ax.plot(-data.EEG_events_anno.mask(data.patient_asleep==0),'k')
	ax.plot(-data.EEG_events_anno.mask(data.patient_asleep==1),'r')
	ax.set_ylim([-1.5,5.5])
	ax.set_xticklabels([])
	ax.xaxis.set_visible(False)
	
	# plot all signals in separate subplots
	channels = ['ABD','CHEST','PTAF','CFlow','SpO2','ECG','HR']
	for p, c in enumerate(channels):
		if c in data.columns:
			ax = fig.add_subplot(ff+10+s, sharex=ax1)
			ax.plot(data[c].mask(data.patient_asleep==0),'b')
			ax.plot(data[c].mask(data.patient_asleep==1),'r')
			if c == 'SpO2' and 'PulseQuality' in data.columns:
				ax.plot(data.index, data.PulseQuality, 'y', alpha=0.2)
				ax.set_ylim([85,101])
				ax.set_ylabel('%')
			elif c == 'PTAF' and hdr['cpap_start'] != -1 and not version == 'Tool':
				# mark start of cpap
				dt = hdr['cpap_start']
				cpap_on_dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 100000*(dt.microsecond // 100000))
				ax.scatter(cpap_on_dt, data.PTAF.loc[cpap_on_dt], c='k', zorder=10)
			ax.set_title(c, fontdict=fontdict,pad=-1)
			ax.xaxis.set_visible(False) if not p == len(channels)-1 else ax.xaxis.set_visible(True)
			s += 1
		else:
			raise Exception('Channel %s not found during plotting for patient %s ..'%(c,nr+1))

	myFmt = DateFormatter("%H:%M:%S")
	ax.xaxis.set_major_formatter(myFmt)

	if version == 'Tool':
		fig = plt.figure(figsize=(7.6,6))
		ax1 = fig.add_subplot(711)
		ax1.plot(data['F3_M2'].mask(data.patient_asleep==0),'b')
		ax1.plot(data['F3_M2'].mask(data.patient_asleep==1),'r')
		ax1.set_title('F3_M2', fontdict=fontdict,pad=-1)
		ax1.set_ylim([-300,300])
		ax1.xaxis.set_visible(False)

		channels = ['F4_M1', 'C3_M2', 'C4_M1', 'O1_M2', 'O2_M1']
		s=2
		for p, c in enumerate(channels):
			if c in data.columns:
				ax = fig.add_subplot(700+10+s, sharex=ax1)	
				ax.plot(data[c].mask(data.patient_asleep==0),'b')
				ax.plot(data[c].mask(data.patient_asleep==1),'r')
				ax.set_title(c, fontdict=fontdict,pad=-1)
				ax.xaxis.set_visible(False) if not p == len(channels)-1 else ax.xaxis.set_visible(True)
				ax.set_ylim([-300,300])
				s += 1
			else:
				raise Exception('Channel %s not found during plotting for patient %s ..'%(c,nr+1))

		myFmt = DateFormatter("%H:%M:%S")
		ax.xaxis.set_major_formatter(myFmt)

		# plot all spectograms
		# for ch in spectrogram_dict.keys():
		# 	if ch in ['spectrogram_time', 'spectrogram_freq']:
		# 		continue
		# 	plt.figure(figsize=(7.6,6))
		# 	plt.pcolormesh(spectrogram_dict['spectrogram_time'], spectrogram_dict['spectrogram_freq'], spectrogram_dict[ch],
		# 						shading='gouraud', cmap='jet')
		# 	# plt.ylim([0,20])
		# 	ax.set_title(ch, fontdict=fontdict,pad=-1)
		# 	plt.xlabel('Time [sec]')
		# 	plt.ylabel('Frequency [Hz]')
		# 	plt.show()

	# save figure
	# plt.savefig('Test.svg',dpi=600)



def retrieve_signals_and_labels_Wolfgang(path_signal, annotations, sleeplab_table, irec):
	# read in raw data:
	signal, params = load_mgh_signal(path_signal)
	fs = int(params['Fs'])
	signal_len = signal.shape[0]

	# get annotation arrays:
	annotations = annotations_preprocess(annotations, fs)
	apnea = vectorize_respiratory_events(annotations, signal_len)
	stage = vectorize_sleep_stages(annotations, signal_len)
	arousal = vectorize_arousals(annotations, signal_len)
	channel_names = signal.columns
	cpap_annotations = [x for x in annotations['event'] if 'cmh2o' in str(x).lower()]
	cpap_annotations = [x for x in cpap_annotations if '0.0' not in x]
	if len(cpap_annotations) == 0:
		cpap_annotations = [x for x in annotations["event"] if (('treatment' in str(x).lower()) & ('start' in str(x).lower()))]
		cpap_start = -1
	if len(cpap_annotations) > 0:
		cpap_annotations = annotations.loc[np.isin(annotations.event, cpap_annotations)].iloc[0][['time']].values
		cpap_start = cpap_annotations[0]

	start_time = annotations.loc[0, 'time']

	channels = []
	for i, c in enumerate(channel_names):
		if 'ptaf' in c.lower():
			ch = 'PTAF'
		elif 'cflow' in c.lower():
			ch = 'CFlow'
		elif 'spo2' in c.lower():
			ch = 'SpO2'
		else:
			ch = c.upper()	

		channels.append(ch)

	return signal.values, channels, apnea, stage, arousal, fs, cpap_start, start_time




# tool loading functions

def load_tool_segments():
    event_tags = [  'None', 'Obstructive apneas', 'Central apneas', \
                	'Mixed apneas', 'Hypopneas three', 'Hypopneas four', 'RERAs']
    tool_events = {}    
    # run over all event types
    for tag in event_tags:
        # specify bin files
        tool_folder = '/media/cdac/hdd/Auto_respiration_Nassi/Tool/Tool_selection/%s/*.csv'%tag
        bin_files = glob.glob(tool_folder)
        
        # run over all bin files
        for bin_file in bin_files:
            b = bin_file.split('/')[-1].split('.')[0]
            f = pd.read_csv(bin_file, sep='\t',header=0)
            tool_events['%s_%s'%(tag, b)] = f

    return tool_events

def select_tool_events(tool_events):
    selection = {}
    patient_names = np.array([])
    keys = [*tool_events.keys()]
    for b in keys + ['Hypopneas three_bin10']:
        if b != 'Hypopneas three_bin10': 
            e_bin = tool_events[b] 
        else:
            e_bin = tool_events['Hypopneas four_bin10']
            e_bin = e_bin.drop(rem).reset_index(drop=True)

        # sample as many up to k events
        k = 20 if b != 'None_bin11' else 120
        k = k if len(e_bin) >= k else len(e_bin)
        ind = list(range(0,len(e_bin)))

        # select <k> random events from bin list
        ind = random.sample(ind, k)
        selection[b] = e_bin.loc[ind].reset_index(drop=True)

        # retrieve unique patient names
        patient_names = np.concatenate([patient_names, selection[b]['patient_tag'].unique().astype(str)])

        # remember events alrealy taken for hypopneas bin 10
        if b == 'Hypopneas four_bin10':
            rem = ind 

    # return unique patient names and selection
    patient_names = np.unique(patient_names)

    return patient_names, selection

def save_selection(patient_names, selection, patient_file, selection_folder):
	# save patient names in txt file
	with open(patient_file, 'w+') as f:
		f.write('patients:\n')
		for x in patient_names:
			f.write(str(x) + '\n')
	
	# save selection in csv files
	for key in selection.keys():
		output_file = selection_folder + key + '.csv'
		data = selection[key]
		data.to_csv(output_file, header=data.columns, index=None, sep='\t', mode='w+')

def load_selection(patient_file, selection_folder):
	# read in patient names 
	patient_names = pd.read_csv(patient_file, sep='\n').values

	# read in binned event segemnts
	selection = {}
	event_files = glob.glob(selection_folder + '*')
	for f in event_files:
		tag = f.split('/')[-1].split('.')[0]
		selection[tag] = pd.read_csv(f, sep='\t')

	return patient_names, selection

def replace_paths_by_tool_paths(all_paths, research_csv):
	# replace all paths by paths from tool events
	tool_patient_names = research_csv['patient_tag'].values

	tool_patient_paths = []
	for t in np.unique(tool_patient_names):
		tool_patient_paths = np.concatenate([tool_patient_paths,[p for p in all_paths if t in p]])

	return tool_patient_paths

def research_event_to_df(research_csv, patient_tag):
	# specify event info from research.csv
	event_df = pd.DataFrame([], columns=['type', 'bin', 'time'])
	events = research_csv.loc[np.where(research_csv['patient_tag']==patient_tag)]
	for (_,event) in events.iterrows():
		tag = event.bin
		b = tag.split(' ')[-1]
		t = tag.split(b)[0][:-1]
		time_stamp = str_to_time_stamp(event['date_time'])
		event_df = event_df.append({'type': t, 'bin': b, 'time': time_stamp}, ignore_index=True)

	return event_df