import numpy as np
import pandas as pd

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
		else:
			# remove first end when priority = event2
			ends = [e for e in ends if loc != e]
			# and convert first start into second start
			starts[np.where(starts==p)[0][0]] = loc

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

def window_correction(array, window_size):
	half_window = int(window_size//2)
	events = find_events(array)
	corr_array = np.array(array)
	
	# run over all events in array
	for st, end in events:
		label = array[st]
		corr_array[st-half_window : end+half_window] = label

	return corr_array.astype(int) 