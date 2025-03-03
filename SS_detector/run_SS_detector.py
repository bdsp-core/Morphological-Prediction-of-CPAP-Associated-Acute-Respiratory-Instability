import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.signal import savgol_filter, convolve

from .Algorithm_functions import *
from ..preprocessing_functions.find_events import find_events, window_correction


# Analysis functions
def compute_event_ratios(data, fs, show=False):
    # divide number of events by sleep time
    sleep_time = np.sum(data.patient_asleep==1) / fs / 3600
    events, apneas = [], []

    for label_tag in ['Apnea', 'flow_reductions']:
        if not label_tag in data.columns:
            print(f'No {label_tag} labels found!')
            events.append(np.nan); apneas.append(np.nan)
            continue
        labels = data[label_tag].fillna(0).astype(int)
        num = len(find_events(labels>0))
        ratio = np.round(num / sleep_time, 1)
        events.append(ratio)
        
        labels[labels>3] = 0
        num = len(find_events(labels>0))
        ratio = np.round(num / sleep_time, 1)
        apneas.append(ratio)

    if show:
        print('Events  / hour:\t %s (T) vs %s (A)'%(events[0], events[1]))
        print('Apnea\'s / hour:\t %s (T) vs %s (A)'%(apneas[0], apneas[1]))

    return events, apneas

def do_self_sim_tests(data, cycle, conv_scores, Fs):
    tests = {}

    # single peak
    d_t = (cycle[1] - cycle[0]) / Fs
    pass_test = True if d_t < 120 and d_t > 10 else False
    tests['duration test'] = (pass_test, d_t)
    
    # relative height test
    sig = data.loc[list(range(cycle[0],cycle[1])), 'Ventilation_combined'].values
    # thresh = data.loc[list(range(cycle[0],cycle[1])), 'overall_hyp_thresh'].values
    thresh = np.nanmean(data.loc[:, 'pos_excursion_hyp'].values)
    pass_test = True if np.any(sig > thresh) else False
    tests['relative height'] = (pass_test, int(pass_test))

    # peak timing score
    p_t = (1 - np.abs((cycle[2]-cycle[0]) - (cycle[1]-cycle[2])) / (cycle[1]-cycle[0])) * 100
    pass_test = True if p_t > 50 else False
    tests['peak timing'] = (pass_test, p_t)

    # vertical mirror score
    # v_s = (1 - np.abs(np.diff(conv_scores[:2]))) * 100
    # pass_test = True if v_s > 70 else False
    # tests['vertical symmetry'] = (pass_test, v_s)
    # if not pass_test:
    #     data.loc[cycle[2], 'TAGGED'] = 2

    # vertical mirror score
    h_s = conv_scores[2] * 100
    pass_test = True if h_s > 50 else False
    tests['horizontal symmetry'] = (pass_test, h_s)

    # vertical mirror score
    # v_m = 1 #conv_scores[2]
    # pass_test = True if v_m > 0.4 else False
    # tests['vertical mirror'] = (pass_test, v_m)


    if np.all([tests[key][0] for key in tests.keys()]):
        data.loc[cycle[2], 'TAGGED'] = 1
    # else:
    #     data.loc[cycle[2], 'TAGGED'] = 2
    return data, tests


# Computation functions
def assess_potential_self_sim_spots(data, Fs):
    # binarize labels
    data['y_algo'] = np.array(data['flow_reductions'] > 0)*-4.
    data['Smooth_pos_envelope'] = 0
    data['Smooth_neg_envelope'] = 0 
    data['TAGGED'] = 0
    # data['overall_hyp_thresh'] = data.loc[:, 'pos_excursion_hyp'].rolling(60*60*Fs, center=True, min_periods=10*60*Fs).apply(np.nanmean)

    # run over each potential self similarity region
    for self_sim_st, self_sim_end in find_events(data.potential_self_sim):
        # get all flow reductions in self similarity region
        self_sim_region = list(range(self_sim_st, self_sim_end))
        flow_lims = find_events(data.loc[self_sim_region, 'flow_reductions'])

        # fig = plt.figure(figsize=(9.5,6))
        for i, (st, end) in enumerate(flow_lims[:-1]):
            # define corrected event locations
            st, end = self_sim_st + st, self_sim_st + end
            next_start, next_end = self_sim_st + flow_lims[i+1][0], self_sim_st + flow_lims[i+1][1]

            # skip events with wake
            # if np.any(data.loc[st:next_end, 'patient_asleep']==0): continue

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
            
            # plot segment
            # fig = plot_segment(data.loc[region_full,:], cycle, conv_scores, Fs, fig, i)

        #     if i == 29: break

        # plt.show()
        # import pdb; pdb.set_trace()

    return data

def compute_envelopes(data, Fs, channels='abd'):
    # set ABD trace to ventilation combined
    data['Ventilation_combined'] = data[channels] if type(channels) != list else data[channels].mean(axis=1)

    # compute envelope and baseline
    new_df = compute_envelope(data['Ventilation_combined'], int(Fs), env_smooth=5)
    data.loc[:, 'Ventilation_pos_envelope'] = new_df['pos_envelope'].values
    data.loc[:, 'Ventilation_neg_envelope'] = new_df['neg_envelope'].values
    data.loc[:, 'Ventilation_default_baseline'] = new_df['baseline'].values
    data.loc[:, 'Ventilation_baseline'] = new_df['correction_baseline'].values
    data.loc[:, 'baseline2'] = new_df['baseline2'].values

    return data

def combine_flow_reductions(data):
    # set data arrays
    apneas = data.Ventilation_drop_apnea
    hypopneas = data.Ventilation_drop_hypopnea
    grouped_hypopneas = find_events(hypopneas)
    data['flow_reductions'] = apneas
    # add hypopneas to apnea array
    for st, end in grouped_hypopneas:
        region = list(range(st, end))
        # insert when no apnea is found in that region
        if np.all(apneas[region] == 0):
            data.loc[region, 'flow_reductions'] = 4
        # if apnea is found, extend apnea region to hypopnea region
        # else:
        #     data.loc[region, 'flow_reductions'] = 1
    
    return data

def merge_small_events(data, Fs):
    # define global hypopnea flow reduction threshold
    global_hyp_thresh = np.nanmedian(data['pos_excursion_hyp'])
    while True:
        # run over all flow reductions
        all_flow_reductions = find_events(data['flow_reductions']>1)
        for i, (st, end) in enumerate(all_flow_reductions[:-1]):
            next_st = all_flow_reductions[i+1][0]
            next_end = all_flow_reductions[i+1][1]
            ss = int(np.median((st, end)))
            ee = int(np.median((next_st, next_end)))
            region = list(range(ss, ee))
            # skip events that would become >2min
            if (next_end-st) > 120*Fs: continue

            # if ventilation trace stays below local or global hyp threshold, merge events (by filling inbetween region)
            if len(data.loc[region, 'pos_excursion_hyp'].dropna()) == 0: continue
            local_thresh = np.nanmedian(data.loc[region, 'pos_excursion_hyp'])
            local_trace = data.loc[region, 'Ventilation_combined']
            if np.all(local_trace < local_thresh) or np.all(local_trace < global_hyp_thresh):
                # only if there is intermittant sleep
                if not np.any(data.loc[list(range(end, next_st)), 'Stage'] == 5):
                    vals, cnts = np.unique(data.loc[region, 'flow_reductions'], return_counts=True)
                    num = 1 if len(vals[1:]) == 1 and vals[1] == 1 else 4
                    data.loc[list(range(st, next_end)), 'flow_reductions'] = num
                    break
        if i == len(all_flow_reductions)-2: 
            break
        
    return data

def remove_wake_events(data):
    # run over all flow reductions
    all_flow_reductions = find_events(data['flow_reductions'])
    for i, (st, end) in enumerate(all_flow_reductions[:-1]):
        region = list(range(st, end))
        if np.sum(data.loc[region, 'patient_asleep']==0) > 0.75*len(region):
            data.loc[region, 'flow_reductions'] = 0

    return data

def tag_potential_self_sim_spots(data, Fs):
    data['potential_self_sim'] = 0
    labels = np.array(data.flow_reductions.fillna(0) > 0).astype(int)

    epoch_size = int(180*Fs)
    epoch_inds = np.arange(0, len(labels)-epoch_size+1, 5*Fs)
    seg_ids = list(map(lambda x:np.arange(x, x+epoch_size), epoch_inds))

    for seg_id in seg_ids:
        try:
            if len(find_events(labels[seg_id])) >= 3:
                data.loc[seg_id, 'potential_self_sim'] = 1
        except:import pdb; pdb.set_trace()

    return data

def post_process_self_sim(data, Fs):
    data['self similarity'] = 0
    data['consecutive complexes'] = 0
    data['ss_conv_score'] = np.nan

    # find 3 consecutive HLG-looking breathing oscillations    
    tagged = data.TAGGED
    window = 180*Fs # use a sliding window with length <window>
    rolling_sum = tagged.rolling(window, center=True).sum()
    data.loc[rolling_sum>=3, 'consecutive complexes'] = 2
    data['consecutive complexes'] = window_correction(data['consecutive complexes'], window_size=window)

    # assess self-similarity for each three complexs
    for st, end in find_events(data['consecutive complexes']):
        # find all HLG complexes within this chain
        complexes = find_events(data.loc[list(range(st,end)), 'TAGGED'])
        if st>0: complexes += st
        # run over each chain and assess each three complexes
        for t in range(len(complexes)-2):
            tags = complexes[t:t+3]
            conv_score = assess_three_breathing_oscillations(data, tags, Fs)
            num = 2 if conv_score > 0.75 else 1
        
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
            
    # plt.plot(data['consecutive complexes'])
    # plt.plot(-data['self similarity'])
    # plt.show()
    # import pdb; pdb.set_trace()

    return data

def compute_smooth_envelope(data, region):
    # analyze the two envelope traces
    for env_tag in ['Smooth_pos_envelope', 'Smooth_neg_envelope']:
        # create smoothed envelope
        original_env = env_tag.replace('Smooth', 'Ventilation')
        data.loc[region, env_tag] = savgol_filter(data.loc[region, original_env], 51, 1)

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
    # for env_tag in ['Smooth_pos_envelope', 'Smooth_neg_envelope']:
    #     # specify region of interest
    #     first_half = np.abs(data.loc[cycle[0]:cycle[2], env_tag].values)
    #     second_half = np.abs(data.loc[cycle[2]:cycle[1], env_tag].values)

    #     # flip first half
    #     first_half = np.flip(first_half)

    #     # determine which half is longer
    #     end = len(first_half) if len(first_half) < len(second_half) else len(second_half)
    #     first_half = first_half[:end]
    #     second_half = second_half[:end]

    #     # normalize envelopes
    #     first_half = (first_half - np.mean(first_half)) / (np.std(first_half) + 0.000001)
    #     second_half = (second_half - np.mean(second_half)) / (np.std(second_half) + 0.000001)

    #     # apply convolution
    #     conv_scores.append(np.nanmax(convolve(first_half, second_half, mode='same')) / len(first_half))

    # specify region of interest
    baseline = data.loc[cycle[0]:cycle[1], 'Ventilation_baseline'].values
    pos = data.loc[cycle[0]:cycle[1], 'Smooth_pos_envelope'].values - baseline
    neg = baseline - data.loc[cycle[0]:cycle[1], 'Smooth_neg_envelope'].values

    # normalize envelopes
    pos = (pos - np.nanmean(pos)) / (np.nanstd(pos) + 0.000001)
    neg = (neg - np.nanmean(neg)) / (np.nanstd(neg) + 0.000001)

    # apply convolution
    pos[np.isnan(pos)] = 0
    neg[np.isnan(neg)] = 0
    val = np.nanmax(convolve(pos, neg, mode='same')) / len(pos)
    conv_scores[2] = val
    # print(val)
    # import pdb; pdb.set_trace()
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

    return np.max(conv_scores)
  

# Plotting functions
def plot_segment(data, cycle, conv_scores, Fs, fig, i):
    # set all signals
    ventilation = data['Ventilation_combined'].values
    sleep_stages = data['Stage'].values
    y_tech = data['y_tech'].values
    y_algo = data['y_algo'].values
    smooth_pos = data['Smooth_pos_envelope'].values
    smooth_neg = data['Smooth_neg_envelope'].values
    baseline = data['Ventilation_baseline'].values
    
    # create plotting arrays
    signal = np.array(ventilation)
    signal[sleep_stages==5] = np.nan
    wake = np.array(ventilation)
    wake[sleep_stages!=5] = np.nan    

    # setup figure
    fontdict = {'fontsize':8, 'ha':'right'}
    ax = fig.add_subplot(6, 5, i+1)

    ###################################################################################################
    _, tests = do_self_sim_tests(data, cycle, conv_scores, Fs)

    # print horizontal / vertical symmetry
    # txt = 'h: %.2f'%(tests['horizontal symmetry'][1])
    # ax.text(len(signal), 3, txt, **fontdict)
    valid = [tests[key][0] for key in tests.keys()]
    failed_tests = np.array([key for key in tests.keys()])[~np.array(valid)]
    sig_color = 'b' if np.all(valid) else 'r'
    
    ###################################################################################################

    # plot signals
    ax.plot(signal, sig_color, lw=.5)
    ax.plot(wake, 'y', lw=.5)
    ax.plot(smooth_pos, 'k')
    ax.plot(smooth_neg, 'k')
    ax.plot(baseline, 'k', lw=0.3)

    # plot labels
    # y_tech[y_tech==0] = np.nan
    # y_algo[y_algo==0] = np.nan
    # ax.plot(y_tech, 'k', lw=3)  
    # ax.plot(y_algo, 'k', lw=3)  

    # mark cycle region
    ax.plot([cycle[0]-data.index[0], cycle[0]-data.index[0]], [-4, 4] , c='k', lw=1, linestyle='dashed')
    ax.plot([cycle[1]-data.index[0], cycle[1]-data.index[0]], [-4, 4] , c='k', lw=1, linestyle='dashed')
    ax.plot([cycle[2]-data.index[0], cycle[2]-data.index[0]], [-4, 4] , c='r', lw=1.5, linestyle='dashed')

    fontdict = {'fontsize':8, 'ha':'left', 'va':'top'}
    # txt = 'pos: %.2f'%conv_scores[0]
    # ax.text(0, 3, txt, **fontdict)
    # txt = 'neg: %.2f'%conv_scores[1]
    # ax.text(0, -3.5, txt, **fontdict)
    if len(failed_tests) > 0:
        txt = 'FAILED:'
        for failed_test in failed_tests:
            txt += '\n%s -> %.2f'%(failed_test, tests[failed_test][1])
        ax.text(0, 4.8, txt, **fontdict)

    
    # set axis layout
    ax.set_ylim([-4, 5])
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', bottom=False)
    

    # if i%5 != 0:
    #     ax.set_yticklabels([])
    # if i < 25:
    #     ax.set_xticklabels([])      

    return fig
 
def self_sim_plot(data, hdr, out_path):
    # take middle 5hr segment --> // 10 rows == 30 min per row
    Fs = hdr['newFs']
    block_size = Fs * 3600 * 5
    # data = original_data.copy(deep=True)
    # if len(original_data) > block_size:
    #     cut = (len(original_data) - block_size) // 2
    #     data = original_data.loc[cut:cut+block_size-1, :].reset_index(drop=True)
        # data = original_data.loc[:block_size-1, :].reset_index(drop=True)
    
    # set signal variables
    signal = data.Ventilation_combined.values
    abd = data.abd.values
    sleep_stages = data.Stage.values
    y_tech = data.Apnea.values
    y_algo = data.flow_reductions.values
    potential_self_sim = data.potential_self_sim.values
    tagged_breaths = data.TAGGED.values
    ss_conv_score = data.ss_conv_score.values
    num_tags = round(len(find_events(data.TAGGED)) / (np.sum(data.patient_asleep==1) / Fs / 3600), 1)
    self_sim = data['self similarity'].values
   
    # define the ids each row
    nrow = 10
    row_ids = np.array_split(np.arange(len(signal)), nrow)
    row_ids.reverse()

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 30

    # color all wake parts red
    sleep = np.array(signal)
    sleep[np.isnan(sleep_stages)] = np.nan
    sleep[sleep_stages==5] = np.nan
    wake = np.zeros(signal.shape)
    wake[np.isnan(sleep_stages)] += signal[np.isnan(sleep_stages)]
    wake[sleep_stages==5] += signal[sleep_stages==5]
    wake[wake==0] = np.nan
    env_pos = data.Smooth_pos_envelope.values
    env_neg = data.Smooth_neg_envelope.values
    baseline = data.Ventilation_baseline.values

    # PLOT SIGNALS
    for ri in range(nrow):
        # plot signal
        ax.plot(sleep[row_ids[ri]]+ri*row_height, c='k', lw=.5)
        ax.plot(wake[row_ids[ri]]+ri*row_height, c='r', lw=.5)

        # add abd
        # ax.plot(abd_sleep[row_ids[ri]]+ri*row_height-10, c='k', lw=1)

        # plot envelopes + baseline
        ax.plot(env_pos[row_ids[ri]]+ri*row_height, c='b', lw=0.5)
        ax.plot(env_neg[row_ids[ri]]+ri*row_height, c='b', lw=0.5)
        ax.plot(baseline[row_ids[ri]]+ri*row_height, c='g', lw=0.8)

    # PLOT LABELS
    for yi in range(6):
        if yi==0:
            labels = y_tech                 # plot tech label
            label_color = [None, 'b', 'g', 'c', 'm', 'r', None, 'g']
        elif yi==1:
            labels = y_algo                 # plot algo label
            label_color = [None, 'k', 'k', 'k', 'm', 'r', None, 'g']
        elif yi == 2:
            labels = potential_self_sim     # shaded area for potential self-sim spots
            label_color = [None, 'k', 'r']
        elif yi == 3:
            labels = tagged_breaths         # '*' for HLG breathing oscillations
            label_color = [None, 'k', 'r']
        elif yi == 5:
            labels = self_sim               # shaded red area for self-sim spots
            label_color = [None, 'k', 'r'] if not '_clean/' in out_path else [None, None, 'r']

        # run over each plot row
        for ri in range(nrow):
            if yi < 2 and not '_clean/' in out_path:
                # plot tech annonation
                ax.axhline(ri*row_height-3*(2**yi), c=[0.5,0.5,0.5], ls='--', lw=0.2)  # gridline
            
            # group all labels and plot them
            loc = 0
            for i, j in groupby(labels[row_ids[ri]]):
                len_j = len(list(j))
               
                if not np.isnan(i) and label_color[int(i)] is not None:
                    # i is the value of the label
                    # list(j) is the list of labels with same value
                    labs = [0, 1] if not '_clean/' in out_path else [0]
                    if yi in labs:
                        # add scored events
                        ax.plot([loc, loc+len_j], [ri*row_height-3*(2**yi)]*2, c=label_color[int(i)], lw=2)
                    # elif yi == 2:
                        # create red area for potential self-sim spots
                        # ax.plot([loc, loc+len_j], [ri*row_height+8]*2, c=label_color[int(i)], lw=2, alpha=1)
                    elif yi == 3 and not '_clean/' in out_path:
                        # tag '*' for self_sim complexes
                        tag = '*' if i == 1 else '\''
                        ax.text(loc, ri*row_height+8, tag, ha='center')
                        c_score = np.round(ss_conv_score[row_ids[ri]][loc], 2)
                        if not np.isnan(c_score):
                            ax.text(loc, ri*row_height+14, str(c_score), ha='center', fontsize=5)
                    elif yi == 5:
                        # create red area self_sim events
                        ax.plot([loc, loc+len_j], [ri*row_height+7]*2, c=label_color[int(i)], lw=2, alpha=1)

                loc += len_j
                
    # plot layout setup
    ax.set_xlim([0, max([len(x) for x in row_ids])])
    ax.axis('off')

    # create title
    events, apneas = compute_event_ratios(data, Fs)
    self_sim_score = np.round((np.sum(data['self similarity']==2) / (np.sum(data.patient_asleep==1))) * 100, 1)
    title = '\nSelf-sim score: %s%%\n%s flagged SS oscillations [/hr sleep]'%(self_sim_score, num_tags)
    title += '\n Events/hr: %s (T) vs %s (A)'%(events[0], events[1])
    title += '\n Apneas/hr: %s (T) vs %s (A)'%(apneas[0], apneas[1])
    plt.title(title) 
    plt.tight_layout()

    # save the figure
    out_path += f' {self_sim_score}%% - {num_tags}.pdf'
    plt.savefig(out_path)
    plt.close()



# RUN ALGO
def run_SS_algorithm(data, hdr, out_path, plot_version='default'):
    # compute envelope and baseline on ABD trace
    channels = out_path.split('/')[-2].split('(')[0].split('_')[0]
    if 'abd' in channels and 'chest' in channels: channels = ['abd','chest']
    data = compute_envelopes(data, hdr['newFs'], channels=channels)
    
    # compute ventilation drops
    data = assess_ventilation(data, hdr)
    if 'Stage' in data.columns:
        data['patient_asleep'] = np.logical_and(data.Stage>0, data.Stage<5)
    else:
        data['Stage'] = 1
        print('No sleep stages available -> set all to "N1"')

    # create flow reduction array
    data = combine_flow_reductions(data)

    # merge small separate event into long events
    data = merge_small_events(data, hdr['newFs'])

    # remove wake events
    data = remove_wake_events(data)

    # find potential self-similarity regions
    data = tag_potential_self_sim_spots(data, hdr['newFs'])

    # assess potential self-similarity regions
    data = assess_potential_self_sim_spots(data, hdr['newFs'])

    # apply AASM-rule post-processing
    data = post_process_self_sim(data, hdr['newFs'])

    # create a beautiful plot
    if 'default' in plot_version:
        self_sim_plot(data, hdr, out_path) 
    if 'clean' in plot_version:
        clean_folder = out_path.split(hdr['patient_tag'])[0][:-1] + '_clean/'
        os.makedirs(clean_folder, exist_ok=True)
        self_sim_plot(data, hdr, clean_folder + hdr['patient_tag']) 
    
    return data




