import numpy as np
import pandas as pd
import glob, h5py, os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from preprocessing_functions.find_events import find_events
from hypoxic_burden import compute_hypoxic_burden

save_df_path = './csv_files/'
hf5data_folder = './'


# loading functions
def filter_paths(version, all_paths, dataset):
    all_p = [p.split('/')[-1].split('.hf5')[0] for p in all_paths]
    if dataset == 'mgh':
        # select MGH paths according to version
        if version == 'splits': 
            # find split night recordins
            split_recs = select_split_recordings(all_paths, dataset)
            run_paths = [(p, None) for p in all_p if p in split_recs]
            print(f' --> {len(run_paths)} paths found.')
    
    return run_paths

def load_sim_output(path):
    # init DF
    cols = ['E_sim', 'T_sim', 'T_osc', 'apnea', 'sleep_stages', 'test_type','cpap_start',
            'E_SS', 'T_SS', 'newFs','ss_conv_score', 'T_tagged', 'spo2']
    data = pd.DataFrame([], columns=cols)
    
    f = h5py.File(path, 'r')
    for key in cols:
        vals = f[key][:]
        data[key] = vals
    f.close()

    # add mgh V5 labels
    data = load_MGH_labels(path, data)

    # add patient asleep
    data['patient_asleep'] = np.logical_and(data.sleep_stages < 5, data.sleep_stages > 0)

    # header:
    header_fields = ['patient_tag', 'newFs', 'Fs', 'test_type', 'rec_type', 'cpap_start','E_SS', 'T_SS', 'T_osc']
    hdr = {}
    for hf in header_fields:
        if not hf in data.columns: continue
        hdr[hf] = data.loc[0, hf]
        data = data.drop(columns=[hf])

    return data, hdr

def load_MGH_labels(path, data):
    # check if we have MGH v5 labels, else return data
    col = path.split('/')[-1].split('.hf5')[0]
    split_rec_labels = save_df_path + 'split_rec_labels.hf5'
    f = h5py.File(split_rec_labels, 'r')
    channels = f['PathNames'][:].astype(str)
    if col in channels:
        ind = np.where(col==channels)[0][0]
        labels = f['Xy'][:, ind]
        labels = labels[~np.isnan(labels)]
        data['V5_labels'] = labels 
    f.close()

    return data

def select_split_recordings(all_paths, dataset):
    # read split recordings from .csv
    split_recs_csv = save_df_path + f'{dataset.upper()}_split_recs.csv'
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

def remove_bad_signal_recordings(df):
    # list of bad signals quality recordings
    bad_recs = [
        'b346da6',
        'dd20181',
        'b551d67',
        '0be8481',
        'fe5cdc8',
        '97a9256',
        'a58f9df',
        '76f60a8',
        '9b91a5f',
        'ad0ee71',
        'f3a9d3e',
        'f1487f4',
        'd300222',
    ]
    # filter good ids
    IDs = df.subjectID.values
    good_ids = [i for i, ID in enumerate(IDs) if ID[:len(bad_recs[0])] not in bad_recs]
    df = df.loc[good_ids].reset_index(drop=True)

    return df    


# computation functions
def extract_SS_scores(df, row, data, hdr):  
    # add SS scores for the complete night  
    reg = np.arange(len(data)) 
    df.loc[row, f'E_SS'] = np.round((np.sum(data.loc[reg, 'E_sim']==1) / (np.sum(data.loc[reg, 'patient_asleep']==1))) * 100, 1)
    df.loc[row, f'T_SS'] = np.round((np.sum(data.loc[reg, 'T_sim']==1) / (np.sum(data.loc[reg, 'patient_asleep']==1))) * 100, 1)
    
    # in case of split night, compute SS before and after CPAP 
    if hdr['cpap_start'] > 0:
        regions = [np.arange(hdr['cpap_start']), np.arange(hdr['cpap_start'], len(data))]
        for r, reg in enumerate(regions):
            if np.all(data.loc[reg, 'patient_asleep']==0): continue

            # compute SS scores
            E_SS = np.round((np.sum(data.loc[reg, 'E_sim']==1) / (np.sum(data.loc[reg, 'patient_asleep']==1))) * 100, 1)
            T_SS = np.round((np.sum(data.loc[reg, 'T_sim']==1) / (np.sum(data.loc[reg, 'patient_asleep']==1))) * 100, 1)
            df.loc[row, f'E_SS{r+1}'] = E_SS
            df.loc[row, f'T_SS{r+1}'] = T_SS

            # compute AHI/CAI
            runs, tags = ['apnea'], ['4%']
            if 'V5_labels' in data.columns:
                runs += ['V5_labels']
                tags += ['3%']
            else:
                # add RDI / AHI / CAI
                df.loc[row, f'RDI{r+1}_3%'] = np.nan
                df.loc[row, f'AHI{r+1}_3%'] = np.nan
                df.loc[row, f'CAI{r+1}_3%'] = np.nan
            for tag, run in zip(tags, runs):
                rdi, ahi, cai, sleep_time = compute_AHI_CAI(data.loc[reg, run], data.loc[reg, 'sleep_stages'], exclude_wake=True) 
                # add RDI / AHI / CAI
                df.loc[row, f'RDI{r+1}_{tag}'] = rdi
                df.loc[row, f'AHI{r+1}_{tag}'] = ahi
                df.loc[row, f'CAI{r+1}_{tag}'] = cai
                
                # add proportion SS events
                T_prop = compute_proportion_ss_events(data.loc[reg, run], data.loc[reg, 'sleep_stages'].values, data.loc[reg, 'ss_conv_score'])
                df.loc[row, f'TSS_prop{r+1}_{tag}'] = T_prop

    return df

def create_SS_scatter_plot(big_df, dataset):
    df = big_df[['T_SS', 'T_osc', 'E_SS', 'mean_CAI']].dropna()
    # setup figure
    fig, axs = plt.subplots(2,2, figsize=(10,16))
    plt.suptitle(f'Statistics {dataset} dataset\n(N={len(df)})')

    # scatter plot
    for i in range(4):
        if i == 0:
            x = 'T_SS'    
            y = 'E_SS'
            ax = axs[0][0]
        elif i == 1:
            x = 'E_SS'    
            y = 'mean_CAI'
            ax = axs[0][1]
        elif i == 2:
            x = 'T_osc'    
            y = 'mean_CAI'
            ax = axs[1][0]
        elif i == 3:
            x = 'T_SS'    
            y = 'mean_CAI'
            ax = axs[1][1]
        sns.scatterplot(data=df, x=x, y=y, ax=ax, color='k')
        maxi = 1.2*np.max((df[x], df[y]))
        
        # add correlation
        corr_dic = {'ha':'left', 'va':'top', 'fontsize':8}
        sns.regplot(x=x, y=y, data=df, ax=ax, scatter=False, color='r')
        corr = stats.pearsonr(df[x], df[y])
        ax.annotate('Pearson\'s r: %.2g\np-value:       %.1g'%(corr[0], corr[1]), (1, 0.98*maxi), **corr_dic)
        # add diagonal
        # line = np.arange(-1, maxi+1)
        # ax.plot(line, line, 'k--', alpha=0.5, lw=0.5)

        # layout
        ax.set_title(f'Scatterplot: {x} vs {y}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        
        ax.axis([0, maxi, 0, maxi])

    plt.show()

def compute_SS_histogram(data, hdr):
    # segment data into 5 minute windows
    ss_array = data.ss_conv_score.values
    epoch_size = int(round(5*60*hdr['newFs']))
    epoch_inds = np.arange(0, len(ss_array)-epoch_size+1, epoch_size)
    seg_ids = list(map(lambda x:np.arange(x, x+epoch_size), epoch_inds))

    # run over all segments, and save mean SS score
    score_present = data.ss_conv_score.dropna().index.values
    bin_means = np.zeros(len(seg_ids))
    for i, seg_id in enumerate(seg_ids):
        # skip if patient is asleep for < 25% in this segment
        if data.loc[seg_id, 'patient_asleep'].sum() < 0.20*epoch_size: 
            bin_means[i] = -1
            continue
        # if no ss score is found, keep zero
        if not any([True if s in seg_id else False for s in score_present]): continue
        # compute average SS score within window
        bin_means[i] = data.loc[seg_id, 'ss_conv_score'].dropna().mean()
     
    ### create histogram ###
    bars = histogram_bins_to_bars(bin_means)

    return bars

def compute_Osc_histogram(data, hdr):
    # segment data into 5 minute windows
    tagged_array = data.T_tagged.values
    epoch_size = int(round(5*60*hdr['newFs']))
    epoch_inds = np.arange(0, len(tagged_array)-epoch_size+1, epoch_size)
    seg_ids = list(map(lambda x:np.arange(x, x+epoch_size), epoch_inds))

    # run over all segments, and save mean SS score
    score_present = np.where(data.T_tagged > 0)[0]
    bin_sum = np.zeros(len(seg_ids))
    for i, seg_id in enumerate(seg_ids):
        # skip if patient is asleep for < 25% in this segment
        if data.loc[seg_id, 'patient_asleep'].sum() < 0.20*epoch_size: 
            bin_sum[i] = -1
            continue
        # if no ss score is found, keep zero
        if not any([True if s in seg_id else False for s in score_present]): continue
        # compute average SS score within window
        bin_sum[i] = data.loc[seg_id, 'T_tagged'].sum() / 5
        if bin_sum[i] > 1 : bin_sum[i] = 1 
    
    ### create histogram ###
    bars = histogram_bins_to_bars(bin_sum)

    return bars
            
def histogram_bins_to_bars(bins):
    ### create histogram ###
    bins = bins[bins>=0]
    step = 0.1
    steps = np.arange(0, 1.1, step)
    bars = []    
    for block in steps[:-1]:
        # normalize bins
        percentage = sum(np.logical_and(np.array(bins)>=block, np.array(bins)<block+step+0.0001)) / len(bins) * 100
        bars.append(percentage)
    
    return bars

def stage_effect_oscillations(data, hdr):
    # set stage mapping
    stage_map = {'stage_REM':4, 'stage_N1':3, 'stage_N2':2, 'stage_N3':1}

    vals = {}
    for key in stage_map.keys():
        ids = ids_rem = np.where(data.sleep_stages==stage_map[key])[0]
        # compute time in stage
        vals[f'{key}_time'] = len(ids) / 3600 / hdr['newFs']

        # compute total breathing oscillations in stage
        vals[f'{key}_osc'] = np.nansum(data.loc[ids, 'T_tagged'])
    
    return vals

def compute_proportion_ss_events(resp, stage, conv, exclude_wake=True):
    # compute sleep time
    stage[~np.isfinite(stage)] = 0
    patient_asleep = np.logical_and(stage<5, stage>0)
    sleep_time = np.sum(patient_asleep==1) / 36000
    if sleep_time == 0: return 0

    # compute num SS events
    all_count = len(find_events(conv>=0)) if exclude_wake else np.sum(conv>0)
    if all_count == 0: return 0
    SS_count = len(find_events(conv>=0.8)) if exclude_wake else np.sum(conv>0)
    if SS_count == 0: return 0
    SS_per_hrs_sleep = round(SS_count / sleep_time, 2)

    # compute AHI
    vals = np.array(resp[patient_asleep]) if exclude_wake else np.array(resp)
    vals[vals==7] = 4
    vals[vals>4] = 0
    vals[vals<0] = 0
    all_count = len(find_events(vals>0)) if exclude_wake else np.sum(vals>0)
    # if all_count == 0: return np.nan

    # # compute SS event proportion
    # SS_proportion = round(SS_count / all_count * 100, 2)
    
    return SS_per_hrs_sleep

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


# CPAP success prediction
def compute_CPAP_SUCCESS(data, df, row, hdr, label_version='4%'):
    # compute CPAP success
    dd = data.loc[hdr['cpap_start']:]
    col = 'apnea' if label_version == '4%' else 'V5_labels'
    rdi_post, ahi_post, cai_post, _ = compute_AHI_CAI(dd[col].values, dd.sleep_stages.values, exclude_wake=True)
    df.loc[row, f'CPAP success {label_version}'] = -1
    # define success
    if cai_post < 5 and ahi_post < 10: 
        df.loc[row, f'CPAP success {label_version}'] = True
    # define failure
    elif cai_post >= 10 or ahi_post >= 30: 
        df.loc[row, f'CPAP success {label_version}'] = False

    return df
   
def save_SS_histogram_bars(SS_bars, Osc_bars, ID, bar_output_folder):
    SS_bars = np.array(SS_bars)
    Osc_bars = np.array(Osc_bars)
    # save data in .hf5 file
    out_file = bar_output_folder + ID + '.hf5'
    with h5py.File(out_file, 'w') as f:
        dtypef = 'float32'
        dXy = f.create_dataset('SS_bars', shape=SS_bars.shape, dtype=dtypef, compression="gzip")
        dXy[:] = SS_bars.astype(float)

        dtypef = 'float32'
        dXy = f.create_dataset('Osc_bars', shape=Osc_bars.shape, dtype=dtypef, compression="gzip")
        dXy[:] = Osc_bars.astype(float)
    

# plotting functions
def plot_SS_histogram_for_paper(df, locs):
    fsz = 14
    fig, axs = plt.subplots(1, 4, figsize=(18,6)) 
    # title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in title.split(' ')])
    # fig.suptitle(title, fontsize=12)
    # fig.tight_layout()
    for i, (key, loc) in enumerate(zip(df.keys(), locs)):
        # set ax layout
        ax = axs[loc]
        if loc==0:
            ax.set_ylabel('Fraction of 5-min segments', fontsize=fsz, fontweight='bold')
            yticks = [item.get_text() for item in ax.get_yticklabels()]
            ax.set_yticklabels(yticks, fontsize=fsz)

        else: 
            yticks=[]
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_ticks([])
        
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        
        xticks = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticklabels(xticks, fontsize=fsz)
        ax.set_xlabel('SS score', fontsize=fsz, fontweight='bold')

        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 101)
        
        # ax.annotate('ABCD'[loc], (0.5, 99), ha='center', va='top', fontsize=12, fontweight='bold')
        
        ### create histogram ###
        bars = df[key]
        step = 0.1
        bins = np.arange(0, 1.1, step)
        for block, percentage in zip(bins[:-1], bars):
            # plot bar
            c = 'gray' if block < 0.8 else 'red'
            c = 'gray'
            ax.bar(block, percentage, color=c, edgecolor='k', width=step, align='edge')
        
def total_histogram_plot(bars1, bars2):
    # mean histogram
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    mean1, mean2 = np.mean(bars1, 0), np.mean(bars2, 0)
    ranges = np.arange(0.05,1,0.1)
    ax.bar([r-0.023 for r in ranges], mean1, color='g', edgecolor='k', width=0.0455, label=f'success (N={len(bars1)})')
    ax.bar([r+0.023 for r in ranges], mean2, color='r', edgecolor='k', width=0.0455, label=f'failure (N={len(bars2)})')

    # set layout
    ax.set_ylabel('%', fontsize=10, fontweight='bold')
    ax.set_xlabel('SS conv. score', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 101)

    plt.title('Mean Histogram: CPAP success vs failure\nAvg. SS score within 5min segments.', fontweight='bold')
    plt.legend()

def histograms_sorted_on_hypoxia_burden(bars, burdens, rows, sim_df, total_rows, total_columns):
    # run over CPAP success and CPAP failure
    for i, (bars_, burdens_, rows_) in enumerate(zip(bars, burdens, rows)):
        cnt = 0
        # setup figure
        fig, axs = plt.subplots(total_rows, total_columns, figsize=(30,16))
        # sort bars/burdens
        inds = np.flip(np.argsort(burdens_))
        sorted_bars = bars_[inds]
        sorted_burdens = np.array(burdens_)[inds]
        sorted_rows = np.array(rows_)[inds]
        # run over each recording
        for bar, burden, row in zip(sorted_bars, sorted_burdens, sorted_rows):
            df = sim_df.loc[row]
            ax = SS_histogram_ax_layout(cnt, axs, total_rows, total_columns)
            ### create histogram ###
            step = 0.1
            bins = np.arange(0, 1.1, step)
            for block, percentage in zip(bins[:-1], bar):
                c = 'gray' if block < 0.8 else 'red'
                ax.bar(block, percentage, color=c, edgecolor='k', width=step, align='edge')
            # add some annotations:
            study = f'#{int(df.study_num)}'
            spo2 = str(round(burden, 1)) + '%'
            corr_dic = {'ha':'left', 'va':'top', 'fontsize':6, 'fontweight':'bold'}
            ax.annotate(f'{study}\n {spo2}', (0.03, 97), **corr_dic)
            # AHI 
            rdi = f'RDI: {df.rdi} | {df.RDI1} <-> {df.RDI2}'
            ahi = f'AHI: {df.ahi} | {df.AHI1} <-> {df.AHI2}'
            cai = f'CAI: {df.cai} | {df.CAI1} <-> {df.CAI2}'
            # SS score
            t_ss = f'SS(t): {df.T_SS1}%'
            corr_dic = {'ha':'right', 'va':'top', 'fontsize':6}
            ax.annotate(f'{rdi}\n{ahi}\n{cai}\n{t_ss}', (0.97, 97), **corr_dic)
            cnt += 1

        # set title
        tag = 'Avg. SS score within 5min segments.\nSorted based on average SPo2.'
        if i == 0:
            fig.suptitle(f'Histogram (SUCCESS): {tag}', fontweight='bold')
        else:
            fig.suptitle(f'Histogram (FAILURE): {tag}', fontweight='bold')

def SS_histogram_ax_layout(i, axs, total_rows, total_columns):
    row = i // total_columns
    col = i % total_columns
    ax = axs[row][col]
    
    # if col == 0: 
    # ax.set_ylabel('%', fontsize=10, fontweight='bold')
    # else: ax.set_yticks([])
    # if row == total_rows-1:
    ax.set_xlabel('SS conv. score', fontsize=10, fontweight='bold')
    # else: ax.set_xticks([]) 
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 101)

    return ax


if __name__ == '__main__':
    dataset = 'mgh'
    date = '03_20_2023'

    # set input folder
    input_folder = hf5data_folder + f'new_data/'

    # set save folder
    sim_df_path = save_df_path + 'new_sim_df.csv'
    bar_output_folder = f'./new_histogram_bars/'
    os.makedirs(bar_output_folder, exist_ok=True)    

    # init empty params
    skip, count1, count2 = 0, 0, 0
    rows1, rows2, burden1, burden2, = [], [], [], []
    selected = ['c7a87858fdbf', '662a5a', 'b46ebb901', '69944f6fe513c']  # examples from paper
    locs = np.array([3, 2, 1, 0])
    select_dic, select_locs = {}, []
    
    # set all recording paths from dataset
    all_paths = glob.glob(input_folder + '*.hf5')
    run_paths = filter_paths('splits', all_paths, dataset)
    sim_df = pd.DataFrame([])

    # run over all paths
    for p, (path1, _) in enumerate(run_paths):
        study_num = p+1
        subjectID = path1.split('/')[-1].split('.hf')[0]           
        print(f'\nAssessing {dataset} recording: #{p}/{len(run_paths)} --> {subjectID}')

        # load recording
        try:
            path = input_folder + path1 + '.hf5'
            data, hdr = load_sim_output(path)
        except: 
            print(f'ERROR for ID: {subjectID}')

        # put data into dataframe
        row = len(sim_df)
        sim_df.loc[row, 'study_num'] = p+1
        sim_df.loc[row, 'subjectID'] = subjectID
        sim_df.loc[row, 'test_type'] = hdr['test_type']
        sim_df.loc[row, 'cpap_start'] = hdr['cpap_start']
        sim_df.loc[row, 'T_osc'] = hdr['T_osc']
        # add AHI and CAI
        rdi, ahi, cai, sleep_time = compute_AHI_CAI(data.apnea, data.sleep_stages.values, exclude_wake=True)  
        sim_df.loc[row, 'rdi'] = rdi
        sim_df.loc[row, 'ahi'] = ahi
        sim_df.loc[row, 'cai'] = cai
        sim_df.loc[row, 'sleep_time'] = sleep_time

        # compute and add SS scores
        sim_df = extract_SS_scores(sim_df, row, data, hdr)

        # set data before CPAP and compute histograms
        data_seg = data.loc[:hdr['cpap_start'], :].copy()
        SS_bars = compute_SS_histogram(data_seg, hdr)
        Osc_bars = compute_Osc_histogram(data_seg, hdr)

        # add proportion SS events
        if 'V5_labels' in data.columns:
            T_prop = compute_proportion_ss_events(data['V5_labels'], data['sleep_stages'].values, data['ss_conv_score'])
            sim_df.loc[row, f'TSS_prop_all'] = T_prop

        # compute stage effects
        stage_effects_dic = stage_effect_oscillations(data_seg, hdr)
        for key in stage_effects_dic.keys():
            sim_df.loc[row, key] = stage_effects_dic[key]
        
        # save histograms bars
        save_SS_histogram_bars(SS_bars, Osc_bars, path1, bar_output_folder)

        # do for 3% labels and 4% labels
        for lv in ['3%', '4%']:
            # skip, when AHI before CPAP < 10 
            if np.isnan(sim_df.loc[row, f'AHI1_{lv}']): continue
            if sim_df.loc[row, f'AHI1_{lv}'] < 10: continue

            # compute CPAP success, based on events after CPAP
            sim_df = compute_CPAP_SUCCESS(data, sim_df, row, hdr, lv)
            # skip, when CPAP success is undefined
            if sim_df.loc[row, f'CPAP success {lv}'] == -1: continue
            
            # compute hypoxic burden
            hypoxia_burden, hypoxia_time_metrics = compute_hypoxic_burden(data_seg, int(hdr['newFs']), label_version=lv)
            # hypoxia_burden, hypoxia_time_metrics = 1, [1]
            sim_df.loc[row, f'hypoxic_burden_{lv}'] = hypoxia_burden
            sim_df.loc[row, f'T90burden_{lv}'] = hypoxia_time_metrics[0]

        ############################
        # CREATE HISTOGRAM FIGURES #
        ############################
        # save for paper figure
        select = [ID for ID in selected if ID in subjectID]
        if len(select)>0: 
            select_dic[subjectID] = SS_bars
            select_locs.append(locs[[i for i, s in enumerate(selected) if subjectID[:len(s)]==s][0]])

        # CPAP SUCCESS
        if sim_df.loc[row, f'CPAP success 3%'] == True:
            bars1 = np.vstack([bars1, np.array(SS_bars)]) if count1 > 0 else np.array(SS_bars)
            burden1.append(hypoxia_burden)
            rows1.append(row)
            count1 += 1
            
        # CPAP FAILURE
        elif sim_df.loc[row, f'CPAP success 3%'] == False:
            bars2 = np.vstack([bars2, np.array(SS_bars)]) if count2 > 0 else np.array(SS_bars)
            burden2.append(hypoxia_burden)
            rows2.append(row)
            count2 += 1

    # create Paper figure
    # plot_SS_histogram_for_paper(select_dic, select_locs)
    # plt.show()
    # import pdb; pdb.set_trace()

    ################################################################################################################
    # save DataFrame
    sim_df.to_csv(sim_df_path, header=sim_df.columns, index=None, mode='w+')
    import pdb; pdb.set_trace()
    ################################################################################################################
