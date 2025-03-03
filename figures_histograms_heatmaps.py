import numpy as np
import pandas as pd
import h5py, sklearn
import matplotlib.pyplot as plt
from scipy.stats import chisquare, wasserstein_distance
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from matplotlib.lines import Line2D

save_df_path = './csv_files/'
hf5data_folder = './'


########################################################
# CPAP success prediction
########################################################
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

def get_avg_bars():
    # set comparison histograms
    bars_s = np.array([ 7.52738052e+01, 1.42857143e-03, 1.42857143e-03, 1.75644599e-01,
                        1.42857143e-03, 6.48384420e-01, 1.96722790e+00, 2.29029065e+00,
                        1.01272662e+01, 9.52023822e+00])
    bars_f = np.array([ 6.32268538e+01, 2.50000000e-03, 2.50000000e-03, 2.50000000e-03,
                        3.00119048e-01, 1.50833114e+00, 2.71634615e+00, 6.16296054e+00,
                        8.43986386e+00, 1.76480255e+01])

    return bars_s, bars_f

def estimate_CPAP_SUCCESS(sim_df, row, bars_i, label_version):
    bars_s, bars_f = get_avg_bars()

    # total distance
    bars = [bars_i, bars_s, bars_f]
    sim_df.loc[row, 'Total bar dist. SUCCESS'] = custom_error(bars[0], ref=bars[1])
    sim_df.loc[row, 'Total bar dist. FAIL'] = custom_error(bars[0], ref=bars[2])

    # correct bars for distribution distance
    bars = []
    for b in [bars_i, bars_s, bars_f]:
        b_ = np.array(b)
        b_[b_==0] = 1e-08
        b_[0] = b_[0] - (np.sum(b_)-100)
        bars.append(b_)

    # compute chi-square distance
    chi_success = chisquare(bars[0], f_exp=bars[1])
    chi_failure = chisquare(bars[0], f_exp=bars[2])
    chi = [chi_success[0], chi_failure[0]]
    # set boundaries -100 <--> 100
    for c in range(2):
        chi[c] = chi[c] if chi[c] >= -100 else -100
        chi[c] = chi[c] if chi[c] <= 100 else 100
    sim_df.loc[row, 'Chi-square dist. SUCCESS'] = chi[0]
    sim_df.loc[row, 'Chi-square dist. FAIL'] = chi[1]
    
    # compute wassersteiner distance
    sim_df.loc[row, 'Wassersteiner dist. SUCCESS'] = wasserstein_distance(bars[0], bars[1])
    sim_df.loc[row, 'Wassersteiner dist. FAIL'] = wasserstein_distance(bars[0], bars[2])

    # add # CPAP success based on metric to DF
    tags = ['Total bar dist. ', 'Chi-square dist. ', 'Wassersteiner dist. ']
    for t, tag in enumerate(tags):
        difference = np.diff([sim_df.loc[row, tag+'SUCCESS'], sim_df.loc[row, tag+'FAIL']])
        sim_df.loc[row, f'est_CPAP success{t+1}'] = True if difference > 0 else False

    # print estimations
    tag = 'SUCCESS' if sim_df.loc[row, f'CPAP success {label_version}']  == True else 'FAILURE'
    if sim_df.loc[row, f'CPAP success {label_version}']  == -1: tag = 'N/A'
    print(f'\n> {tag} <')
    # print(f'chi-2 dist.:\t\t{np.round(np.diff(chi), 2)}')
    # print(f'wd dist.:\t\t{np.round(np.diff(wd), 2)}')
    print(f'Total bar dist.:\t{np.round(custom_error(bars[0], ref=bars[1])-custom_error(bars[0], ref=bars[2]), 2)}\n')

    return sim_df

def custom_error(bar, ref):
    # mean error
    error = []
    for i, (b, r) in enumerate(zip(bar, ref)):
        error.append(abs(b-r))
    mean_error = np.mean(error)
    total_error = sum(error)

    return mean_error, total_error
   
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
    
def load_SS_histogram_bars(df, bar_output_folder):
    # set IDs 
    paths = [f'{bar_output_folder}{ID}.hf5' for ID in df.subjectID]
    
    for i, path in enumerate(paths):
        print(f'Loading histogram bars: #{i+1}/{len(paths)}', end='\r')
        f = h5py.File(path, 'r')
        bars = []
        for key in ['SS_bars']:
            bars.append(f[key][:])
        f.close()

        # create bar matrix
        if i == 0: 
            SS_bars = np.array(bars[0])
        else:
            SS_bars = np.vstack([SS_bars, np.array(bars[0])])

    return SS_bars
    
def predict_CPAP_SUCCESS_from_bars(df, bars):
    # set comparison bars
    bars_s, bars_f = get_avg_bars()

    # compute error scores
    for i in range(len(df)):
        print(f'Comparing histogram bars: #{i}/{len(df)}', end='\r')
        # total distance
        _, df.loc[df.index[i], 'SS SUCCESS'] = custom_error(bars[i, :], ref=bars_s)
        _, df.loc[df.index[i], 'SS FAIL'] = custom_error(bars[i, :], ref=bars_f)

    return df


########################################################
# Main performance analysis code
########################################################
def compute_performance_curves(df, bar_output_folder):
    # load SS bars
    bars = load_SS_histogram_bars(df, bar_output_folder)
    df = predict_CPAP_SUCCESS_from_bars(df, bars)

    l_v = '3%'
    
    # compute logistic regression
    compute_logistic_regression(df, l_v)

    # add AHI / CAI histograms
    plot_AHI_CAI_histograms_normalized_undefined(df, l_v)

    # plot hypoxic-burden vs SS heatmap
    plot_hypoxic_vs_SS_heatmap(df, l_v)            

# Logistic regression
def compute_logistic_regression(df, label_version, CV_folds=5):
    # set arrays
    x, y, _ = set_x_for_LG(df, label_version)

    # divide data into CV folds
    xs, ys, _ = set_cross_validation_folds(x, y, folds=CV_folds)

    # run over all CV folds
    for i in range(1, CV_folds+1):
        # do logistic regression on training data
        clf = LogisticRegressionCV(Cs=10, random_state=0, cv=3, class_weight='balanced', scoring='roc_auc').\
                fit(xs[f'tr_fold_{i}'], ys[f'tr_fold_{i}'])

        # save x, y, predictions, and probabilities
        x_te, y_te = xs[f'te_fold_{i}'], ys[f'te_fold_{i}']
        x = x_te if i==1 else np.concatenate([x, x_te])
        y = y_te if i==1 else np.concatenate([y, y_te])
        pred = clf.predict(x_te) if i==1 else np.concatenate([pred, clf.predict(x_te)])
        prob = clf.predict_proba(x_te)[:,1] if i==1 else np.concatenate([prob, clf.predict_proba(x_te)[:,1]]) 

    # print performance
    print(f' CMT:\n{confusion_matrix(y, pred, labels=[0, 1])}')
    print(f' Acc: {np.round(sum(y==pred)/len(y), 2)}')

def set_cross_validation_folds(x, y, folds):
    # shuffle array
    inds = np.array(sklearn.utils.shuffle(range(len(y)), random_state=0))

    # create CV folds
    xs, ys = {}, {}
    split = len(y)//folds
    for i in range(folds):
        if i < folds-1:
            test_inds = inds[split*i:split*(i+1)]
        else: 
            test_inds = inds[split*i:]
        xs[f'tr_fold_{i+1}'] = x[[j for j in range(len(y)) if j not in test_inds]]
        ys[f'tr_fold_{i+1}'] = y[[j for j in range(len(y)) if j not in test_inds]]
        xs[f'te_fold_{i+1}'] = x[test_inds]
        ys[f'te_fold_{i+1}'] = y[test_inds]

    return xs, ys, inds

def set_x_for_LG(df_og, label_version):
    # y --> True CPAP success label
    df = df_og.loc[[i for i, x in enumerate(df_og[f'CPAP success {label_version}']) if x in ['True', 'False']]].copy()
    df[f'CPAP success {label_version}'] = [x=='True' for x in df[f'CPAP success {label_version}']]
    y = np.array([1 if x==False else 0 for x in df[f'CPAP success {label_version}']])

    # x --> predicted values 
    x = np.array(df[f'SS SUCCESS'] - df[f'SS FAIL'])
    x = np.expand_dims(x, axis=1)

    return x, y, df

# Histograms
def plot_AHI_CAI_histograms(df, label_version):
    fig = plt.figure(figsize=(14, 8))

    for tag, ran_max, ran_step, bar_width, plot_loc in zip(['AHI', 'CAI'], [160, 60], [5, 1.88], [4, 1.5], [1, 3]):
        # plot histogram
        all_index = df[f'{tag}1_{label_version}']
        success = df[f'CPAP success {label_version}']
        index_s, index_f = np.array([]), np.array([])
    
        ranges = np.arange(0, ran_max, ran_step)
        for x in ranges:
            index_range = np.logical_and(all_index>=x, all_index<x+ran_step)
            index_s = np.concatenate([index_s, [len(df.iloc[np.where(np.logical_and(index_range, success==True))])]])
            index_f = np.concatenate([index_f, [len(df.iloc[np.where(np.logical_and(index_range, success==False))])]])

        # compute percentages 
        percentages = []
        for i in range(len(index_s)):
            val = ''
            if index_f[i]>0: 
                val = str(round(index_f[i] / (index_f[i]+index_s[i]), 2)) 
                if val[:2] == '0.': val = val[1:]
                elif val[:2] == '1.': val = val[:2]
            elif index_s[i]>0: 
                val = '0'
            
            percentages.append(val)
        # find threshold where everythin is 100%
        plus_100 = [i for i, x in enumerate(percentages) if x not in ['','1.']][-1]

        # create bar plot
        if label_version == '3%':
            ax = plt.subplot(2,2,plot_loc) 
        elif label_version == '4%':
            ax = plt.subplot(2,2, 1+plot_loc) 
        ax.bar([r+bar_width/2 for r in ranges], index_s+index_f, color='k', width=bar_width, label='Total')
        ax.bar([r+bar_width/2 for r in ranges], index_f, color='r', width=bar_width, label='Fails CPAP')
        str_add = 0.04*max(index_s+index_f)
        for i, (x, y) in enumerate(zip([r+bar_width/2 for r in ranges], index_s+index_f)):
            if i <= plus_100:
                y_ = max(y+str_add, 1.5*str_add)
                ax.annotate(percentages[i], (x, y_), ha='center', va='bottom', fontsize=5, weight='bold', c='r')
            elif i==plus_100+1:
                ax.annotate('all fail', (ran_max-bar_width/3, 2*str_add), ha='right', va='bottom', fontsize=5, weight='bold', c='r')
                ax.arrow(x-bar_width/3,str_add, ran_max-(x-bar_width/3), 0, length_includes_head=True, color='k',
                            head_width=str_add, head_length=bar_width/3)
                break
        # layout for lower plot
        if plot_loc == 1 and label_version == '3%':
            ax.set_title('Histograms: AHI and CAI', fontsize=11, weight='bold')
        if plot_loc == 4 and label_version == '4%':
            ax.set_xlabel('Events per hour of sleep', weight='bold', fontsize=10)
        # layout for both
        ax.set_xlim([-0.05*ran_max, 1.05*ran_max])
        ax.set_ylim([0-0.05*max(index_s+index_f), 1.2*max(index_s+index_f)])
        ax.set_ylabel('Fraction of recordings', weight='bold', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        title = title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in f'{tag}-hypopnea rule ({label_version[0]})'.split(' ')])
        ax.legend(title=title, fontsize='x-small', title_fontsize='small', alignment='left')
        ax.patch.set_facecolor('blue')
        ax.patch.set_alpha(0.1)

def compute_sex_age_stratification(sim_df, IDs, tag):
    # set CAISR mapping DF
    df = pd.read_csv(save_df_path + 'caisr_mgh_v4_table1.csv')
    # correct IDs
    IDs = [ID.split('_')[0] for ID in IDs]
    locs = []
    for hash_ID in IDs:
        ID = hash_ID.split('_')[0]
        if not ID in df.HashID.values: continue
        loc = np.where(ID==df.HashID.values)[0][0]
        locs.append(loc)
    df = df.loc[locs, :].reset_index(drop=True)
    assert len(IDs) >= len(locs)-1

    proportion = 0
    if tag == 'Sex':
        # compute male proportion
        totals = (sum(df.Sex=='Male'), sum(df.Sex=='Female'))
        total = (sum(df.Sex=='Male') + sum(df.Sex=='Female'))
        if total > 0: 
            proportion = sum(df.Sex=='Male') / total 

    elif tag == 'Age':
        # compute proportion under age of 60
        totals = (sum(df.age<60),  sum(df.age>=60))
        total = (sum(df.age<60) + sum(df.age>=60))
        if total > 0: 
            proportion = sum(df.age<60) / total 

    elif tag == 'Stage':
        # compute REM SS proportion
        nrem_cols = ['stage_N1_osc', 'stage_N2_osc', 'stage_N3_osc']
        totals = (sim_df['stage_REM_osc'].sum(),  sum(sim_df[nrem_cols].sum()))
        proportion = sum(sim_df[nrem_cols].sum()) / (sim_df['stage_REM_osc'].sum() + sum(sim_df[nrem_cols].sum()))
        # time_proportion = sim_df['rem_time'].sum() / (sim_df['rem_time'].sum() + sim_df['nrem_time'].sum())

    return totals, proportion

def plot_AHI_CAI_histograms_normalized_undefined(df, label_version):
    # set df
    df = df.loc[[i for i, x in enumerate(df[f'CPAP success {label_version}']) if x in ['True', 'False']]].copy()
    df[f'CPAP success {label_version}'] = [x=='True' for x in df[f'CPAP success {label_version}']]

    # create Histogram 
    fsz = 15
    fig = plt.figure(figsize=(16, 8))
    tags = ['CAI', 'AHI', 'SS']
    ran_maxs = [70, 140, 1.4]
    
    # run over each Histogram type
    for ii, (tag, ran_max) in enumerate(zip(tags, ran_maxs)):
        ran_step, bar_width= ran_max/28, ran_max/35
        sbar_width = bar_width/5
        # set type specific variables
        if tag=='AHI':
            rem_max = 139 
            xtick_range = np.arange(0, ran_max+1, 10)
            all_index = df[f'{tag}1_{label_version}'].values
        elif tag == 'CAI':
            rem_max = 59
            xtick_range = np.arange(0, ran_max-1, 20)
            all_index = df[f'{tag}1_{label_version}'].values
        elif tag == 'SS':
            rem_max = 0.99
            xtick_range = np.arange(0, 1.01, 0.2)
            SS = np.array(df[f'SS SUCCESS'] - df[f'SS FAIL'])
            all_index = (SS-np.quantile(SS, 0.05))/(np.quantile(SS, 0.95)-np.quantile(SS, 0.05))
            all_index[all_index<0] = 0
            all_index[all_index>rem_max] = rem_max

        # create histogram bars
        print(f'{sum(all_index>rem_max)} were excluded from {tag} histogram')
        all_index[all_index>rem_max] = np.nan
        success = df[f'CPAP success {label_version}'].values.astype(str)
        index_s, index_f, index_m = np.array([]), np.array([]), np.array([])
        ranges = np.arange(0, ran_max, ran_step)
        for x in ranges:
            index_range = np.logical_and(all_index>=x, all_index<x+ran_step)
            index_s = np.concatenate([index_s, [len(df.iloc[np.where(np.logical_and(index_range, success=='True'))])]])
            index_f = np.concatenate([index_f, [len(df.iloc[np.where(np.logical_and(index_range, success=='False'))])]])
            index_m = np.concatenate([index_m, [len(df.iloc[np.where(np.logical_and(index_range, success=='-1'))])]])

        # compute percentages 
        percentages = []
        for i in range(len(index_s)):
            total = index_f[i]+index_s[i]+index_m[i]
            val = ''
            if index_f[i]>0: 
                val = round(index_f[i] / (total)*100)
            elif index_s[i]>0: 
                val, m_val = 0, 0
            percentages.append(str(val))

        # normalize bars
        xlocs = [r+bar_width/1.5 for r in ranges]
        fails = index_f/(index_s+index_f+index_m)
        succs = np.ones(len(index_s)) - fails
        succs[(index_s+index_f+index_m)==0] = np.nan
        
        # create bar plot
        lw = 1
        ax = plt.subplot(3,1, ii+1) 
        s_label = sum(success=='True')
        f_label = sum(success=='False')
        ax.bar(xlocs, succs, color='w', ec='k', alpha=1, width=bar_width, lw=lw, label=f'CPAP success (N={s_label})', bottom=fails)
        ax.bar(xlocs, fails, color='k', ec='k', alpha=1, width=bar_width, lw=lw, label=f'CPAP failure (N={f_label})')
        
        # add stratification bars
        strat_tags = ['Sex', 'Age', 'Stage']
        for i, strat_tag in enumerate(strat_tags):
            if i!=0: continue # <-- select which run
            # select run specific variables
            if strat_tag == 'Sex':
                tt = ['Male', 'Female']
                color = ['lightskyblue', 'lightcoral']
            elif strat_tag == 'Age':
                tt = [r'$<60$', r'$\geq60$']
                color = ['yellowgreen', 'olivedrab']
            elif strat_tag == 'Stage':
                tt = ['NREM', 'REM']
                strat_tag = ['orchid', 'darkorchid']
                
            # extract proportions
            total0, total1, strat_proportions = 0, 0, []
            for x in ranges:
                index_range = np.logical_and(all_index>=x, all_index<x+ran_step)
                # compute proportion males
                IDs_fail = df.iloc[np.where(np.logical_and(index_range, success=='False'))].subjectID
                IDs_success = df.iloc[np.where(np.logical_and(index_range, success=='True'))].subjectID
                totals_f, prop_fail = compute_sex_age_stratification(df, IDs_fail, strat_tag)
                totals_s, prop_success = compute_sex_age_stratification(df, IDs_success, strat_tag)
                # save male/female totals
                total0 += totals_f[0]+totals_s[0]
                total1 += totals_f[1]+totals_s[1]
                # save proportions
                strat_proportions.append((prop_success, prop_fail))
            # compute vertical bars for sex
            succs_male = succs * np.array([m[0] for m in strat_proportions])
            fails_male = fails * np.array([m[1] for m in strat_proportions])

            # create m/f bars
            xs = np.array([r+bar_width/1.5 -2*sbar_width +4*sbar_width  for r in ranges])
            ym_s = succs_male
            yf_s = succs-succs_male
            ym_f = fails_male
            yf_f = fails-fails_male

            # failure bar
            ids = np.where(yf_f>0)[0]
            label = f'{tt[1]} (N={total1})'
            ax.bar(xs[ids], yf_f[ids], color=color[1], ec='k', lw=lw, width=sbar_width, label=label, bottom=ym_f[ids])
            ids = np.where(ym_f>0)[0]
            label = f'{tt[0]} (N={total0})'
            ax.bar(xs[ids], ym_f[ids], color=color[0], ec='k', lw=lw, width=sbar_width, label=label)
            # success bar
            ids = np.where(ym_s>0)[0]
            ax.bar(xs[ids], ym_s[ids], color=color[0], ec='k', lw=lw, width=sbar_width, bottom=fails[ids])
            ids = np.where(yf_s>0)[0]
            ax.bar(xs[ids], yf_s[ids], color=color[1], ec='k', lw=lw, width=sbar_width, bottom=fails[ids]+ym_s[ids])

        # add totals
        str_add = 0.02
        ystr = succs + fails + str_add
        for i, (x, y) in enumerate(zip(xlocs, ystr)):
            ax.annotate(str(round((index_s+index_f+index_m)[i])), (x, y), ha='center', va='bottom', fontsize=fsz, weight='bold', c='k')

        # set ax labels
        ax.set_xlabel(f'{tag} before CPAP', weight='bold', fontsize=fsz)
        ax.set_ylabel('Fraction of recordings', weight='bold', fontsize=fsz)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # layout for both figures
        ax.set_xlim([-0.005*ran_max, 1.005*ran_max])
        ax.set_ylim([-0.02, 1.1])
        ax.tick_params(axis='both', which='major', labelsize=fsz)
        ax.tick_params(axis='both', which='major', labelsize=fsz)
        ax.set_xticks(xtick_range)
        ticks = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticklabels(ticks, fontsize=fsz)
        if ii == 2:
            # setup legend
            bars, labels = ax.get_legend_handles_labels()
            tags = [l.split(' (')[0] for l in labels] 
            vals = ['('+l.split(' (')[1] for l in labels if '(' in l]
            # features
            handles = []
            for bar, tag in zip(bars, tags):
                handle = bar
                handle.set_label(tag)
                handles.append(handle)
            # N totals
            for val in vals:
                handle = Line2D([0],[0], label=val, c='none', markersize=None, marker=None)
                handles.append(handle)

            ax.legend(handles=handles, fontsize=fsz+4, loc=4, facecolor='w', framealpha=0.3, handletextpad=0.8, 
                        alignment='right', edgecolor='k', frameon=False, ncol=2, columnspacing=-2.5)  
        plt.tight_layout()

# hypoxic-burden vs SS graph
def plot_hypoxic_vs_SS_heatmap(df, label_version):
    fig = plt.figure(figsize=(10, 10))
    if label_version == '3%':
        l = 0
    else:
        l = 1


    hypoxic_max = 165
    for i in range(2):
        if i == 0: 
            ax = plt.subplot(2, 2, 1+(2*l))
            inds = [i for i, x in enumerate(df[f'CPAP success {label_version}']) if x=='True']
            title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in f'Success group'.split(' ')])
        elif i == 1:
            ax = plt.subplot(2, 2, 2+(2*l))
            inds = [i for i, x in enumerate(df[f'CPAP success {label_version}']) if x=='False']
            title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in f'Failure group'.split(' ')])
        
        # filter failure patients
        SS = np.array(df[f'SS SUCCESS'] - df[f'SS FAIL'])
        SS = (SS-np.min(SS))/(np.max(SS)-np.min(SS))
        x = SS[inds]
        # x = prob[inds]
        y = df[f'hypoxic_burden_{label_version}'].values[inds]

        # create heatmap
        xedges = np.arange(0, 1.1, 0.01)
        yedges = np.arange(0, hypoxic_max, 5)
        bins = (xedges, yedges)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        # plot heatmap
        ff = ax.hexbin(x, y, extent=extent, gridsize=16, cmap='inferno') #gist_heat
        
        # add reference line 
        # ax.plot([0.5, 0.5], [0, 160], 'w', lw=1)
        
        # set layout
        ax.axis([0, 1, 0, 160])
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_yticks(np.arange(0, 161, 40))
        if i==0:
            ax.set_ylabel('Hypoxic burden', weight='bold', fontsize=10)

        ax.set_xlabel(f'Self-similarity', weight='bold', fontsize=10)
        
        
        ax.set_title(title, fontsize=11)
    
# Sleep stage effects
def get_stage_proportions(df, stages):
    data_absolute, data_normalized, times = [], [], []
    for stage in stages:
        # absolute number of osc
        osc = df[f'stage_{stage}_osc'].sum()
        data_absolute.append(osc)
        # relative number of osc
        time = df[f'stage_{stage}_time'].sum()
        times.append(time)
        norm = osc / time
        data_normalized.append(norm)

    return data_absolute, data_normalized, times

def func(pct, allvalues):
    # Creating autocpt arguments
    absolute = int(pct / 100.*np.sum(allvalues))
    
    return "{:.1f}%".format(pct, absolute) 

def plot_stage_pie_chart(df):
    # set data + labels
    stages = ['N3', 'N2', 'N1', 'REM']
    data_absolute, data_normalized, times = get_stage_proportions(df, stages)
    
    ## create pie chart##
    colors = ['steelblue'] * 3 + ['lightcoral']
    wp = {'linewidth': 2, 'edgecolor': "black", 'alpha':0.8}
    txt = {'fontsize':12, 'color':"k"}
    
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # run over absolute/relative data
    for i, (ax, data) in enumerate(zip([ax1, ax2], [data_absolute, data_normalized])):
        # set start angle
        startangle = 90 + data[-1]/sum(data)*360/2

        if i==0:
            explode = [0.03, 0.02, 0.01, 0.1]
        else:
            explode = [0.02, 0.02, 0.02, 0.07]
        
        # Creating plot
        wedges, texts, autotexts = ax.pie(data,
                                        autopct = lambda pct: func(pct, data),
                                        pctdistance = 0.7,
                                        explode = explode,
                                        labels = stages,
                                        shadow = False,
                                        colors = colors,
                                        startangle = startangle,
                                        wedgeprops = wp,
                                        radius = 1,
                                        textprops = txt)
        
        # Adding legend
        if i==0:
            tags = ['REM', 'NREM']
            ax.legend(np.flip(wedges[-2:]), tags, loc=0, fontsize=14, alignment='left', 
                        facecolor='grey', framealpha=0.5, edgecolor='k')  
        

        plt.setp(autotexts, size=12)
        if i==0:
            title = 'Absolute\n'
        else:
            title = 'Relative (per hour)\n'
        ax.set_title(title, fontsize=16)



if __name__ == '__main__':
    dataset = 'mgh'
    date = '03_20_2023'

    # set input folder
    input_folder = hf5data_folder + f'{dataset}_{date}/'

    # set bar output folder
    bar_output_folder = f'{hf5data_folder}{dataset}_histogram_bars_{date}_all/'

    # check if sim_df is availible
    sim_df_path = f'{save_df_path}sim_df_{date}_all_proportionSS.csv'

    # load sim DFs
    sim_df = pd.read_csv(sim_df_path)
    sim_df = remove_bad_signal_recordings(sim_df)
    print(f'* Loaded sim DF *')
    l_v = '3%'
    sim_df.loc[sim_df[f'CPAP success {l_v}']=='-1.0', f'CPAP success {l_v}'] = '-1'
    sim_df = sim_df.rename(columns={"hypoxic_burden": f"hypoxic_burden_{l_v}"})
    # print stats
    print(f'Label version: {l_v}')
    for tag, string in zip(['success', 'failure', 'unknown'], ['True', 'False', '-1']):
        num = sum(sim_df[f'CPAP success {l_v}']==string)
        perc = round(num/sum(sim_df[f'CPAP success {l_v}']!='-1')*100, 1)
        if tag=='unknown': perc = '-'
        print(f'{num} ({perc}%) {tag} cases')

    ################################################################################
    plot_stage_pie_chart(sim_df)

    # compute ROC / PR curves 
    compute_performance_curves(sim_df, bar_output_folder)

    plt.show()
    import pdb; pdb.set_trace()



    
    
