import numpy as np
import pandas as pd
import h5py, sklearn, random
import matplotlib.pyplot as plt
from scipy.stats import chisquare, wasserstein_distance
from sklearn.metrics import (auc, confusion_matrix,
                                precision_recall_curve, roc_curve)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import calibration_curve
from matplotlib.lines import Line2D


save_df_path = './files/'
hf5data_folder = './'


# loading functions
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

# CPAP success prediction
def get_avg_bars():
    # set comparison histograms    
    bars_s = np.array([ 8.58861006e+01, 2.27557174e-03, 2.35882436e-03, 4.10730499e-02,
                        2.25716064e-01, 7.15676405e-01, 1.34515251e+00, 2.98662603e+00,
                        5.25704461e+00, 3.54445646e+00])
    bars_f = np.array([ 7.10282924e+01, 0.00000000e+00, 4.30292599e-03, 7.20926500e-02,
                        2.86860696e-01, 8.82034402e-01, 2.53836400e+00, 5.43247801e+00,
                        1.09304741e+01, 8.85365958e+00])

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
    # INIT ROC/PR figure
    _, axs = plt.subplots(2, 3, figsize=(18,12))

    # load SS bars
    bars = load_SS_histogram_bars(df, bar_output_folder)
    df = predict_CPAP_SUCCESS_from_bars(df, bars)

    # run over the error distance types
    single_tags = ['CAI', 'Burden', 'SS_original', 'SS_improved'] 
    combined_tags = ['CAI+SS+Burden', 'CAI+SS', 'CAI+Burden', 'SS+Burden']
    l_v = '3%'
    print(f'\n\n Compute performance for label version {l_v}')
    for tag in single_tags+combined_tags:
        print('\n*********************************')
        print(f' Assessing > {tag} <')
        print('*********************************')
        # Compute ROC / PR curves
        if tag in single_tags:
            prob, y, _ = compute_logistic_regression(df, l_v, tag, axs=axs[0,:], bars=bars)
            if tag in ['SS_improved', 'SS_original', 'SS_proportion', 'CAI', 'Burden']:
                compute_calibration_curve(prob, y, tag, axs[0][2])
        elif tag in combined_tags:
            prob, y, _ = compute_logistic_regression(df, l_v, tag, axs=axs[1,:])
            compute_calibration_curve(prob, y, tag, axs[1][2])          

    # set layout     
    set_AUC_curve_layout(axs[0,:2], 'Individual')
    set_AUC_curve_layout(axs[1,:2], 'Combined')
    set_calibration_curve_layout([axs[0][2], axs[1][2]])


# Logistic regression
def compute_logistic_regression(df, label_version, tag, axs=[], bars=[], CV_folds=5):
    # set arrays
    x, y, new_df = set_x_for_LG(df, tag, bars, label_version)

    # divide data into CV folds
    xs, ys, inds = set_cross_validation_folds(x, y, folds=CV_folds)

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

    # filter zero CAI1 - mid AHI 
    # z = new_df.iloc[inds].reset_index(drop=True)
    # filtered_ids = np.where(z[f'CAI1_3%']<1)[0]
    # y = y[filtered_ids]
    # prob = prob[filtered_ids]
    # inds = inds[filtered_ids]
    # z = z.iloc[filtered_ids].reset_index(drop=True)
    # print(len(inds))
    # import pdb; pdb.set_trace()

    # do bootstrapping for confidence intervals
    mean_roc, CI_roc = do_bootstrapping(y, pred, prob, my_auc_roc)
    mean_pr, CI_pr = do_bootstrapping(y, pred, prob, my_auc_pr)

    # set line color
    line_color = set_line_color(tag)

    # plot ROC curve
    fpr, tpr, _ = roc_curve(y, prob)
    area = '%.2f [%.2f-%.2f]'%(mean_roc, CI_roc[0], CI_roc[1])
    axs[0].plot(fpr, tpr, line_color, label=f'{tag}${area}')

    # plot PR curve
    precision, recall, _ = precision_recall_curve(y, prob)
    area = '%.2f [%.2f-%.2f]'%(mean_pr, CI_pr[0], CI_pr[1])
    axs[1].plot(recall, precision, line_color, label=f'{tag}${area}')

    return prob, y, inds

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

def set_x_for_LG(df_og, tag, bars, label_version):
    # y --> True CPAP success label
    df = df_og.loc[[i for i, x in enumerate(df_og[f'CPAP success {label_version}']) if x in ['True', 'False']]].copy()
    df[f'CPAP success {label_version}'] = [x=='True' for x in df[f'CPAP success {label_version}']]
    y = np.array([1 if x==False else 0 for x in df[f'CPAP success {label_version}']])

    # x --> predicted values 
    for i, t in enumerate(tag.split('+')):
        if t in ['AHI', 'CAI']:
            x_ = np.expand_dims(df[f'{t}1_{label_version}'], axis=1)
        elif t == 'SS1':
            x_ = np.array(df[f'Total bar dist. SUCCESS'] - df[f'Total bar dist. FAIL'])
            # x_ = (x_-np.min(x_))/(np.max(x_)-np.min(x_))
            x_ = np.expand_dims(x_, axis=1)
        elif t in ['SS', 'SS_improved']:
            x_ = np.array(df[f'SS SUCCESS'] - df[f'SS FAIL'])
            x_ = np.expand_dims(x_, axis=1)
        elif t in ['E_SS1', 'SS_original']:
            x_ = np.expand_dims(df['E_SS1'], axis=1)
        elif t == 'T_SS1':
            x_ = np.expand_dims(df['T_SS1'], axis=1)
        elif t == 'SS_proportion':
            vals = df[f'TSS_prop1_{label_version}'].values
            vals[np.isnan(vals)] = 0
            x_ = np.expand_dims(vals, axis=1)
        elif t == 'Burden':
            x_ = np.expand_dims(df[f'hypoxic_burden_{label_version}'], axis=1)
        elif t == 'bars':
            x_ = bars
        else:
            # difference between the two distances
            x_ = np.array(df[f'{t} SUCCESS'] - df[f'{t} FAIL'])
            # x_ = (x_-np.min(x_))/(np.max(x_)-np.min(x_))
            x_ = np.expand_dims(x_, axis=1)
        x = x_ if i==0 else np.concatenate([x, x_], 1)

    return x, y, df

def do_bootstrapping(y, x, proba, my_stat, n_bootstraps=100, percentage='95%'):
    # init empty arrays
    n_options = y.shape[0]
    index_original = np.arange(n_options).astype(int)
    metrics = []

    # run over all bootstraps
    for n in range(n_bootstraps):
        print('bootstrap #%s / %s'%(n+1, n_bootstraps), end='\r')

        # take random samples 
        index_bootstrap = random.choices(index_original, k=n_options) 

        # compute performance metrics
        true = y[index_bootstrap]
        prob = proba[index_bootstrap]

        # compute metric
        metric = my_stat(true, prob)
        metrics.append(metric)

    # compute mean +- CI intervals
    perc = int(percentage[:-1]) / 100
    metrics = np.array(metrics)
    mean = np.round(np.mean(metrics), 2)
    lower_bound = np.round(np.quantile(metrics, 1-perc, axis=0), 2)
    upper_bound = np.round(np.quantile(metrics, perc, axis=0), 2)
    
    return mean, [lower_bound, upper_bound]


# ROC PR curves 
def my_auc_roc(y, p):
    # compute ROC AUC
    fpr, tpr, _ = roc_curve(y, p)
    AUC = auc(fpr, tpr)

    return AUC

def my_auc_pr(y, p):
    # compute ROC AUC
    precision, recall, _ = precision_recall_curve(y, p)
    AUC = auc(recall, precision)

    return AUC


def compute_cmt_metrics(cmt):
    TP, TN, FN, FP = cmt[0,0], cmt[1,1], cmt[0,1], cmt[1,0]
    acc, sens, spec, prec, f1 = -1, -1, -1, -1, -1
    # accuracy
    if (TP+FP+TN+FN) > 0: 
        acc = ((TP+TN) / (TP+FP+TN+FN))
    # sensitivity
    if (TP+FN) > 0: 
        sens = (TP / (TP+FN))
    # specificity
    if (TN+FP) > 0: 
        spec = (TN / (TN+FP))
    # precision
    if (TP+FP) > 0: 
        prec = (TP / (TP+FP))    
    # F1 score
    if (2*TP + FP + FN) > 0: 
        f1 = ((2*TP) / (2*TP + FP + FN))  

    return acc, sens, spec, prec, f1
 
def set_line_color(tag):
    # single tags
    if tag == 'CAI':
        line_color = 'k'
    elif tag == 'Burden':
        line_color = 'r'
    elif tag == 'Total bar dist.' or tag == 'SS_improved':
        line_color = 'b'
    elif tag == 'bars':
        line_color = 'g'
    elif tag == 'SS1':
        line_color = 'm'
    elif tag == 'T_SS':
        line_color = 'b.-'
    elif tag == 'SS_original':
        line_color = 'b:'
    elif tag == 'SS_proportion':
        line_color = 'b--'
    elif tag == 'Chi-square dist.':
        line_color = 'b--'
    elif tag == 'Wassersteiner dist.':
        line_color = 'b:'
    # combined tags
    elif tag == 'CAI+SS':
        line_color = 'g'
    elif tag == 'CAI+Burden':
        line_color = 'm'
    elif tag == 'SS+Burden':
        line_color = 'y'
    elif tag == 'SS+SS_proportion':
        line_color = 'b'
    elif tag == 'CAI+SS+Burden':
        line_color = 'c'
    elif tag == 'SS_proportion+AHI':
        line_color = 'r--'
    elif tag == 'SS_proportion+CAI+Burden':
        line_color = 'b--'
    

    return line_color

def set_AUC_curve_layout(axs, subtitle):
    # set layout for both curves
    for n in range(2):
        ax = axs[n]
        # set layout
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        # setup legend
        lines, labels = ax.get_legend_handles_labels()
        tags = [l.split('$')[0] for l in labels] 
        vals = [l.split('$')[1] for l in labels]
        # features
        handles = []
        for line, tag in zip(lines, tags):
            handle = Line2D([0],[0], label=tag, c=line.get_c(), linestyle=line.get_linestyle())
            handles.append(handle)
        # auc values
        for val in vals:
            handle = Line2D([0],[0], label=val, c='none')
            handles.append(handle)

        if n==0: subtitle += ' features - AUC '
        title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in subtitle.split(' ')]) + '[95% CI]'
        legend = ax.legend(handles=handles, loc=4, ncol=2, fontsize=9, title=title, title_fontsize=10, alignment='left', 
                    facecolor='grey', framealpha=0.5, edgecolor='k', columnspacing=0)  
        
        # axes
        if n == 0:
            if 'Individual' in subtitle:
                ax.set_title(f'ROC', fontsize=11, weight='bold')
            ax.set_xlabel('1 - specificity', weight='bold', fontsize=10)
            ax.set_ylabel('sensitivity', weight='bold', fontsize=10)
            ax.plot([-0.1, 1.1], [-0.1, 1.1], 'grey', lw=1)
        if n == 1:
            # ax.set_yticks([])
            if 'Individual' in subtitle:
                ax.set_title(f'PR', fontsize=11, weight='bold')
            ax.set_xlabel('sensitivity', weight='bold', fontsize=10)
            ax.set_ylabel('precision', weight='bold', fontsize=10)
            ax.plot([-0.1, 1.1], [1.1, -0.1], 'grey', lw=1)
        

# calibration curve
def compute_calibration_curve(prob, y, tag, ax):    
    # compute calibration curve
    yy, xx = calibration_curve(y, prob)

    # set line color
    line_color = set_line_color(tag)

    # plot calibration curve
    marker = 'o'
    if 'original' in tag:
        marker = '^'
    elif 'proportion' in tag:
        marker = 'x'
    ax.plot(xx, yy, line_color, marker=marker, label=tag, markersize=6)
    
def set_calibration_curve_layout(axs):
    # set layout for both curves
    for n in range(2):
        ax = axs[n]
        # add reference line
        ax.plot([-0.1, 1.1], [-0.1, 1.1], 'grey', lw=1)
        
        # set axes
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_ylabel('Ratio of patients that fail CPAP', weight='bold', fontsize=10)

        # add legend and title
        if n == 0:
            # ax.set_xticks([])
            ax.set_title('Calibration', fontsize=11, weight='bold')
            title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in 'Individual features'.split(' ')])
        elif n == 1:
            title = ''.join([r"$\bf{" + x + "}$" + ' ' for x in 'Combined features'.split(' ')])
        ax.legend(loc=4, ncol=1, fontsize=9, title=title, title_fontsize=10, alignment='left', facecolor='grey', 
                        framealpha=0.5, edgecolor='k')  
        ax.set_xlabel(f'Predicted CPAP failure risk', weight='bold', fontsize=10)
        # ax.set_yticks([])



if __name__ == '__main__':
    dataset = 'mgh'
    date = '03_20_2023'

    # set input folder
    input_folder = hf5data_folder + f'{dataset}_{date}/'

    # set bar output folder
    bar_output_folder = f'{hf5data_folder}{dataset}_histogram_bars_{date}_all/'

    # check if sim_df is availible
    sim_df_path = f'{save_df_path}sim_df_{date}_all.csv'

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

    # compute ROC / PR curves 
    compute_performance_curves(sim_df, bar_output_folder)

    plt.show()
    import pdb; pdb.set_trace()



    
    
