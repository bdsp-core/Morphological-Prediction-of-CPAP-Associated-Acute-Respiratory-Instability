
def get_cohort_channels(cohort):
    
    """
    Different cohorts/datasets have different signal channels and types of annotations available. In the prepared dataset,
    currently in /media/mad3/Projects/SLEEP/CAISR1/data_prepared , we harmonized the data as good as possible, however small
    differences still exist (differences we do not want to lose at this point.). Therefore, make sure to use the correct 
    channel array when you load prepared data.
    Input:
    cohort: cohort name, must be the same as the folder name in the data_prepared directory.
    Output:
    channels: list of channels for the specified cohort. len(channels) == Xy.shape[1]
    """

    if cohort == 'mgh':
        channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 
        'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'spo2', 'ecg', 'stage', 'arousal', 'resp']
    
    elif cohort == 'mgh_resp3':
        channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 
        'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'spo2', 'ecg', 'stage', 'arousal', 'resp', 'resp4']        

    if cohort == 'mgh_v3':
        # MOST UP-TO-DATE VERSION OF MGH DATA.
        # 1. contains resp == resp events with hypopnea3% rule, and resp4 (original mgh labels, in case we need it). USE RESP3 FOR CAISR!
        # 2. contains leg EMG ("rat" and "lat") and limb movement annotations ("limb")

        channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 
                    'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'spo2', 'ecg', 
                    'lat', 'rat', 'stage', 'arousal', 'resp', 'resp4', 'limb']
        
    elif cohort in ['shhs', 'mros', 'mesa']:
        channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'airflow', 'ptaf',
                    'cflow', 'spo2', 'ecg', 'stage', 'arousal', 'resp']

    elif cohort == 'mesa_v2':
        # MOST UP-TO-DATE VERSION OF MESA DATA, contains leg==leg emg and limb==limb movement annotations.
        channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'airflow', 'ptaf', 'cflow', 
        'spo2', 'ecg', 'leg', 'stage', 'arousal', 'resp', 'limb']

    elif cohort == 'mros_v2':
        # MOST UP-TO-DATE VERSION OF MROS DATA, contains leg emgs (lat, rat), and limb==limb movement annotations.
        channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'airflow', 'ptaf', 'cflow', 
        'spo2', 'ecg', 'lat', 'rat', 'stage', 'arousal', 'resp', 'limb']
        
    elif cohort == 'penn':
        channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'airflow', 'ptaf', 
                    'cflow', 'spo2', 'ecg', 'stage_0', 'stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5']

    elif cohort == 'penn_v2':
        channels = ['c3-m2', 'c4-m1', 'e1-m2', 'chin1-chin2', 'abd', 'chest', 'airflow', 'ptaf', 
                    'cflow', 'spo2', 'ecg', 'stage_0', 'stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5',
                    'stage'] # stage == majority vote

    elif cohort == 'robert':
        channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 
                    'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'spo2', 'ecg', 'lat', 'rat',
                    'stage', 'arousal', 'resp', 'limb', # stage, arousal, resp, limb == majority votes.
                    'stage_0', 'arousal_0', 'resp_0', 'limb_0', 
                    'stage_1', 'arousal_1', 'resp_1', 'limb_1', 
                    'stage_2', 'arousal_2', 'resp_2', 'limb_2']

    elif cohort in ['robert_v2', 'robert_v3']:
        channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'e2-m1', 'chin1-chin2', 
                    'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'spo2', 'ecg', 'lat', 'rat',
                    'stage', 'arousal', 'resp', 'limb', 
                    'stage_0', 'arousal_0', 'resp_0', 'limb_0', 
                    'stage_1', 'arousal_1', 'resp_1', 'limb_1', 
                    'stage_2', 'arousal_2', 'resp_2', 'limb_2']

    elif cohort in ['robert_v4']:
        channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'e2-m1', 'chin1-chin2', 
                    'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'spo2', 'ecg', 'lat', 'rat', 
                    'cpap_pressure', 'cpap_on',
                    'stage', 'arousal', 'resp', 'limb',
                    'stage_0', 'arousal_0', 'resp_0', 'limb_0', 
                    'stage_1', 'arousal_1', 'resp_1', 'limb_1', 
                    'stage_2', 'arousal_2', 'resp_2', 'limb_2']

    return channels


def get_channel_idxs(channels, sig_tags):
    import numpy as np

    idxs = []
    for sig in sig_tags:
        idxs.append(np.where(sig==np.array(channels))[0][0])

    assert(len(idxs) == len(sig_tags))
    
    return np.array(idxs)

