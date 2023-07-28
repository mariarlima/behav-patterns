import numpy as np
import pandas as pd
import pylab as plt
from collections import Counter
from dcarte.utils import mine_transition


'''
SIMILARITY METHOD
'''

def sim_metric(s_i, s_j, trans_size):
    """
    Calculate similarity accounting for temporal sequence of events 
    and whether elements happened consecutively.
    Similarity metric: Ordering-based Sequence Similarity
    """
    def g_(s1,s2):
        U12 = list((Counter(s1) - Counter(s2)).elements())
        U21 = list((Counter(s2) - Counter(s1)).elements())
        return len(U12)+len(U21)
    def get_abs_common(s1,s2, common):
        sai = np.array([i for i, x in enumerate(s1) if x == common])
        saj = np.array([i for i, x in enumerate(s2) if x == common])
        s = 0
        for i in range(min(len(sai), len(saj))):
            s+=abs(sai[i]-saj[i])
        return s
    def f_(s1, s2, C12):
        total = 0
        for ele in C12:
            common = ele
            total+= get_abs_common(s1,s2,common)
        return total/trans_size
    C12 = list(set(s_i).intersection(s_j))
    g = g_(s_i,s_j)
    f = f_(s_i,s_j, C12)
    dis = (f+g)/(2*trans_size)
    return 1-dis

def sim_matrix(x,tran_size):
    """
    Get the similarity matrix
    """
    n = x.shape[0]
    sim = np.zeros((n,n))
    data = x.str.split('>').to_numpy()
    for i,p1 in enumerate(data):
        for j,p2 in enumerate(data[i:]):
            sim[i,j+i] = sim[j+i,i] = sim_metric(p1,p2,tran_size)
    return sim

def get_sim(
      df,  
      pid,
      transition_value = 'event',
      target = "s_start",
      window = 4, # default as transitions made of 5 events 
      source = False, # source, if false target value becomes sink
      dur = 600 # filter duration of transition lower than 10min (600 s)
      ):
      """
      Mine transition for speficic arguments and compare similarity 
      of each transition with all the previous and subsequent ones
      and construct a dataframe to inspect what happened in specific intervals 
      """
      tran_size = window + 1 # if window changes
      df = df.query('@pid in patient_id ')
      expr = f'source == "{target}"' if source else f'sink == "{target}"'
      trans = mine_transition(df, value = transition_value, window = window)
      trans = trans.query(expr).query(f'dur < {dur}')
      ds = sim_matrix(trans.transition, tran_size) # if window changes
      comp_df = pd.concat([trans.reset_index(drop=True),pd.DataFrame(ds)],axis=1)
      return ds,comp_df

def plot_similarity_types(user, win, dur, data, vitals=True):
    """
    Similarity matrix plots of activity vectors for 
    given user, window and duration parameters.
    Includes different plots for 3 Alexa type events (s_start, s_end, r)
    """
    fig,ax = plt.subplots(1,3,figsize= (16,5))
    ds_1, df_1 = get_sim(data, user, source=False, target='s_start', window=win,dur = dur) #sink
    ds_2, df_2 = get_sim(data, user, source=True, target='s_end', window=win,dur = dur) #source
    ds_3, df_3 = get_sim(data, user ,source=False, target='r', window=win,dur = dur) #sink
    #vmin and vmax define the data range the colormap covers
    u = ax[0].imshow(ds_1,vmin=0.12,vmax=1.0,cmap='hot')
    ax[0].set_title('events preceded questionnaire')
    ax[1].imshow(ds_2,vmin=0.12,vmax=1.0,cmap='hot')
    ax[1].set_title('events followed questionnaire')
    ax[2].imshow(ds_3,vmin=0.12,vmax=1.0,cmap='hot')
    ax[2].set_title('events preceded random')
    desc = 'dataset [all vitals as same event]' if vitals else 'dataset [different vitals events]'
    fig.suptitle(f'{user} window={win}, dur={dur}, {desc}')
    cbar = fig.colorbar(u, ax=ax.ravel().tolist(), shrink=0.95)
    return 

def get_sim_df(pid, data, win=4, dur=600):
    """
    Compute the similarity matrix and full dataframe for different Alexa triggers
    1) before questionnaire
    2) after questionnaire
    3) before random
    """
    ds_1, df_1 = get_sim(data, pid, source=False, target='s_start', window=win,dur = dur) #sink
    ds_2, df_2 = get_sim(data, pid, source=True, target='s_end', window=win,dur = dur) #source
    ds_3, df_3 = get_sim(data, pid, source=False, target='r', window=win,dur = dur) #sink
    return ds_1, ds_2, ds_3, df_1, df_2, df_3

def plot_with_date(df, months, user, sink=True, before_quest=True):
    """
    Plot similarity matrix with dates corresponding to when activity vectors took place
    """
    counts = df.set_index('start_date').resample('1M').count()
    _df_plot = df.loc[:,0:]
    date_locs = [0]
    for d in counts.dur:
        date_locs.append(date_locs[-1] + d)
    fig, ax = plt.subplots(figsize=(15, 12))
    im = ax.imshow(_df_plot,vmin=0.12,vmax=1.0,cmap='hot')
    t = r'$t_n$' if sink else r'$t_1$'
    target = 'Start' if before_quest else 'Random' 
    ax.tick_params(axis='both', which='major', labelsize=14) # tick labels font size
    plt.title(f'{user}, {t}={target}', fontsize= 17) 
    plt.xticks(date_locs[:-1], months,rotation = 45)
    plt.yticks(date_locs[:-1], months)
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.ax.tick_params(labelsize=14) # colorbar font size
    return 