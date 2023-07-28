import numpy as np
import pandas as pd
import pylab as plt
import datetime
from datetime import timedelta

'''
DATA FILTERING USING HEALTH EVENTS
'''

def seven_days(df, pid, health_events):
    """
    Compute arrays with total usage before and after health events 
    lus the difference between the 2 periods
    """
    pid = pid
    df_p=df.query('@pid in patient_id')
    check = []
    subt = []
    for s, e in health_events:
        before = df_p[(df_p.date >= (s - timedelta(days=7))) & (df_p.date < s)]
        before['health_event'] = 'before'
        after = df_p[(df_p.date > e) & (df_p.date <= (e + timedelta(days=7)))]
        after['health_event'] = 'after'
        sum_b = len(before)
        sum_a = len(after)
        check.append([sum_b, sum_a])
        diff = sum_b - sum_a
        subt.append(diff)
    return check, subt

def get_bef_aft_arrays(array):
    """
    Extract data within interval for each health event considered
    """
    before =[] 
    after =[]
    for (b,a) in array:
        before.append(b)
        after.append(a)
    return before, after

def plot_time_series(df, pid, health_events):
    """
    Visualize health events over time and total Alexa triggers
    """
    df_p=df.query('@pid in patient_id')
    fig, ax = plt.subplots(figsize=(17, 7))
    ax.scatter(df_p.timeframe.dt.date,
            df_p.timeframe.dt.hour,
            color='royalblue')
    # Set title and labels for axes
    ax.set(xlabel="Date",
        ylabel="Time of day",
        title="user11 transitions preceding Alexa questionnaire requests over time")
    # add vertical lines with health events/periods
    for s, e in health_events:
        ax.axvspan(s, e, alpha=0.5, color='red')
    plt.show()
    return