import numpy as np
import pandas as pd

'''
ENVIRONMENTAL AND VOICE DATA FILTERING ACROSS COHORT
'''

def filter_total_events(df, pid, s, e):
    """
    Filter all by start and end dates per participant and count total events
    """
    start = pd.to_datetime(s).date()
    end = pd.to_datetime(e).date()
    pid = pid
    df_p = df.query('@pid in patient_id')
    df_p_filter = df_p[(df_p.start_date.dt.date >= start) & (df_p.start_date.dt.date <= end)]
    return df_p_filter

def concat_total_events(df):
    p1 = filter_total_events(df, 'P1', '2021-05-07', '2021-10-23')
    p2 = filter_total_events(df, 'P2', '2021-05-13', '2022-06-05')
    p3 = filter_total_events(df, 'P3', '2021-05-16', '2022-06-01')
    p4 = filter_total_events(df, 'P4', '2021-05-14', '2022-06-05')
    p5 = filter_total_events(df, 'P5', '2021-09-28', '2022-05-30')
    p6 = filter_total_events(df, 'P6', '2021-09-08', '2022-05-26')
    p7 = filter_total_events(df, 'P7', '2021-09-08', '2022-05-29')
    p8 = filter_total_events(df, 'P8', '2021-09-08', '2021-10-01')
    p9 = filter_total_events(df, 'P9', '2021-09-08', '2021-12-19')
    p10 = filter_total_events(df, 'P10', '2021-09-20', '2022-01-05')
    p11 = filter_total_events(df, 'P11', '2021-09-25', '2021-12-07')
    p12 = filter_total_events(df, 'P12', '2021-10-28', '2022-06-05')
    p13 = filter_total_events(df, 'P13', '2021-10-05', '2022-06-06')
    p14 = filter_total_events(df, 'P14', '2021-10-05', '2022-06-04')
    df_new_all = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14], ignore_index=True)
    return df_new_all

def usage_days_months(df):
    """
    Calculate total days (data collection period) per participant
    """
    def calculate_months(start, end):
        return (end.year - start.year) * 12 + (end.month - start.month)
    df_first = df.groupby('patient_id').date.min().rename('first').reset_index()
    df_last = df.groupby('patient_id').date.max().rename('last').reset_index()
    # use df.date column to calculate difference
    df_difference = df.groupby('patient_id').date.apply(lambda x: calculate_months(x.min(), x.max())).rename('usage_months').reset_index()
    df_difference_ = df.groupby('patient_id').date.apply(lambda x: (x.max() - x.min()).days).rename('usage_days').reset_index()
    df_first = df_first.merge(df_last, on = 'patient_id') 
    df_first = df_first.merge(df_difference, on = 'patient_id')
    df_first = df_first.merge(df_difference_, on = 'patient_id')
    return df_first

