import numpy as np
import pandas as pd
import pylab as plt

'''
ENVIRONMENTAL AND VOICE DATA FILTERING ACROSS COHORT
'''

def plot_usage_average_weekly(df, pid, xy):
    pid = pid
    df_u=df.query('@pid in patient_id')

    week_relative = df_u.set_index('timeframe').resample('7D').size()
    month_relative = df_u.set_index('timeframe').resample('30D').size()/4.1
    daily = df_u.set_index('timeframe').resample('1d').size()

    m = np.argmax(list(week_relative))
    m_count = list(week_relative)[m]
    m_day = list(week_relative.index)[m]
    fig, ax = plt.subplots(figsize=(12,4), sharex=True)
    ax.plot(month_relative.index, list(month_relative), color='#56A5EC', marker='o', linewidth=2, markersize=2)
    ax.plot(week_relative.index, list(week_relative),color='#104E8B', marker='o', linewidth=2, markersize=2) #'#104E8B'
    ax.set_title(f'{pid}', fontname="Arial", fontsize=16, fontweight="bold")
    m_day_string = str(m_day).split(' ', 1)[0]
    m_count_new = round(m_count, 2)
    ax.annotate(
            f'Date: {m_day_string}\nN triggers: {m_count_new}',
            xy=(m_day, m_count), xytext=xy,
            textcoords='offset points', ha='left', va='bottom', fontsize='large',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(['Weekly average triggers on monthly basis', 'Weekly number of triggers'], fontsize='x-large')
    return 

def novelty_effect(df, pid):
    """
    Calculate weekly average number of Alexa interactions on a monthly basis
    and total weekly interactions.
    """
    df_p=df.query('@pid in patient_id')
    week_relative = df_p.set_index('timeframe').resample('7D').size()
    novelty_weekly_average = week_relative[:12].sum()/12
    postnovelty_weekly_average = week_relative[12:].sum()/(len(week_relative)-12)
    month_relative = df_p.set_index('timeframe').resample('30D').size()
    total_months = f'total {len(month_relative)} months'
    return pid, novelty_weekly_average, postnovelty_weekly_average, total_months


def topics_novelty(df, pid, topic):
    """
    Compute time slots (mor, aft, eve) for a certain user and topic 
    """
    df_ = df.query('@pid in patient_id')
    df = df_[df_.topic == topic]

    # Get first date and calculate date after 3 months
    first_date = df['timeframe'].iloc[0]
    three_months_later = first_date + pd.DateOffset(months=3)

    # Split df into two
    df_first_3_months = df[df['timeframe'] <= three_months_later]
    total_3_months = len(df_first_3_months)
    df_after_3_months = df[df['timeframe'] > three_months_later]
    total_after_3_months = len(df_after_3_months)
    results = []

    # Loop over the dataframes
    for df in [df_first_3_months, df_after_3_months]:
        c = df['time_slot'].value_counts()
        total = len(df)
        mor = round(c.get('morning', 0)/total * 100,2) if total != 0 else 0
        aft = round(c.get('afternoon', 0)/total * 100, 2) if total != 0 else 0
        eve = round(c.get('evening', 0)/total * 100, 2) if total != 0 else 0

        results.append((mor, aft, eve))

    return results, total_3_months, total_after_3_months