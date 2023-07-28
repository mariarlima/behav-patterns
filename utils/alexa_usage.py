import numpy as np
import pandas as pd
import altair as alt

'''
MINING ALEXA USAGE OVER TIME
'''

names={
    'r':'random',
    's_start':'questionnaire'
    }
    
def get_usage(
    df, 
    pid,
    hours=[-1, 12, 17, 24],
    slots=['morning', 'afternoon', 'evening']
    ):
    """
    Check usage statistics on user level
    """
    df = df.query('@pid in patient_id')
    # drop s_end type
    df = df[df['int_type'] != 's_end'] 
    # replace by 'random' and 'questionnaire'
    df= df.replace({'int_type': names})
    # troubleshooting
    df.date = pd.to_datetime(df.date)
    df = df.assign(time_slot= pd.cut(df.timeframe.dt.hour,hours,labels=slots))
    df = df.reset_index(drop=True)
    df['counts'] =1
    return df

def get_usage_all(
    df, 
    hours = [-1, 12, 17, 24],
    slots = ['morning', 'afternoon', 'evening']
    ):
    """
    Check usage for all users  
    """
    df = df[df['int_type'] != 's_end'] # drop s_end type
    df= df.replace({'int_type': names}) # replace by 'random' and 'questionnaire'
    df = df.assign(time_slot= pd.cut(df.timeframe.dt.hour,hours,labels=slots))
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['timeframe']).dt.date
    df['counts'] =1
    return df

def plot_heatmap_daily(df, user):
    """
    Plot heatmap with daily usage on user level
    """
    plot = alt.Chart(df).mark_rect().encode(
        alt.X('date(timeframe):O', title='day'),
        alt.Y('yearmonth(timeframe):O', title='month'),
        alt.Color('sum(counts):Q', scale=alt.Scale(scheme='purpleblue'), title='N triggers')
        ).properties(
            width=510,
            height=155,
            title = f'{user}'
        )
    return plot

def plot_monthly(df, user):
    """
    Plot monthly usage on user level 
    """
    colours = ['RGB(0,178,238)', 'RGB(16,78,139)']
    rgba_colours = ['rgba' + c[3:-1] + ',0.85)' for c in colours]
    scale=alt.Scale(range=rgba_colours)
    plot = alt.Chart(df).mark_bar().encode(
        alt.X('sum(counts):Q', title='N triggers'),
        alt.Y('yearmonth(timeframe):O', title=''),
        alt.Color(
            'int_type', 
            legend=alt.Legend(title="Interaction type"), 
            # scale=alt.Scale(range=['#CD3278', '#104E8B']))
            scale=scale)
    ).properties(
        width=170,
        height=155,
        title = f'{user}'
    )
    return plot

def get_usage_topics(
    df, 
    hours = [-1, 12, 17, 24],
    slots = ['morning', 'afternoon', 'evening']
    ):
    """
    Filter dataset with predicted topic for all users
    """
    df= df.drop(columns = ['int_type', 'embeddings']).replace({'int_type': names}) # replace by 'random' and 'questionnaire'
    df = df.assign(time_slot= pd.cut(df.timeframe.dt.hour,hours,labels=slots))
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['timeframe']).dt.date
    df['counts'] =1
    return df

def plot_topics_monthly(df, pid):
    """
    Plot prevalence of topics (normalized) for each month per participant 
    """
    df=df.query('@pid in patient_id')
    df_counts =df.groupby('topic').resample('30D', on = 'timeframe').size().reset_index().rename(columns = {0:'count'})
    colours = ['RGB(25,25,112)', 'RGB(16,78,139)', 'RGB(70,130,180)', 'RGB(0,178,238)', 'RGB(135,206,250)', 'RGB(162,181,205)', 'RGB(255,160,122)', 'RGB(238,99,99)', 'RGB(139,34,82)', 'RGB(220,20,60)']
    rgba_colours = ['rgba' + c[3:-1] + ',0.85)' for c in colours]
    scale=alt.Scale(range=rgba_colours)
    ax = alt.Chart(df_counts).mark_bar().encode(
        alt.X('sum(count):Q', title = 'total', stack = 'normalize'),
        alt.Y('yearmonth(timeframe):O', scale=alt.Scale(reverse=True), title = 'month', stack=True),
        alt.Color('topic:N', legend=alt.Legend(title="topic"), scale=scale),
        alt.Order('color_site_sort_index:Q', sort='ascending')
    ).properties(
        width=400,
        height=200,
        title=f"{pid}"
        )
    return ax

def topics_time_prevalence(df, pid, topic):
    """
    Compute time slots (mor, aft, eve) for a certain user and topic 
    """
    df_ = df.query('@pid in patient_id')
    df_filter = df_[df_.topic == topic]
    c  = df_filter.time_slot.value_counts()
    total = len(df_filter)
    mor = round(c.morning/total * 100,2)
    aft = round(c.afternoon/total * 100, 2)
    eve = round(c.evening/total * 100, 2)
    return mor, aft, eve