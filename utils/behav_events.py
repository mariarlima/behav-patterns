import numpy as np
import pandas as pd
from collections import Counter

'''
CALCULATE PROB BEHAVIOURAL EVENTS IN ACTIVITY SEQUENCES
'''

def probability_sensor_trigger(df):
    """
    Returns counts of each behavioural event 
    """
    data = df
    l = []
    for i in range(len(data)):
        l += data.iloc[i]
    counts = Counter(l)
    return counts

def contingency_prob(df, random = True):
    """
    Compute table with probability of each behavioural event 
    """
    events = list(df.event.unique())
    print(events)
    seq = df.transition.apply(lambda x: x[:-2]).apply(lambda x: x.split('>')) if random else df.transition.apply(lambda x: x[:-8]).apply(lambda x: x.split('>'))
    result = probability_sensor_trigger(seq)
    prob_events = pd.DataFrame(columns=['A(%)'])
    for event in events:
        if event not in result.keys():
            result[event] = 0
        new = np.array([round(result[event]/sum(result.values())*100, 2)]) # percentage
        probs = pd.concat([prob_events, pd.DataFrame(new, columns=['A(%)'], index = [f'{event}'])])
        prob_events = probs
    return probs

def get_array(df, random = True):
    events = list(df.event.unique())
    seq = df.transition.apply(lambda x: x[:-2]).apply(lambda x: x.split('>')) if random else df.transition.apply(lambda x: x[:-8]).apply(lambda x: x.split('>'))
    result = probability_sensor_trigger(seq)
    t = []
    for event in events:
        if event not in result.keys():
            result[event] = 0
        t.append(result[event])
    return t

def get_array_percent(df, random=True):
    events = list(df.event.unique())
    seq = df.transition.apply(lambda x: x[:-2]).apply(lambda x: x.split('>')) if random else df.transition.apply(lambda x: x[6:]).apply(lambda x: x.split('>'))
    result = probability_sensor_trigger(seq)
    t = []
    for event in events:
        if event not in result.keys():
            result[event] = 0
        t.append(round(result[event]/sum(result.values())*100,2))
    return t