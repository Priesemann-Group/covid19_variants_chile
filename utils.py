import pandas as pd
import numpy as np
from datetime import timedelta

# Define changepoints (we define all vars to surpress the automatic prints)
def get_cps(data_begin, data_end, interval=7, offset=0):
    """
    Generates and returns change point array.
    
    Parameters
    ----------
    data_begin : dateteime
        First date for possible changepoints
    data_end : datetime
        Last date for possible changepoints
    interval : int, optional
        Interval for the proposed cp in days. default:7
    offset : int, optional
        Offset for the first cp to data_begin
    """
    change_points = [
    ]
    count = interval-offset
    for day in pd.date_range(start=data_begin, end=data_end):
        if count/interval >= 1.0:
            # Add cp
            change_points.append(
                dict(  # one possible change point every sunday
                    pr_mean_date_transient=day,
                    pr_sigma_date_transient=1.5,
                    pr_sigma_lambda=0.2,  # wiggle compared to previous point
                    relative_to_previous=True,
                    pr_factor_to_previous=1.0,
                    pr_sigma_transient_len=0.5,
                    pr_median_transient_len=4,
                    pr_median_lambda=0.125,
                )
            )
            count = 1
        else:
            count = count +1
    return change_points

def day_to_week_matrix(sim_begin, sim_end, weeks, fill=False):
    """
    Returns the matrix mapping a day to an week.
    Does more or less the same as pandas resample but we can use it in 
    the model.
    
    Parameters
    ----------
    sim_begin : datetime
    sim_end : datetime
    weeks : array-like, datetimes
        Begining date of week. Normally variants.index
    fill : bool
        Wheater or not to fill the not defined datapoints with ones
    Interval
    [first_week_day,first_week_day+7)
    """
    days = pd.date_range(sim_begin, sim_end)
    m = np.zeros((len(days),len(weeks)))
    for i, d in enumerate(days):
        for j, week_begin in enumerate(weeks):
            week_end = week_begin+timedelta(days=7)
            if d >= week_begin and d < week_end:
                m[i,j] = 1
                
        if fill:
            if d < weeks[0]:
                m[i,0] = 1
            if d >= weeks[-1]:
                m[i,-1] = 1
    return m