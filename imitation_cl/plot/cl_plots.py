from copy import deepcopy
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import datetime
import pandas as pd
from shutil import copyfile
import torch
import seaborn as sns
from pprint import pprint

from imitation_cl.logging.utils import read_dict, write_dict, count_lines, check_dirtree

# Colors for per-task performance
colors = ['#000000', '#696969', '#556b2f', '#8b4513', 
          '#483d8b', '#008000', '#000080', '#9acd32', 
          '#20b2aa', '#8b008b', '#ff0000', '#ffa500', 
          '#aaaa00', '#7cfc00', '#deb887', '#8a2be2',
          '#00ff7f', '#dc143c', '#00bfff', '#0000ff', 
          '#d8bfd8', '#ff7f50', '#ff00ff', '#db7093', 
          '#ff1493', '#ee82ee']

# Colors for last model performance
colors_models = {'FT':'#f58231', 'SI':'#911eb4', 'MAS':'rosybrown', 
                 'HN':'#e6194B', 'SG':'dodgerblue', 'iFlow': '#800000',
                 'CHN': '#3cb44b'}


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def get_eval_files(log_multiseed_dir, eval_file_name='eval_results.json'):
    """Fetches the evaluation logs from a log directory
    containing results for multiple seeds

    Args:
        log_multiseed_dir (str): Path to directory containing logs for multiple seeds
    """
    eval_files = list()
    for f in glob.glob(f'{log_multiseed_dir}/*/{eval_file_name}', recursive=True):
        eval_files.append(f)
    return eval_files

def merge(source, destination):
    """
    Merges 2 dicts recursively.

    If leaves are lists, they are extended.
    If leaves are ints, floats, strs, they are concatenated 
    into a string, if the leaves are not the same.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        elif isinstance(value, list):
            destination[key] += value
        else:
            if destination[key] != value:
                destination[key] = f'{destination[key]},{value}'
            else:
                destination[key] = value

    return destination

def get_cl_metrics(arr):
    """
    Given an array of validation accuracies (each current task along the rows,
    and accuracy of the tasks in the columns), this function computes the 
    CL metrics according to https://arxiv.org/pdf/1810.13166.pdf
    These metrics are computed:
        - Accuracy,
        - Backward Transfer,
        - BWT+,
        - REM,
    """

    n = arr.shape[0]

    # Accuracy considers the average accuracy by considering the diagonal 
    # elements as well as all elements below it
    # This is equivalent to computing the sum of the lower traingular matrix
    # and dividing that sum by N(N+1)/2
    acc = np.sum(np.tril(arr))/(n*(n+1)/2.0)

    # Backward transfer (BWT) 
    bwt = 0.0
    for i in range(1, n):
        for j in range(0, i):
            bwt += (arr[i,j] - arr[j,j])
    bwt /= (n*(n-1)/2.0)   

    rem = 1.0 - np.abs(np.min([bwt, 0.0]))
    bwt_plus = np.max([bwt, 0.0])

    return acc, bwt, bwt_plus, rem

def accuracy(errors, threshold, num_seeds=None):
    """
    Calcuate how many of the errors are lower than the 
    `threshold`

    If `num_seeds` is not None, then `errors` is split into
    `num_seeds` parts, and the accuracy of each part is 
    returned as an array of length `num_seeds`.

    Args:
        errors (numpy array): Errors between individual trajectories (num_traj,)
        threshold (float): Threshold for determining if two trajectories are similar

    Returns:
        float: Percentage of predicted trajectories which are accurate (0.0 to 1.0)
        or
        np array: See explanation above.
    """
    errors = np.array(errors)
    if num_seeds is None:
        return np.sum(errors<=threshold)/errors.shape[0]
    else:
        # We should be able to split errors into num_seeds equal parts
        assert num_seeds>0 and len(errors)%num_seeds==0, f'num_seeds: {num_seeds}, len(errors): {len(errors)}'
        errors = np.split(errors, num_seeds)
        acc = list()
        for error in errors:
            acc.append(np.sum(error<=threshold)/error.shape[0])
        return np.array(acc)

def get_arr(logdir_multiseed, acc_thresh_dict=None, agg_mode='median'):
    """
    Reads a set of evaluation log files for multiple seeds and computes 
    the aggregated metrics with error bounds. Also computes the CL metrics 
    with error bounds.

    Args:
        logdir_multiseed (str): Path to the parent log folder containing folders for different seeds
        acc_thresh_dict (dict): Thresholds used for computing CL metrics
        agg_mode (str): Should be either 'mean' or 'median'
    Returns:
        metric_arr, log_metric_arr, acc_arr, cl_metric_arr, seeds, description, cl_sequence
        TODO: Complete description
    """

    # Fetch the log files for the different seeds
    eval_files = get_eval_files(logdir_multiseed)

    # Create a merged JSON file from the different
    # JSON log files for the different seeds
    eval_dict = None

    for eval_file in eval_files:
        eval_dict_single = read_dict(eval_file)
        if eval_dict is None:
            eval_dict = deepcopy(eval_dict_single)
        else:
            eval_dict = merge(eval_dict, eval_dict_single)
            
    args = eval_dict['args']
    data = eval_dict['data']

    local_home = os.path.expanduser('~')

    # Count the number of seeds
    if isinstance(args['seed'], int):
        seeds = args['seed']
        num_seeds = 1
    else:
        seeds = args['seed'].split(',')
        num_seeds = len(seeds)
    
    description = args['description']

    num_tasks, cl_sequence = count_lines(os.path.join(local_home, '/'.join(args['seq_file'].split('/')[3:])))

    # How many metrics are measured?
    metrics = data['metric_errors']['train_task_0']['eval_task_0'].keys()

    # How many data points per metric?
    num_data_points = len(data['metric_errors']['train_task_0']['eval_task_0']['swept'])

    # Create a numpy array for each metric, which are stored in a dict

    # For storing the trajectory metrics
    mid_metric_arr = dict()
    high_err_metric_arr = dict()
    low_err_metric_arr = dict()

    # For storing the log_10 of the trajectory metrics
    mid_log_metric_arr = dict()
    high_err_log_metric_arr = dict()
    low_err_log_metric_arr = dict()

    # For storing the accuracy of the trajectories
    acc_arr = dict()

    # Initialize the arrays
    for metric in metrics:
        if agg_mode is not None:
            mid_metric_arr[metric] = np.zeros((num_tasks, num_tasks))
            high_err_metric_arr[metric] = np.zeros((num_tasks, num_tasks))
            low_err_metric_arr[metric] = np.zeros((num_tasks, num_tasks))

            mid_log_metric_arr[metric] = np.zeros((num_tasks, num_tasks))
            high_err_log_metric_arr[metric] = np.zeros((num_tasks, num_tasks))
            low_err_log_metric_arr[metric] = np.zeros((num_tasks, num_tasks))

            acc_arr[metric] = np.zeros((num_tasks, num_tasks, num_seeds))
        else:
            mid_metric_arr[metric] = np.zeros((num_tasks, num_tasks, num_data_points))
            mid_log_metric_arr[metric] = np.zeros((num_tasks, num_tasks, num_data_points))

    for task_name in cl_sequence:
        train_index = cl_sequence.index(task_name)

        task_results = data['metric_errors'][f'train_task_{train_index}']

        for eval_index in range(train_index+1):
            eval_results = task_results[f'eval_task_{eval_index}']

            for metric_name, metric_errors in eval_results.items():
                
                if agg_mode == 'mean':
                    mid_metric = np.mean(metric_errors)
                    high_err_metric = mid_metric + np.std(metric_errors)
                    low_err_metric = mid_metric - np.std(metric_errors)

                    mid_log_metric = np.mean(np.log10(metric_errors))
                    high_err_log_metric = mid_log_metric + np.std(np.log10(metric_errors))
                    low_err_log_metric = mid_log_metric - np.std(np.log10(metric_errors))
                elif agg_mode == 'median':
                    mid_metric = np.median(metric_errors)
                    high_err_metric = np.percentile(metric_errors, 75)
                    low_err_metric = np.percentile(metric_errors, 25)

                    mid_log_metric = np.median(np.log10(metric_errors))
                    high_err_log_metric = np.percentile(np.log10(metric_errors), 75)
                    low_err_log_metric = np.percentile(np.log10(metric_errors), 25)
                elif agg_mode is None:
                    # Do not compute aggregate error metrics, return the raw data
                    mid_metric = metric_errors
                    high_err_metric = None
                    low_err_metric = None

                    mid_log_metric = np.log10(metric_errors)
                    high_err_log_metric = None
                    low_err_log_metric = None
                else:
                    raise NotImplementedError(f'Unknown agg_mode: {agg_mode}')

                # `acc` will be an array of length `num_seeds`
                if agg_mode is not None:
                    acc = accuracy(metric_errors, threshold=acc_thresh_dict[metric_name], num_seeds=num_seeds)
                    acc_arr[metric_name][train_index, eval_index] = acc

                    mid_metric_arr[metric_name][train_index, eval_index] = mid_metric
                    high_err_metric_arr[metric_name][train_index, eval_index] = high_err_metric
                    low_err_metric_arr[metric_name][train_index, eval_index] = low_err_metric

                    mid_log_metric_arr[metric_name][train_index, eval_index] = mid_log_metric
                    high_err_log_metric_arr[metric_name][train_index, eval_index] = high_err_log_metric
                    low_err_log_metric_arr[metric_name][train_index, eval_index] = low_err_log_metric
                else:
                    mid_metric_arr[metric_name][train_index, eval_index] = np.array(mid_metric)
                    mid_log_metric_arr[metric_name][train_index, eval_index] = np.array(mid_log_metric)

    if agg_mode is None:
        # If we do not want aggregate metrics
        return mid_metric_arr, mid_log_metric_arr
    else:
        # If we want aggrgate metrics
        metric_arr = {'mid': mid_metric_arr,
                    'high': high_err_metric_arr,
                    'low': low_err_metric_arr}

        log_metric_arr = {'mid': mid_log_metric_arr,
                        'high': high_err_log_metric_arr,
                        'low': low_err_log_metric_arr}

        # Calculate the CL metrics
        mid_cl_metric_arr = dict()
        high_cl_metric_arr = dict()
        low_cl_metric_arr = dict()

        for metric_name in metrics:

            acc_list, bwt_list, bwt_plus_list, rem_list = list(), list(), list(), list()

            for seed_idx in range(num_seeds):
                arr = acc_arr[metric_name][:,:,seed_idx]
                acc, bwt, bwt_plus, rem = get_cl_metrics(arr)
                acc_list.append(acc)
                bwt_list.append(bwt)
                bwt_plus_list.append(bwt_plus)
                rem_list.append(rem)

            if agg_mode == 'mean':
                mid_acc = np.mean(acc_list)
                high_acc = mid_acc + np.std(acc_list)
                low_acc = mid_acc - np.std(acc_list)

                mid_bwt = np.mean(bwt_list)
                high_bwt = mid_bwt + np.std(bwt_list)
                low_bwt = mid_bwt - np.std(bwt_list)

                mid_bwt_plus = np.mean(bwt_plus_list)
                high_bwt_plus = mid_bwt_plus + np.std(bwt_plus_list)
                low_bwt_plus = mid_bwt_plus - np.std(bwt_plus_list)

                mid_rem = np.mean(rem_list)
                high_rem = mid_rem + np.std(rem_list)
                low_rem = mid_rem - np.std(rem_list)

            elif agg_mode == 'median':
                mid_acc = np.median(acc_list)
                high_acc = np.percentile(acc_list, 75)
                low_acc = np.percentile(acc_list, 25)

                mid_bwt = np.median(bwt_list)
                high_bwt = np.percentile(bwt_list, 75)
                low_bwt = np.percentile(bwt_list, 25)

                mid_bwt_plus = np.median(bwt_plus_list)
                high_bwt_plus = np.percentile(bwt_plus_list, 75)
                low_bwt_plus = np.percentile(bwt_plus_list, 25)

                mid_rem = np.median(rem_list)
                high_rem = np.percentile(rem_list, 75)
                low_rem = np.percentile(rem_list, 25)
            else:
                raise NotImplementedError(f'Unknown agg_mode: {agg_mode}')

            mid_cl_metric_arr[metric_name] = {'acc': mid_acc,'bwt': mid_bwt, 'bwt_plus': mid_bwt_plus, 'rem': mid_rem}
            high_cl_metric_arr[metric_name] = {'acc': high_acc,'bwt': high_bwt, 'bwt_plus': high_bwt_plus, 'rem': high_rem}
            low_cl_metric_arr[metric_name] = {'acc': low_acc,'bwt': low_bwt, 'bwt_plus': low_bwt_plus, 'rem': low_rem}

        cl_metric_arr = {'mid': mid_cl_metric_arr,
                        'high': high_cl_metric_arr,
                        'low': low_cl_metric_arr}

        return metric_arr, log_metric_arr, cl_metric_arr, seeds, description, cl_sequence


            


def plot_per_task_comparison(compared_method_codes, log_dirs, axes, y_lims, title, fontsize, cl_sequence, alpha=0.15):

    num_tasks = 0
    per_task_errors_mid = dict()
    per_task_errors_high = dict()
    per_task_errors_low = dict()
    for method_code, log_dir in zip(compared_method_codes, log_dirs):
        per_task_errors_mid[method_code] = dict()
        per_task_errors_high[method_code] = dict()
        per_task_errors_low[method_code] = dict()

        mid_metric_arr, mid_log_metric_arr = get_arr(log_dir, acc_thresh_dict=None, agg_mode=None)

        for metric in mid_log_metric_arr.keys():
            per_task_errors_mid[method_code][metric] = list()
            per_task_errors_high[method_code][metric] = list()
            per_task_errors_low[method_code][metric] = list()

            arr = mid_log_metric_arr[metric]

            (rows, cols, elements) = arr.shape
            num_tasks = rows
            
            for r in range(rows):
                per_task_errors_mid[method_code][metric].append(np.median(arr[r,0:r+1,:]))
                per_task_errors_high[method_code][metric].append(np.quantile(arr[r,0:r+1,:], 0.75))
                per_task_errors_low[method_code][metric].append(np.quantile(arr[r,0:r+1,:], 0.25))

    for k,v in per_task_errors_mid.items():
        if k == 'SG':
            lw = 4
            ms = 7
        else:
            lw = 4
            ms = 7

        x = np.arange(0,num_tasks)

        axes[0].plot(v['swept'], label=k, color=colors_models[k], lw=lw, marker='o', markersize=ms, markeredgecolor=adjust_lightness(colors_models[k], amount=0.5), ls='-')
        high = per_task_errors_high[k]['swept']
        low = per_task_errors_low[k]['swept']
        axes[0].fill_between(x, low, high, color=colors_models[k], alpha=alpha)

        axes[1].plot(v['frechet'], label=k, color=colors_models[k], lw=lw, marker='o', markersize=ms, markeredgecolor=adjust_lightness(colors_models[k], amount=0.5), ls='-')
        high = per_task_errors_high[k]['frechet']
        low = per_task_errors_low[k]['frechet']
        axes[1].fill_between(x, low, high, color=colors_models[k], alpha=alpha)

        axes[2].plot(v['dtw'], label=k, color=colors_models[k], lw=lw, marker='o', markersize=ms, markeredgecolor=adjust_lightness(colors_models[k], amount=0.5), ls='-')
        high = per_task_errors_high[k]['dtw']
        low = per_task_errors_low[k]['dtw']
        axes[2].fill_between(x, low, high, color=colors_models[k], alpha=alpha)

    axes[0].set_ylabel('$\log_{10}$\n(Swept Area error)', fontsize=fontsize)
    axes[1].set_ylabel('$\log_{10}$\n(Frechet error)', fontsize=fontsize)
    axes[2].set_ylabel('$\log_{10}$\n(DTW error)', fontsize=fontsize)

    for i,a in enumerate(axes):
        a.set_ylim(y_lims[i])
        a.grid(True, lw=0.2, color='#bbbbbb')

    #axes[0].set_title(title, fontsize=fontsize)

    cl_sequence_ = list()
    for i,c in enumerate(cl_sequence):
        if i%5 == 0:
            cl_sequence_.append(i)
        else:
            cl_sequence_.append('')


    for i,a in enumerate(axes):

        a.set_xticks(np.arange(num_tasks))

        if i == 2:
            if len(cl_sequence)<=7:
                cl_sequence_ = cl_sequence
                a.set_xticklabels(cl_sequence_, fontsize=fontsize-2, rotation=0, ha='center')
            else:
                a.set_xticklabels(cl_sequence_, fontsize=fontsize-2, rotation=0, ha='center')

        if len(cl_sequence)>7:
            a.xaxis.set_major_locator(ticker.MultipleLocator(5))
            a.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        
        a.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.setp(a.get_yticklabels(), fontsize=fontsize-2)
        plt.setp(a.get_xticklabels(), fontsize=fontsize-2)

    for i,a in enumerate(axes):
        if i<2 :
            a.set_xticklabels([])
        else:
            xlabel='Task' if len(cl_sequence)<=7 else 'Task ID'
            a.set_xlabel(xlabel, fontsize=fontsize)
        
    axes[-1].legend(loc='lower center', bbox_to_anchor=(0.45, -0.6), ncol=6, fontsize=fontsize-2)




def line_plot_multiseed(tril_arr_mid, tril_arr_high_err, tril_arr_low_err, cl_sequence, method_code, ax, xlabel, ylabel, title, fontsize=12):

    num_tasks = tril_arr_mid.shape[0]

    handles = list()
    labels = list()
    for i in range(0, num_tasks):
        task_name = list(cl_sequence)[i]
        x = np.arange(i,num_tasks)
        y_mid = tril_arr_mid[i:,i]
        y_high_err = tril_arr_high_err[i:,i]
        y_low_err = tril_arr_low_err[i:,i]
        h = ax.plot(x, 
                    y_mid, 
                    marker="o", 
                    markersize=2.0, 
                    markeredgecolor=adjust_lightness(colors_models[method_code], 0.2), 
                    markeredgewidth=0.5,
                    lw=1.0, 
                    ls="-", 
                    label=task_name, 
                    color=colors_models[method_code])
        _ = ax.fill_between(x, 
                            y_low_err, 
                            y_high_err, 
                            color=colors_models[method_code], 
                            alpha=0.15)
        handles.append(h)
        labels.append(task_name)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_xticks(np.arange(num_tasks))

    cl_sequence_ = list()
    for i,c in enumerate(cl_sequence):
        if i%5 == 0:
            cl_sequence_.append(i)
        else:
            cl_sequence_.append('')

    if len(cl_sequence)<=7:
        cl_sequence_ = cl_sequence
        ax.set_xticklabels(cl_sequence_, fontsize=fontsize-2, rotation=0, ha='center')
    else:
        ax.set_xticklabels(cl_sequence_, fontsize=fontsize-2, rotation=0, ha='center')

        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    for axis in [ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.grid(True, lw=0.2, color='#bbbbbb')
    return handles, labels

def per_task_performance_multiseed(metric_arr, log_metric_arr, method_code, axes, y_lims, task_names_map, title, fontsize=12, ylabel=None):

    # Remove .mat extension
    cl_sequence = task_names_map.values()
    
    plot_idx = 0
    for metric_name, mid_metric_arr_ in log_metric_arr['mid'].items():
        high_metric_arr_ = log_metric_arr['high'][metric_name]
        low_metric_arr_ = log_metric_arr['low'][metric_name]
        
        xlabel='Task' if len(cl_sequence)<=7 else 'Task ID'

        title_= title
        
        if ylabel is not None:

            # For prettier labels
            if metric_name == 'swept':
                metric_name = 'Swept Area error'
            elif metric_name == 'frechet':
                metric_name = 'Frechet error'
            elif metric_name == 'dtw':
                metric_name = 'DTW error'

            ylabel='$\log_{10}$'+f'({metric_name})'

        h,l = line_plot_multiseed(mid_metric_arr_, 
                                  high_metric_arr_, 
                                  low_metric_arr_, 
                                  cl_sequence, 
                                  method_code,
                                  axes[plot_idx], 
                                  xlabel=xlabel, 
                                  ylabel=ylabel, 
                                  title=title_, 
                                  fontsize=fontsize)   

        #axes[plot_idx].set_yscale('log')

        plot_idx += 1

    for i, ax in enumerate(axes):
        ax.set_ylim(y_lims[i])

    handles, labels = axes[-1].get_legend_handles_labels()
    return handles, labels

def plot(log_dirs, 
         acc_thresh_dict, 
         plot_dir, 
         plot_name,
         compared_method_codes,
         task_names_map_file,
         figsize=(20,5), 
         fontsize=12, 
         y_lims=[(-1.1,2.5),(-2,2),(1,5)], 
         bbox_to_anchor=[1.1, 1.6], 
         ax_titles=None,
         leg_cols=13):

    task_names_map = read_dict(task_names_map_file)

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    iters = (log_dirs[0].split('/')[-1]).split('_')[3].replace('it','')
    repeats = (log_dirs[0].split('/')[-1]).split('_')[4].replace('re','')
    title = f'Iterations: {iters}, End point repeat: {repeats}'

    if ax_titles is None:
        ax_titles = [f"Task embedding dimension: {l.split('_')[-1].replace('te','')}" for l in log_dirs]

    cols = len(log_dirs)

    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = plt.GridSpec(3, 8, figure=fig)

    # row 0
    ax0_0 = fig.add_subplot(gs[0, 0:1])
    ax0_1 = fig.add_subplot(gs[0, 1:2])
    ax0_2 = fig.add_subplot(gs[0, 2:3])
    ax0_3 = fig.add_subplot(gs[0, 3:4])
    ax0_4 = fig.add_subplot(gs[0, 4:5])
    ax0_5 = fig.add_subplot(gs[0, 5:6])
    ax0_6 = fig.add_subplot(gs[0, 6:8])

    # row 1
    ax1_0 = fig.add_subplot(gs[1, 0:1])
    ax1_1 = fig.add_subplot(gs[1, 1:2])
    ax1_2 = fig.add_subplot(gs[1, 2:3])
    ax1_3 = fig.add_subplot(gs[1, 3:4])
    ax1_4 = fig.add_subplot(gs[1, 4:5])
    ax1_5 = fig.add_subplot(gs[1, 5:6])
    ax1_6 = fig.add_subplot(gs[1, 6:8])

    # row 2
    ax2_0 = fig.add_subplot(gs[2, 0:1])
    ax2_1 = fig.add_subplot(gs[2, 1:2])
    ax2_2 = fig.add_subplot(gs[2, 2:3])
    ax2_3 = fig.add_subplot(gs[2, 3:4])
    ax2_4 = fig.add_subplot(gs[2, 4:5])
    ax2_5 = fig.add_subplot(gs[2, 5:6])
    ax2_6 = fig.add_subplot(gs[2, 6:8])

    axes = np.array([[ax0_0, ax0_1, ax0_2, ax0_3, ax0_4, ax0_5, ax0_6],
                     [ax1_0, ax1_1, ax1_2, ax1_3, ax1_4, ax1_5, ax1_6],
                     [ax2_0, ax2_1, ax2_2, ax2_3, ax2_4, ax2_5, ax2_6]])

    #fig, axes = plt.subplots(3, cols, figsize=figsize, sharex=True)           

    plot_per_task_comparison(compared_method_codes, log_dirs, axes[:,-1],y_lims=y_lims, title='Comparison', fontsize=fontsize, cl_sequence=task_names_map.values())     

    metrics = list()
    log_metrics = list()
    cl_metrics = list()
    for i, log_dir in enumerate(log_dirs):
        print(f'Processing {log_dir}')
        if log_dir=='':
            continue
        else:
            metric_arr, log_metric_arr, cl_metric_arr, seeds, description, cl_sequence = get_arr(logdir_multiseed=log_dir, 
                                                                                                 acc_thresh_dict=acc_thresh_dict)
            handles, labels = per_task_performance_multiseed(metric_arr=metric_arr, 
                                                             log_metric_arr=log_metric_arr, 
                                                             axes=axes[:, i], 
                                                             y_lims=y_lims, 
                                                             task_names_map=task_names_map, 
                                                             method_code=compared_method_codes[i],
                                                             title=ax_titles[i],
                                                             fontsize=fontsize,
                                                             ylabel=None if i>0 else True)
            metrics.append(metric_arr)
            log_metrics.append(log_metric_arr)
            cl_metrics.append(cl_metric_arr)
   
    #axes[-1,-1].legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc='center', ncol=leg_cols, fontsize=fontsize)
    #fig.suptitle(title, fontsize=15)
    plt.subplots_adjust(wspace=0.17, hspace=0.11, top=0.92)
    plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')

    return metrics, log_metrics, cl_metrics, seeds, cl_sequence

def plot_split(log_dirs, 
               acc_thresh_dict, 
               plot_dir, 
               plot_name,
               compared_method_codes,
               task_names_map_file,
               figsize_per_task_metric=(20,2), 
               figsize_per_task_cumu=(10,8),
               fontsize=12, 
               y_lims=[(-1.1,2.5),(-2,2),(1,5)], 
               bbox_to_anchor=[1.1, 1.6], 
               ax_titles=None,
               leg_cols=13):

    task_names_map = read_dict(task_names_map_file)

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    if ax_titles is None:
        ax_titles = [f"Task embedding dimension: {l.split('_')[-1].replace('te','')}" for l in log_dirs]

    cols = len(log_dirs)

    # Separte per-task line plots for each metric
    fig_per_task_metrics = dict()
    ax_per_task_metrics = dict()
    metrics_names = ['swept', 'frechet', 'dtw']
    for metric_name in metrics_names:
        fig_per_task_metrics[metric_name], ax_per_task_metrics[metric_name] = plt.subplots(1, 6, figsize=figsize_per_task_metric)

    # 1 plot for cumulative per-task (all metrics)
    fig_per_task_cumu, ax_per_task_cumu = plt.subplots(3, 1, figsize=figsize_per_task_cumu)        

    plot_per_task_comparison(compared_method_codes, log_dirs, ax_per_task_cumu ,y_lims=y_lims, title='Comparison', fontsize=fontsize, cl_sequence=task_names_map.values())     

    metrics = list()
    log_metrics = list()
    cl_metrics = list()
    for i, log_dir in enumerate(log_dirs):
        print(f'Processing {log_dir}')
        if log_dir=='':
            continue
        else:
            metric_arr, log_metric_arr, cl_metric_arr, seeds, description, cl_sequence = get_arr(logdir_multiseed=log_dir, 
                                                                                                 acc_thresh_dict=acc_thresh_dict)

            axes = [ax_per_task_metrics[m][i] for m in metrics_names]

            handles, labels = per_task_performance_multiseed(metric_arr=metric_arr, 
                                                             log_metric_arr=log_metric_arr, 
                                                             axes=axes, 
                                                             y_lims=y_lims, 
                                                             task_names_map=task_names_map, 
                                                             method_code=compared_method_codes[i],
                                                             title=ax_titles[i],
                                                             fontsize=fontsize,
                                                             ylabel=None if i>0 else True)
            metrics.append(metric_arr)
            log_metrics.append(log_metric_arr)
            cl_metrics.append(cl_metric_arr)
   
    #axes[-1,-1].legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc='center', ncol=leg_cols, fontsize=fontsize)
    #fig.suptitle(title, fontsize=15)

    for metric in fig_per_task_metrics.keys():

        fig_per_task_metrics[metric].subplots_adjust(wspace=0.17, hspace=0.05, top=0.92)
        fig_per_task_metrics[metric].savefig(os.path.join(plot_dir, f'{metric}_{plot_name}'), bbox_inches='tight')

    #fig_per_task_cumu.text(0.01, 0.52, 'Error metrics (lower is better)', va='center', rotation='vertical', fontsize=fontsize)
    fig_per_task_cumu.subplots_adjust(wspace=0.17, hspace=0.05, top=0.92)
    fig_per_task_cumu.savefig(os.path.join(plot_dir, f'cumu_{plot_name}'), bbox_inches='tight')

    return metrics, log_metrics, cl_metrics, seeds, cl_sequence

def compute_parameter_size(model_type, multiseed_logdir, task_id, last_task_id=25, verbose=False, split_size=False):

    multiseed_logdir = multiseed_logdir.replace('_processed','')

    if model_type in ['HN','CHN']:

        # Parameter size does not vary with seed, just pick the model from the first log
        model_path = glob.glob(os.path.join(f'{multiseed_logdir}/*/', 'models', f'hnet_{last_task_id}.pth'))[0]

        if torch.cuda.is_available():
            net = torch.load(model_path)
        else:
            net = torch.load(model_path, map_location=torch.device('cpu'))

        task_emb_size = 0
        theta_size = 0
        buffer_size = 0

        # Size of theta does not depend on task_id
        for n,p in net.named_parameters():
            if 'task_embs' in n:
                emb_task_id = int(n.replace('task_embs.', ''))
                if emb_task_id <= task_id:
                    task_emb_size += np.prod(list(p.shape))
            elif 'theta' in n or 'chunk_embs' in n:
                theta_size += np.prod(list(p.shape))
        total_size = task_emb_size + theta_size

        if verbose:
            print(f'task_emb_size={task_emb_size}, theta_size={theta_size}, buffer_size={buffer_size}')

        if split_size:
            return total_size, task_emb_size, theta_size, buffer_size
        else:
            return total_size

    elif model_type in ['FT', 'SI', 'MAS', 'SG']:
        
        # Parameter size does not vary with seed, just pick the model from the first log
        model_path = glob.glob(os.path.join(f'{multiseed_logdir}/*/', 'models', f'node_{last_task_id}.pth'))[0]

        if torch.cuda.is_available():
            net = torch.load(model_path)
        else:
            net = torch.load(model_path, map_location=torch.device('cpu'))

        task_emb_size = 0
        theta_size = 0
        buffer_size = 0

        for k,v in net.state_dict().items():

            # Avoid double counting
            if 'target_network.layer_' in k:
                continue
            
            if 'task_embs' in k:
                emb_task_id = int(k.replace('task_embs.', ''))
                if emb_task_id <= task_id:
                    task_emb_size += np.prod(list(v.shape))
            elif 'target_network.weights' in k:
                theta_size += np.prod(list(v.shape))
            else:
                buffer_size += np.prod(list(v.shape))
                
        total_size = task_emb_size + theta_size + buffer_size

        if model_type == 'SG':
            total_size *= (task_id+1)

        if verbose:
            print(f'task_emb_size={task_emb_size}, theta_size={theta_size}, buffer_size={buffer_size}')

        if split_size:
            return total_size, task_emb_size, theta_size, buffer_size
        else:
            return total_size

    else:
        raise NotImplementedError(f'Unknown model_type {model_type}')

def model_size_efficiency(model_type, multiseed_logdir, total_num_tasks):

    ms = 0

    mem_task0 = compute_parameter_size(model_type=model_type, 
                                       multiseed_logdir=multiseed_logdir, 
                                       task_id=0, 
                                       last_task_id=total_num_tasks-1,
                                       verbose=False)

    for task_id in range(total_num_tasks):

        mem_taskn = compute_parameter_size(model_type=model_type, 
                                           multiseed_logdir=multiseed_logdir, 
                                           task_id=task_id, 
                                           last_task_id=total_num_tasks-1,
                                           verbose=False)

        ms += mem_task0/mem_taskn

    ms /= total_num_tasks
    return min(1.0, ms)

def compute_time_efficiency(log_multiseed_dir, agg_mode='median'):

    log_dirs = get_eval_files(log_multiseed_dir, eval_file_name="log.log")

    time_efficiencies = list()

    regex = 'Training started for task_id'

    for log_path in log_dirs:

        date_str = log_path.split('/')[-2].split('_')[0]
        date_str = f'{date_str[0:2]}:{date_str[2:4]}:{date_str[4:6]}'
        date = datetime.datetime.strptime(date_str, '%y:%m:%d').date()
        next_date = date + datetime.timedelta(days=1)
        next_date_str = next_date.strftime('%y:%m:%d')
    
        task_times = dict()
        with open(log_path, 'r') as file:
            for i,line in enumerate(file):

                # Filter the lines matching the regex
                # We will pick the line matching the regex and the next line
                # and then use the logged times to compute the time taken 
                # for training
                for match in re.finditer(regex, line, re.S):
                    # Find the Task ID
                    task_id = int(line.split('task_id: ')[1].split(' ')[0].strip())
                    t_start = datetime.datetime.strptime(f"{date_str}:{line.split(' ')[0].replace('[','').replace(']','')}", '%y:%m:%d:%H:%M:%S')
                    t_end = datetime.datetime.strptime(f"{date_str}:{next(file).split(' ')[0].replace('[','').replace(']','')}", '%y:%m:%d:%H:%M:%S')
                    t_diff = (t_end - t_start).total_seconds()
                    # If the training ended on the next day
                    if t_diff < 0:
                        t_end = datetime.datetime.strptime(f"{next_date_str}:{next(file).split(' ')[0].replace('[','').replace(']','')}", '%y:%m:%d:%H:%M:%S')
                        t_diff = (t_end - t_start).total_seconds()
                    # End time must be greater than the start
                    assert t_diff > 0, f't_end:{t_end}<t_start:{t_start}'
                    task_times[task_id] = t_diff

        # Compute the time efficiency
        num_tasks = len(list(task_times.keys()))
        summed_time_ratio = 0.0
        for task_id, time in task_times.items():
            summed_time_ratio += task_times[0]/time
        summed_time_ratio /= num_tasks
        time_efficiency = min(1.0, summed_time_ratio)

        time_efficiencies.append(time_efficiency)

    if agg_mode == 'median':
        mid = np.median(time_efficiencies)
        high = np.percentile(time_efficiencies, 75)
        low = np.percentile(time_efficiencies, 25)
    elif agg_mode == 'mean':
        mid = np.mean(time_efficiencies)
        high = mid + np.std(time_efficiencies)
        low = mid - np.std(time_efficiencies)

    return mid, high, low, time_efficiencies

def process_upperbaseline_eval(log_multiseed_dir, processed_log_dir=None):
    """When the training is done with a single network per task, the evaluation
    is only done for the current task. Therefore, the eval log file contains entries
    such as:

    {train_task_0: {eval_task_0: <>},
     train_task_1: {eval_task_1: <>},
     ...
     train_task_25: {eval_task_25: <>},}

    This function converts this log into the following format
    by copying evaluation results from past tasks, so that the format of the
    log file becomes similar to the other methods and the same plotting functions
    can be used for these logs as well.

    {train_task_0: {eval_task_0: <>},
     train_task_1: {eval_task_0: <>, eval_task_1: <>},
     train_task_2: {eval_task_0: <>, eval_task_1: <>, eval_task_2: <>},
     ...
     train_task_25: {eval_task_0: <>, ..., eval_task_25: <>},}

    Args:
        log_multiseed_dir ([type]): [description]
        processed_log_dir ([type], optional): [description]. Defaults to None.
    """

    logs = get_eval_files(log_multiseed_dir, eval_file_name="eval_results.json")

    for log in logs:
        eval_dict = read_dict(log)

        metrics = eval_dict['data']['metrics']
        metrics_copy = deepcopy(metrics)
        metric_errors = eval_dict['data']['metric_errors']
        metrics_errors_copy = deepcopy(metric_errors)

        for k in metrics.keys():
            train_task_id = int(k.replace('train_task_',''))
            for eval_task_id in range(0,train_task_id):
                metrics_copy[k][f'eval_task_{eval_task_id}'] = metrics[f'train_task_{eval_task_id}'][f'eval_task_{eval_task_id}']
                metrics_errors_copy[k][f'eval_task_{eval_task_id}'] = metric_errors[f'train_task_{eval_task_id}'][f'eval_task_{eval_task_id}']

        eval_dict['data']['metrics'] = metrics_copy
        eval_dict['data']['metric_errors'] = metrics_errors_copy

        processed_log_path = log.replace(log_multiseed_dir, processed_log_dir)
        if not os.path.isdir(os.path.dirname(processed_log_path)):
            os.makedirs(os.path.dirname(processed_log_path))
        write_dict(processed_log_path, eval_dict)

        # Copy the log.log file containing timestamp information
        # This is needed for computing the time efficiency CL metric
        copyfile(os.path.join(os.path.dirname(log), 'log.log'),
                 os.path.join(os.path.dirname(processed_log_path), 'log.log'))

        # Copy the trajectory plot files
        traj_plot = glob.glob(os.path.join(os.path.dirname(log), '*.pdf'))

        for tp in traj_plot:
            copyfile(os.path.join(tp),
                     os.path.join(os.path.dirname(processed_log_path), os.path.basename(tp)))

def radar_chart(data_arr, 
                categories, 
                plot_dir,
                file_name,
                figsize=(6,6), 
                alpha=0.2, 
                ylim=(0,1.15), 
                yticks=[0.2,0.4,0.6,0.8,1.0],
                colors=None,
                fontsize=16):

    fig = plt.figure(figsize=figsize)

    N = len(categories)

    angles = [n/float(N)*2*np.pi for n in range(N)]
    angles += angles[:1]

    for row in data_arr:
        row_label = row[0]
        row_data = row[1:].astype(float).tolist() 
        row_data += row_data[:1]

        plt.polar(angles, 
                  row_data, 
                  marker='.', 
                  label=row_label, 
                  lw=3, 
                  markersize=18, 
                  color=colors[row_label], 
                  markeredgecolor=adjust_lightness(colors[row_label], 0.5),
                 )
        plt.fill(angles, row_data, color=colors[row_label], alpha=alpha)

    plt.xticks(angles[:-1], categories, fontsize=fontsize)
    plt.yticks(yticks, color='black', fontsize=fontsize-3)
    plt.ylim(ylim)

    plt.legend(ncol=1, fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.2, -0.15))

    plt.savefig(os.path.join(plot_dir,file_name), bbox_inches='tight')

def plot_last_model_perf(log_metrics, 
                         compared_method_codes,
                         task_names_map_file,
                         plot_dir, 
                         plot_name='final_model.pdf', 
                         ylims=[(-1.2,2.3),(-1.5,2.),(1.0,4.5)], 
                         fontsize=12, 
                         bbox_to_anchor=[0.5, 3.45], 
                         figsize=(12,12),
                         alpha=0.2):

    task_names_map = read_dict(task_names_map_file)

    # Remove .mat extension
    cl_sequence = list(task_names_map.values())

    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for i, compared_method_code in enumerate(compared_method_codes):

        for j, traj_metric in enumerate(['swept', 'frechet', 'dtw']):

            h = ax[j].plot(cl_sequence,
                           log_metrics[i]['mid'][traj_metric][-1], 
                           color=colors_models[compared_method_code],
                           lw=2,
                           marker='o',
                           markeredgecolor=adjust_lightness(colors_models[compared_method_code], 0.6),
                           label=compared_method_code)
            _ = ax[j].fill_between(cl_sequence, 
                                   log_metrics[i]['low'][traj_metric][-1], 
                                   log_metrics[i]['high'][traj_metric][-1], 
                                   color=colors_models[compared_method_code], 
                                   alpha=alpha)
            ax[j].set_ylim(ylims[j])

            # For prettier labels
            if traj_metric == 'swept':
                traj_metric = 'Swept Area'
            elif traj_metric == 'frechet':
                traj_metric = 'Frechet'
            elif traj_metric == 'dtw':
                traj_metric = 'DTW'

            ax[j].set_ylabel('$\log_{10}$'+f'\n({traj_metric})', fontsize=fontsize)
            if j==2:
                ax[j].set_xlabel('Task ID', fontsize=fontsize)

        for a in ax:
            a.set_xticks(np.arange(len(cl_sequence)))
            a.set_xticklabels(np.arange(len(cl_sequence)), fontsize=fontsize-2, rotation=90, ha='center')
            a.tick_params(axis='y', which='major', labelsize=fontsize-2)
            a.grid(True)

        handles, labels = ax[-1].get_legend_handles_labels()
        leg = ax[-1].legend(handles, labels, bbox_to_anchor=bbox_to_anchor, loc='center', ncol=len(compared_method_codes), fontsize=fontsize, columnspacing=1.1, handletextpad=0.2)
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
        plt.subplots_adjust(wspace=0.0, hspace=0.1, top=0.9)
        plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')

def final_model_box_violin(plot_dir, 
                           plot_name, 
                           dataset_name, 
                           select_task_ids, 
                           select_metric, 
                           ylims_last_model, 
                           compared_method_codes,
                           task_names_map_file,
                           log_dirs,
                           box_or_violin = 'box', 
                           width=0.6, 
                           fontsize=16):
    """
    
    1. Plot aggregate plot (1 box per method) and 1 metric
    select_task_ids=[]
    select_metric='dtw'

    2. Plot selected tasks for 1 metric
    select_task_ids=[0,4,8,12,...]
    select_metric='dtw'

    3. Plot all tasks for all metrics
    select_task_ids=None
    select_metric=None
    """

    x_label = 'Task ID'
    if dataset_name == 'helloworld':
        task_names_map = read_dict(task_names_map_file)
        cl_sequence = list(task_names_map.values())
        x_label = 'Task'

    metric_names = ['swept', 'frechet', 'dtw']
    metirc_labels = ['$\log_{10}$(Swept Area error)', '$\log_{10}$(Frechet error)', '$\log_{10}$(DTW error)']

    data = list()
    for method_code, log_dir in zip(compared_method_codes, log_dirs):

        metric_arr, log_metric_arr = get_arr(log_dir, acc_thresh_dict=None, agg_mode=None)

        for metric in log_metric_arr.keys():

            # Choose the results for the final model
            last_metric_arr = metric_arr[metric][-1]
            last_log_metric_arr = log_metric_arr[metric][-1]

            num_tasks, num_data_points = last_log_metric_arr.shape

            for task_id in range(num_tasks):
                for datapoint_id, datapoint in enumerate(last_log_metric_arr[task_id]):
                    data.append([method_code, task_id, metric, datapoint_id, datapoint])

    # Create dataframe
    data_df = pd.DataFrame(data, columns=['method_code', 'task_id', 'metric', 'datapoint_id', 'datapoint'])

    # Filter outliers
    # data_df = data_df[~data_df.groupby(['method_code', 'task_id', 'metric'])['datapoint'].apply(is_outlier)]

    colors = [colors_models[mc] for mc in compared_method_codes]

    if select_task_ids is None:
        # Create box plots for all tasks separately
        # All metrics
        fig, ax = plt.subplots(3,1,figsize=(20,10))
        if box_or_violin == 'box':
            for i, metric in enumerate(metric_names):
                plot_data = data_df[data_df['metric']==metric]
                ax[i] = sns.boxplot(y=plot_data['datapoint'], 
                                    hue=plot_data['method_code'], 
                                    x=plot_data['task_id'], 
                                    ax=ax[i], 
                                    palette=colors, 
                                    width=width, 
                                    showfliers=True,
                                    linewidth=0.5,
                                    fliersize=0.5)
        elif box_or_violin == 'violin':
            for i, metric in enumerate(metric_names):
                plot_data = data_df[data_df['metric']==metric]
                ax[i] = sns.violinplot(y=plot_data['datapoint'], 
                                        hue=plot_data['method_code'], 
                                        x=plot_data['task_id'], 
                                        ax=ax[i], 
                                        palette=colors, 
                                        width=width, 
                                        showfliers=True, 
                                        inner='quartile',
                                        cut=0)

        for i,a in enumerate(ax):
                a.set_ylim(ylims_last_model[i])
                a.grid(True, lw=0.2, color='#bbbbbb')
                a.set_ylabel(metirc_labels[i], fontsize=fontsize)
                a.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                plt.setp(a.get_yticklabels(), fontsize=fontsize-2)
                plt.setp(a.get_xticklabels(), fontsize=fontsize-2)

                if dataset_name == 'helloworld':
                    a.set_xticklabels(cl_sequence)

                if i<2:
                    a.get_legend().remove()
                    a.set_xlabel('')
                else:
                    a.set_xlabel(x_label, fontsize=fontsize)
                    a.legend(bbox_to_anchor=(0.45, -0.5), loc='lower center', ncol=6, borderaxespad=0., fontsize=fontsize)

        #fig.text(0.08, 0.52, 'Error metrics (lower is better)', va='center', rotation='vertical', fontsize=fontsize)
        fig.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')

    elif isinstance(select_task_ids, list) and len(select_task_ids)>0 and select_metric is not None:
        # Create box plots of selected tasks
        # For one metric
        select_metric_idx = metric_names.index(select_metric)
        metric_names = [metric_names[select_metric_idx]]

        fig, ax = plt.subplots(1,1,figsize=(15,1.75))
        ax.set_axisbelow(True)
        ax = [ax]

        if box_or_violin == 'box':
            for i, metric in enumerate(metric_names):
                data_df = data_df[data_df.task_id.isin(select_task_ids)] 
                plot_data = data_df[data_df['metric']==metric]
                ax[i] = sns.boxplot(y=plot_data['datapoint'], 
                                    hue=plot_data['method_code'], 
                                    x=plot_data['task_id'], 
                                    ax=ax[i], 
                                    palette=colors, 
                                    width=width, 
                                    showfliers=True,
                                    linewidth=1.0,
                                    fliersize=0.8)
        elif box_or_violin == 'violin':
            for i, metric in enumerate(metric_names):
                plot_data = data_df[data_df['metric']==metric]
                ax[i] = sns.violinplot(y=plot_data['datapoint'], 
                                        hue=plot_data['method_code'], 
                                        x=plot_data['task_id'], 
                                        ax=ax[i], 
                                        palette=colors, 
                                        width=width, 
                                        showfliers=True, 
                                        inner='quartile',
                                        cut=0)

        for i,a in enumerate(ax):
                a.set_ylim(ylims_last_model[select_metric_idx])
                a.grid(True, lw=0.2, color='#bbbbbb')
                a.set_ylabel(metirc_labels[select_metric_idx], fontsize=fontsize)
                a.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                plt.setp(a.get_yticklabels(), fontsize=fontsize-2)
                plt.setp(a.get_xticklabels(), fontsize=fontsize-2)

                if dataset_name == 'helloworld':
                    a.set_xticklabels([cl_sequence[k] for k in select_task_ids])

                a.set_xlabel(x_label, fontsize=fontsize)
                lgnd = a.legend(bbox_to_anchor=(0.5, 0.95), loc='lower center', ncol=6, borderaxespad=0., fontsize=fontsize-2, facecolor='white', framealpha=1)
                #lgnd = a.legend(bbox_to_anchor=(-0.05, -0.4), loc='lower left', ncol=6, borderaxespad=0., fontsize=fontsize-2)
                for handle in lgnd.legendHandles:
                    handle.width = .5

        #fig.text(0.08, 0.5, 'Lower is better', va='center', rotation='vertical', fontsize=fontsize-2)
        fig.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')

    elif isinstance(select_task_ids, list) and len(select_task_ids)==0 and select_metric is None:
        # Create box plots of all tasks together
        # For all metrics
        fig, ax = plt.subplots(3,1,figsize=(3,6), sharex=True)
        if box_or_violin == 'box':
            for i, metric in enumerate(metric_names):
                plot_data = data_df[data_df['metric']==metric]
                ax[i] = sns.boxplot(y=plot_data['datapoint'], 
                                    x=plot_data['method_code'], 
                                    ax=ax[i], 
                                    palette=colors, 
                                    width=width, 
                                    showfliers=True)
        elif box_or_violin == 'violin':
            for i, metric in enumerate(metric_names):
                plot_data = data_df[data_df['metric']==metric]
                ax[i] = sns.violinplot(y=plot_data['datapoint'], 
                                    x=plot_data['method_code'], 
                                    ax=ax[i], 
                                    palette=colors, 
                                    width=width, 
                                    showfliers=True, 
                                    inner='quartile',
                                    cut=0)

            for i,a in enumerate(ax):
                a.set_ylim(ylims_last_model[i])
                a.grid(True, lw=0.2, color='#bbbbbb')
                a.set_ylabel(metirc_labels[i], fontsize=fontsize)
                a.set_xlabel('')
                a.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                plt.setp(a.get_yticklabels(), fontsize=fontsize-2)
                plt.setp(a.get_xticklabels(), fontsize=fontsize-2)
                a.set_xlabel(x_label, fontsize=fontsize)

            #fig.text(0.03, 0.52, 'Error metrics (lower is better)', va='center', rotation='vertical', fontsize=fontsize)
            fig.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')

    elif isinstance(select_task_ids, list) and len(select_task_ids)==0 and select_metric is not None:
        # Create box plots of all tasks together
        # For one metric
        select_metric_idx = metric_names.index(select_metric)
        metric_names = [metric_names[select_metric_idx]]
        metirc_labels = [metirc_labels[select_metric_idx]]

        fig, ax = plt.subplots(1,1,figsize=(4,2.))
        ax.set_axisbelow(True)
        ax = [ax]
        
        if box_or_violin == 'box':
            for i, metric in enumerate(metric_names):
                plot_data = data_df[data_df['metric']==metric]
                ax[i] = sns.boxplot(y=plot_data['datapoint'], 
                                    x=plot_data['method_code'], 
                                    ax=ax[i], 
                                    palette=colors, 
                                    width=width, 
                                    showfliers=True,
                                    fliersize=3.0)
        elif box_or_violin == 'violin':
            for i, metric in enumerate(metric_names):
                plot_data = data_df[data_df['metric']==metric]
                ax[i] = sns.violinplot(y=plot_data['datapoint'], 
                                    x=plot_data['method_code'], 
                                    ax=ax[i], 
                                    palette=colors, 
                                    width=width, 
                                    showfliers=True, 
                                    inner='quartile',
                                    cut=0)

        for i,a in enumerate(ax):
            a.set_ylim(ylims_last_model[select_metric_idx])
            a.grid(True, lw=0.2, color='#bbbbbb')
            a.set_ylabel(metirc_labels[i], fontsize=fontsize)
            a.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.setp(a.get_yticklabels(), fontsize=fontsize-2)
            plt.setp(a.get_xticklabels(), fontsize=fontsize-2)
            a.set_xlabel('Method', fontsize=fontsize)

            if dataset_name == 'helloworld':
                a.set_xticklabels(compared_method_codes)

        #fig.text(0.04, 0.5, 'Lower is better', va='center', rotation='vertical', fontsize=fontsize-2)
        fig.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight')

def final_model_size(method_codes, num_total_tasks, log_dirs):

    method_logs = dict()
    method_sizes = dict()
    for method_name, log_dir in zip(method_codes, log_dirs):
        method_logs[method_name] = log_dir

    biggest_size = 0.0
    for method_code, multiseed_logdir in method_logs.items():
        total_size, task_emb_size, theta_size, buffer_size = compute_parameter_size(model_type=method_code, 
                                                                                    multiseed_logdir=multiseed_logdir, 
                                                                                    task_id=num_total_tasks-1, 
                                                                                    last_task_id=num_total_tasks-1,
                                                                                    verbose=False,
                                                                                    split_size=True)

        method_sizes[method_code] = total_size

        if total_size > biggest_size:
            biggest_size = total_size

    for method_code, method_size in method_sizes.items():
        method_sizes[method_code] = 1.0 - (float(method_size)/float(biggest_size))
        
    return method_sizes

def create_cl_table(cl_metrics, 
                    compared_method_codes, 
                    compared_method_names, 
                    log_dirs, 
                    nw_info, 
                    plot_dir, 
                    tex_file,
                    traj_metric='dtw'):

    # Compute the time efficiency CL metric
    tes = list()
    for log in log_dirs:
        te, _, _ ,_ =  compute_time_efficiency(log)
        tes.append(te)

    # Compute the FS metric
    models_fs = final_model_size(method_codes=compared_method_codes, 
                                 num_total_tasks=nw_info['total_tasks'], 
                                 log_dirs=log_dirs)

    cl_metrics_np = list()
    for i, method_name in enumerate(compared_method_names):
        method_code = compared_method_codes[i]
        acc = cl_metrics[i]['mid'][traj_metric]['acc']
        #bwt_plus = cl_metrics[i]['mid'][traj_metric]['bwt_plus']
        rem = cl_metrics[i]['mid'][traj_metric]['rem']
        ms = model_size_efficiency(model_type=method_code, 
                                   multiseed_logdir=log_dirs[i], 
                                   total_num_tasks=nw_info['total_tasks'])
        te = tes[i]
        fs = models_fs[compared_method_codes[i]]
        #cl_metrics_np.append([method_code, acc, bwt_plus, rem, ms, te])
        cl_metrics_np.append([method_code, acc, rem, ms, te, fs])

    cl_metrics_np = np.array(cl_metrics_np)

    #cl_metrics_df = pd.DataFrame(cl_metrics_np, columns=['METHOD', 'ACC', 'BWT$^+$', 'REM', 'MS', 'TE'])
    cl_metrics_df = pd.DataFrame(cl_metrics_np, columns=['METHOD', 'ACC', 'REM', 'MS', 'TE', 'FS'])

    # Convert numbers to float
    cl_metrics_df[['ACC', 'REM', 'MS', 'TE', 'FS']] = cl_metrics_df[['ACC', 'REM', 'MS', 'TE', 'FS']].apply(pd.to_numeric)

    cl_metrics_df['CL$_{score}$'] = cl_metrics_df.mean(numeric_only=True, axis=1)
    cl_metrics_df['CL$_{stability}'] = 1.0 - cl_metrics_df.std(numeric_only=True, axis=1)

    if tex_file is not None:
        with open(os.path.join(plot_dir, tex_file), 'w') as texfile:
            texfile.write(cl_metrics_df.to_latex(escape=False, index=False, float_format=lambda x: '%.4f' % x))

    return cl_metrics_df, cl_metrics_np

def copy_trajectory_plots(log_dirs, plot_dir, selected_seed_for_traj_plots, verbose=False): 
    copied_files = list()
    for log_dir in log_dirs:
        traj_plots = get_eval_files(log_dir, eval_file_name='plot_trajectories*.pdf')
        for tp in traj_plots:
            if f'seed{selected_seed_for_traj_plots}' in tp:
                filename = os.path.basename(tp)
                copyfile(tp, os.path.join(plot_dir, filename))
                copied_files.append(os.path.join(plot_dir, filename))
                if verbose:
                    print(f'Copied {tp} to {os.path.join(plot_dir, filename)}')

    return copied_files

def create_paramsize_plot(log_dirs,
                          method_codes,
                          task_names_map_file,
                          last_task_id,
                          plot_dir,
                          plot_file_1,
                          plot_file_2,
                          figsize_1=(10,4),
                          figsize_2=(10,10),
                          fontsize=12,
                          format_in_million=True):

    task_names_map = read_dict(task_names_map_file)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize_1)

    method_logs = dict()
    for method_name, log_dir in zip(method_codes, log_dirs):
        method_logs[method_name] = log_dir

    table = list()

    plot_data = dict()

    for methods_code, multiseed_logdir in method_logs.items():
        plot_data[methods_code] = list()
        for task_id in range(last_task_id+1):
            total_size, task_emb_size, theta_size, buffer_size = compute_parameter_size(model_type=methods_code, 
                                                                                        multiseed_logdir=multiseed_logdir, 
                                                                                        task_id=task_id, 
                                                                                        last_task_id=last_task_id,
                                                                                        verbose=False,
                                                                                        split_size=True)
            row = [methods_code, task_id, task_emb_size, theta_size, buffer_size, total_size]
            row = row[0:2] + [c/(1.0e6 if format_in_million else 1) for c in row[2:]]
            table.append(row)
            plot_data[methods_code].append(total_size/1.0e6 if format_in_million else 1)


    df = pd.DataFrame(table, columns=['methods_code', 'task_id', 'task_emb_size', 'theta_size', 'buffer_size', 'total_size'])

    df_total_size = df[['methods_code', 'task_id', 'total_size']]
    df_total_size = df_total_size.rename(columns={'methods_code': 'Methods', 'task_id': 'Task ID', 'total_size': 'Total size'})

    
    for k,v in plot_data.items():
        ax.plot (v,  marker= 'o', markersize=8, lw=2.5, color=colors_models[k], markeredgecolor=adjust_lightness(colors_models[k], amount=0.5), label=k)

    #ax = sns.lineplot(data=df_total_size, x="Task ID", y="Total size", hue='Methods', palette=colors_models, marker= 'o', markersize=8, lw=2.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:], fontsize=fontsize, ncol=2)

    ax.set_ylabel('Parameters ' + r'($\times 10^6$)', fontsize=fontsize)
    ax.set_xlabel('Task ID', fontsize=fontsize)
    ax.set_xticks(np.arange(0,last_task_id+1))
    ax.set_xticklabels(np.arange(0,last_task_id+1), rotation=0, ha='center', fontsize=fontsize-1)
    if len(list(task_names_map.values()))>7:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    else:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        
    plt.setp(ax.get_xticklabels(), fontsize=fontsize-1)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize-1)
    #ax.set_yticklabels([0,0,10,20,30,40,50], fontsize=fontsize-1)

    ax.grid(True, lw=0.2, color='#bbbbbb')

    plt.savefig(os.path.join(plot_dir, plot_file_1), bbox_inches='tight')

    # Create a grouped bar plot

    # fig, ax = plt.subplots(len(method_logs.keys()), 1, figsize=figsize_2, sharex=True, sharey=True)

    # df['others'] = df['total_size'] - (df['task_emb_size'] + df['theta_size'] + df['buffer_size'])

    # for i, method_code in enumerate(method_codes):
    #     data=df.loc[df['methods_code'] == method_code][['task_emb_size', 'theta_size','buffer_size', 'others']]

    #     ax[i] = data.plot.bar(stacked=True, ax=ax[i], subplots=False, legend=False, color=['r','#469990','#aaffc3','#ffd8b1'], edgecolor='#555555')
    #     ax[i].grid(True)

    # for i,a in enumerate(ax):
    #     a.set_ylabel(r'Parameters ($\times 10^6$)')
    #     a.set_yticks(np.arange(0,50,10))
    #     a.set_title(method_codes[i])

    # ax[-1].set_xlabel('Tasks', fontsize=fontsize)
    # ax[-1].set_xticks(np.arange(0,last_task_id+1))
    # ax[-1].set_xticklabels(task_names_map.values(), rotation=90, ha='center', fontsize=fontsize-2)

    # ax[-1].legend(labels=['Task embed vec', 'Trainable', 'Buffer', 'Non-trainable'], fontsize=fontsize-2)

    # plt.savefig(os.path.join(plot_dir, plot_file_2), bbox_inches='tight')
    
def hyperparameters_to_latex(plot_dir, tex_file, original_log_dirs, method_names):

    # Important hyperparameters to write to latex table
    hyperparams_names_map = {
                             'method'       : 'Method',
                             'num_iter'     : 'Training iterations',
                             'tsub'         : 'Training segments',
                             'lr'           : 'Learning rate',
                             'tnet_dim'     : 'Data dimension',
                             'tnet_arch'    : 'NODE hidden layers',
                             'tnet_act'     : 'NODE activation',
                             'hnet_arch'    : 'Hypernet hidden layers',
                             'task_emb_dim' : 'Task emb. vector dim',
                             'chunk_emb_dim': 'Chunk emb. vector dim',
                             'chunk_dim'    : 'Chunk dim',
                             'explicit_time': 'NODE time input',
                             'beta'         : 'Hypernet regu. strength',
                             'mas_lambda'   : 'MAS regu. strength',
                             'si_c'         : 'SI regu. strength',
                             'si_epsilon'   : 'SI damping parameter',
                             }

    # Save hyperparameters used in experiments
    filtered_command_args_list = list()
    for i, log_dir in enumerate(original_log_dirs):
        command_args = dict()
        command_args_files = glob.glob(os.path.join(log_dir,'*','commandline_args.json'))
        for command_file in command_args_files:
            command_args_new = read_dict(command_file)
            # Merge all command args from different seeds of a method
            command_args = merge(command_args, command_args_new)
        
        # Add a field for the method
        command_args['method'] = method_names[i]

        # Remove args which are not important
        filtered_command_args = dict()
        for k,v in hyperparams_names_map.items():
            if k in command_args.keys():
                filtered_command_args[v] = command_args[k]

        filtered_command_args_list.append(filtered_command_args)

    # Create a dataframe of hyperparameters
    hyperparameter_df = pd.DataFrame(filtered_command_args_list)
    hyperparameter_df = hyperparameter_df.replace(np.nan, '-') 

    # Transpose the dataframe
    hyperparameter_df = hyperparameter_df.set_index('Method').T.rename_axis('Hyperparameters').reset_index()

    # Create a latex table
    hyperparameter_latex_file = os.path.join(plot_dir, tex_file)
    with open(hyperparameter_latex_file, 'w') as texfile:
        texfile.write(hyperparameter_df.to_latex(escape=False, index=False))

    print(f'Created hyperparameter latex file {hyperparameter_latex_file}')

def create_all_plots(experiment_name,
                     dataset_name,
                     time_input,
                     task_names_map_file,
                     acc_thresh_dict,
                     num_iters,
                     total_tasks,
                     selected_seed_for_traj_plots=400,
                     y_lims_per_task=[(0.5,5.0),(-0.5,3.5),(2.0,6.0)],
                     ylims_last_model=[(0.5,4.0),(-0.5,2.5),(2.0,5.0)],
                     final_model_select_task_ids=[],
                     final_model_select_metric='dtw',
                     figsize_per_task=(20,9),
                     figsize_last_task=(10,10),
                     fontsize_per_task=18,
                     fontsize_last_task=18,
                     fontsize_spider=18,
                     show_traj=False):

    name = f'{experiment_name}_{dataset_name}_{time_input}'
    plot_dir = f'plots/{name}'
    base_log_dir = f'logs_{experiment_name}/{dataset_name}_{time_input}'

    print(f'log_dir={base_log_dir}')
    print(f'plot_dir={plot_dir}')

    # Create plot_dir if it does not exist
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    # Find log dirs for methods
    log_prefixes = ['tr_node',
                    'tr_ft_node',
                    'tr_si_node',
                    'tr_mas_node',
                    'tr_chn_node',
                    'tr_hn_node'
                    ]

    # Remove any dirs with the _processed suffix
    processed_dirs = glob.glob(os.path.join(base_log_dir,'*_processed'))
    for p_dir in processed_dirs:
        shutil.rmtree(p_dir)

    log_dirs = list()
    for log_prefix in log_prefixes:
        log_dir = glob.glob(os.path.join(base_log_dir,f'{log_prefix}*'))[0]
        log_dirs.append(log_dir)


    # Process the logs for the single NODE per task method
    process_upperbaseline_eval(log_multiseed_dir=log_dirs[0], 
                               processed_log_dir=f'{log_dirs[0]}_processed')

    original_log_dirs = deepcopy(log_dirs)
    log_dirs[0] = f'{log_dirs[0]}_processed'

    # Check if all log files are present
    check = {1: {'prefix': log_prefixes, 
                 'suffix': []},
             2: {'prefix': [], 
                 'suffix': ['seed200', 'seed400', 'seed800', 'seed1000', 'seed1200']},
             3: {'prefix': ['models', 'eval_results.json', 'log.log', 'plot_trajectories*.pdf', 'commandline_args.json'], 
                 'suffix': []},
            }
    log_check_status = check_dirtree(base_log_dir, check)
    assert log_check_status == 0, f'log_check_status={log_check_status}, logs are not complete!'

    compared_method_names = ['Single NODE per task', 
                             'Finetuning', 
                             'Synaptic Intelligence', 
                             'Memory Aware Synapses', 
                             'Chunked Hypernetworks',
                             'Hypernetworks'
                             ]

    compared_method_codes = ['SG', 
                             'FT', 
                             'SI', 
                             'MAS',
                             'CHN', 
                             'HN'
                             ]

    # Create a latex file with a table of hyperparameters
    hyperparameters_to_latex(plot_dir=plot_dir, 
                             tex_file='hyperparamaters.tex', 
                             original_log_dirs=original_log_dirs, 
                             method_names=compared_method_names)



    # Plot per-task performance and compute CL metrics (ACC, BWT+, REM)
    metrics, log_metrics, cl_metrics, seeds, cl_sequence = plot_split(log_dirs=log_dirs, 
                                                                acc_thresh_dict=acc_thresh_dict, 
                                                                plot_dir=plot_dir, 
                                                                plot_name='per_task.pdf',
                                                                compared_method_codes=compared_method_codes,
                                                                task_names_map_file=task_names_map_file,
                                                                fontsize=fontsize_per_task,
                                                                y_lims=y_lims_per_task, 
                                                                bbox_to_anchor=[-1.8, 3.5],
                                                                ax_titles=compared_method_codes)

    # Create a dataframe for CL metrics
    cl_metrics_df, cl_metrics_np = create_cl_table(cl_metrics, 
                                                   compared_method_codes, 
                                                   compared_method_names, 
                                                   log_dirs, 
                                                   {'total_tasks':total_tasks},
                                                   plot_dir, 
                                                   tex_file='cl_metrics.tex',
                                                   traj_metric='dtw')

    print(cl_metrics_df.to_latex(escape=False, index=False, float_format=lambda x: '%.4f' % x))

    # Create radar chart
    # radar_chart(cl_metrics_np, 
    #             categories=['ACC','REM','MS','TE','FS'], 
    #             plot_dir=plot_dir, 
    #             file_name='cl_metrics_spider.pdf', 
    #             fontsize=fontsize_spider,
    #             colors=colors_models)

    # Plot last model performance
    # plot_last_model_perf(log_metrics=log_metrics, 
    #                     compared_method_codes=compared_method_codes,
    #                     task_names_map_file=task_names_map_file,
    #                     plot_dir=plot_dir,
    #                     plot_name='final_model.pdf',
    #                     ylims=ylims_last_model,
    #                     fontsize=fontsize_last_task,
    #                     figsize=figsize_last_task)

    # Box plot of final model - all tasks together and for selected metric
    final_model_box_violin(plot_dir=plot_dir, 
                           plot_name=f'final_model_all_tasks_{final_model_select_metric}.pdf', 
                           dataset_name=dataset_name, 
                           select_task_ids=[], 
                           select_metric=final_model_select_metric, 
                           ylims_last_model=ylims_last_model, 
                           compared_method_codes=compared_method_codes,
                           task_names_map_file=task_names_map_file,
                           log_dirs=log_dirs,
                           box_or_violin = 'box', 
                           width=0.6, 
                           fontsize=16)

    # Box plot of final model - selected tasks individually and for selected metric
    final_model_box_violin(plot_dir=plot_dir, 
                           plot_name=f'final_model_selected_tasks_{final_model_select_metric}.pdf', 
                           dataset_name=dataset_name, 
                           select_task_ids=final_model_select_task_ids, 
                           select_metric=final_model_select_metric, 
                           ylims_last_model=ylims_last_model, 
                           compared_method_codes=compared_method_codes,
                           task_names_map_file=task_names_map_file,
                           log_dirs=log_dirs,
                           box_or_violin = 'box', 
                           width=0.6, 
                           fontsize=13)

    # Box plot of final model - all tasks individually and for all metrics
    final_model_box_violin(plot_dir=plot_dir, 
                           plot_name=f'final_model_all_tasks_all_metrics.pdf', 
                           dataset_name=dataset_name, 
                           select_task_ids=None, 
                           select_metric=None, 
                           ylims_last_model=ylims_last_model, 
                           compared_method_codes=compared_method_codes,
                           task_names_map_file=task_names_map_file,
                           log_dirs=log_dirs,
                           box_or_violin = 'box', 
                           width=0.6, 
                           fontsize=16)

    # Plot parameter sizes
    create_paramsize_plot(log_dirs=original_log_dirs,
                          method_codes=compared_method_codes,
                          task_names_map_file=task_names_map_file,
                          last_task_id=total_tasks-1,
                          plot_dir=plot_dir,
                          plot_file_1='param_lineplot.pdf',
                          plot_file_2='param_barplot.pdf',
                          figsize_1=(10,3),
                          figsize_2=(10,10),
                          fontsize=17,
                          format_in_million=True)

    # Copy predicted patterns of final model
    copied_files = copy_trajectory_plots(log_dirs, plot_dir, selected_seed_for_traj_plots=selected_seed_for_traj_plots)

    return copied_files, log_dirs
    
            