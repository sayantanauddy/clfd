import numpy as np
from ..metrics.cl_metrics import get_arr_multiseed

colors = ['#000000', '#696969', '#556b2f', '#8b4513', 
          '#483d8b', '#008000', '#000080', '#9acd32', 
          '#20b2aa', '#8b008b', '#ff0000', '#ffa500', 
          '#aaaa00', '#7cfc00', '#deb887', '#8a2be2',
          '#00ff7f', '#dc143c', '#00bfff', '#0000ff', 
          '#d8bfd8', '#ff7f50', '#ff00ff', '#db7093', 
          '#ff1493', '#ee82ee']

def line_plot_multiseed(tril_arr_mid, tril_arr_err, cl_sequence, ax, xlabel, ylabel, title, fontsize=12):
    num_tasks = tril_arr_mid.shape[0]

    handles = list()
    labels = list()
    for i in range(0, num_tasks):
        task_name = list(cl_sequence)[i]
        x = np.arange(i,num_tasks)
        y_mid = tril_arr_mid[i:,i]
        y_err = tril_arr_err[i:,i]
        h = ax.plot(x, y_mid, marker="o", markersize=5, markeredgecolor='black', markeredgewidth=0.5,lw=1.5, ls="-", label=task_name, color=colors[i])
        _ = ax.fill_between(x, y_mid-y_err, y_mid+y_err, color=colors[i], alpha=0.15)
        handles.append(h)
        labels.append(task_name)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks(np.arange(num_tasks))
    ax.set_xticklabels(cl_sequence, fontsize=fontsize, rotation=90, ha='center')

    if title is not None:
        ax.set_title(title)
    ax.grid(True)

    return handles, labels

def per_task_performance_multiseed(log_multiseed_dir, axes, y_lims, task_names_map, title, acc_thresh_dict, fontsize=12):
    
    mid, log10_mid, cl_mid, err, log10_err, cl_err, seeds, description = get_arr_multiseed(log_multiseed_dir, acc_thresh_dict)
    
    # Remove .mat extension
    cl_sequence = task_names_map.values()
    
    plot_idx = 0
    for metric_name, metric_arr_ in log10_mid.items():
        metric_arr_err_ = log10_err[metric_name]
        if plot_idx != 2:
            xlabel = None
        else:
            xlabel='tasks'
        if plot_idx == 0:
            title_= title
        else:
            title_ = None
        h,l = line_plot_multiseed(metric_arr_, metric_arr_err_, cl_sequence, axes[plot_idx], xlabel=xlabel, ylabel=f'log({metric_name})', title=title_, fontsize=fontsize)   
        plot_idx += 1

    for i, ax in enumerate(axes):
        ax.set_ylim(y_lims[i])

    handles, labels = axes[-1].get_legend_handles_labels()
    return handles, labels, cl_mid, cl_err, seeds, description

