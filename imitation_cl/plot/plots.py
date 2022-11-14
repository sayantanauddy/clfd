import numpy as np
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd
import os

pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None  # default='warn'

colors_models_node = {'FT': '#f58231', 'SI': '#911eb4', 'MAS': 'rosybrown',
                      'HN': '#e6194B', 'SG': '#1E90FF', 'iFlow': '#800000',
                      'CHN': '#3cb44b', 'REP': 'gray'}

# Similar colors as above but lighter
#color_models_lsddm = {'SG': '#0000FF', 'HN': '#FF0000', 'CHN': '#00FF00'}
color_models_lsddm = {'FT': '#f58231', 'SI': '#911eb4', 'MAS': 'rosybrown',
                      'HN': '#e6194B', 'SG': '#1E90FF', 'iFlow': '#800000',
                      'CHN': '#3cb44b', 'REP': 'gray'}

lw_models = {'FT': 2, 'SI': 2, 'MAS': 2,
             'HN': 2, 'SG': 2, 'iFlow': 2,
             'CHN': 2, 'REP': 2}

color_palette = {'FT_NODE': '#f58231', 'SI_NODE': '#911eb4', 'MAS_NODE': 'rosybrown',
                 'HN_NODE': '#e6194B', 'SG_NODE': '#1E90FF', 'iFlow': '#800000',
                 'CHN_NODE': '#3cb44b', 'REP_NODE': 'gray',
                 'FT_LSDDM': '#f58231', 'SI_LSDDM': '#911eb4', 'MAS_LSDDM': 'rosybrown',
                 'SG_LSDDM': '#1E90FF', 'HN_LSDDM': '#e6194B', 'CHN_LSDDM': '#3cb44b',
                 'REP_LSDDM': 'gray',
                 'NODE_0': 'mediumslateblue', 'NODE_1': 'cornflowerblue', 'IFLOW_0': 'teal'}


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


def get_cl_acc_arr(database_df, dataset, cl_type, traj_type, data_dim, explicit_time, num_iters, num_tasks, threshold, metric_name):

    # Select the relevant rows
    query = f'(dataset=="{dataset}") and (cl_type=="{cl_type}") and (traj_type=="{traj_type}") and (data_dim=={data_dim}) and (num_iters=={num_iters}) and (explicit_time=={explicit_time})'
    selection_df = database_df.query(query)

    # Find the unique seeds
    seeds = np.unique(selection_df['seed'].tolist())

    # Initialize the acc matrix
    acc_arr = np.zeros((num_tasks, num_tasks, len(seeds)))
    for train_task_id in range(num_tasks):
        for eval_task_id in range(train_task_id+1):
            for i,seed in enumerate(seeds):
                metric = selection_df.query(f'(train_task_id=={train_task_id}) and (eval_task_id=={eval_task_id}) and (seed=={seed})')[metric_name]
                metric = metric.to_numpy()
                acc = (metric < threshold).sum()/len(metric)
                acc_arr[train_task_id, eval_task_id, i] = acc

    return acc_arr, seeds


def get_te_ms_sss_fs(database_df, seed_idx, dataset, cl_type, traj_type, data_dim, explicit_time, num_iters, num_tasks, threshold, metric_name):

    # Select the relevant rows
    query = f'(dataset=="{dataset}") and (cl_type=="{cl_type}") and (traj_type=="{traj_type}") and (data_dim=={data_dim}) and (num_iters=={num_iters}) and (explicit_time=={explicit_time}) and (obs_id==0)'
    selection_df = database_df.query(query).query('train_task_id == eval_task_id')

    # Find the unique seeds
    seeds = np.unique(selection_df['seed'].tolist())

    # Rows for the current seed
    selection_df = selection_df.query(f'seed == {seeds[seed_idx]}')

    num_tasks = np.unique(selection_df['num_tasks'].tolist())
    assert len(num_tasks) == 1
    num_tasks = num_tasks[0]

    task_ids = selection_df['train_task_id'].tolist()
    train_times = selection_df['train_time'].tolist()
    model_param_cnt = selection_df['model_param_cnt'].tolist()

    # Compute time efficiency
    summed_time_ratio = 0.0
    for task_id, time in zip(task_ids, train_times):
        summed_time_ratio += train_times[0]/time
    summed_time_ratio /= num_tasks
    time_efficiency = min(1.0, summed_time_ratio)

    # Compute model size efficiency
    ms = 0
    for task_id in range(num_tasks):
        ms += model_param_cnt[0]/model_param_cnt[task_id]
    ms /= num_tasks
    ms = min(1.0, ms)

    # Compute final model size
    max_param_query = f'(dataset=="{dataset}") and (traj_type=="{traj_type}") and (data_dim=={data_dim}) and (num_iters=={num_iters}) and (explicit_time=={explicit_time}) and (obs_id==0)'
    max_param_df = database_df.query(max_param_query).query('train_task_id == eval_task_id')
    max_param_df = max_param_df.query(f'seed == {seeds[seed_idx]}')
    max_param_df = max_param_df.query(f'train_task_id == {task_id}')
    max_model_param_cnt = max(max_param_df['model_param_cnt'].tolist())
    fs = 1.0 - (model_param_cnt[task_id]/max_model_param_cnt)

    # Samples storage size efficiency
    if cl_type == 'REP':
        sss = 1.0 - min(1.0, ((num_tasks * (num_tasks+1))/(2*num_tasks))/num_tasks)
    else:
        sss = 1.0

    return time_efficiency, ms, sss, fs


def create_cl_df(database_df,cl_queries):
    cl_data = list()
    for cl_query in cl_queries:
        # print(f'Processing {cl_query}')
        acc_arr, seeds = get_cl_acc_arr(database_df, **cl_query)
        num_seeds = acc_arr.shape[-1]
        for seed_idx in range(num_seeds):

            # ACC, BWT, BWT_PLUS, REM
            acc, bwt, bwt_plus, rem = get_cl_metrics(acc_arr[:,:,seed_idx])

            # TE, MS, SSS
            time_efficiency, ms, sss, fs = get_te_ms_sss_fs(database_df, seed_idx, **cl_query)


            row = dict(dataset=cl_query['dataset'],
                    cl_type=cl_query['cl_type'],
                    traj_type=cl_query['traj_type'],
                    data_dim=cl_query['data_dim'],
                    explicit_time=cl_query['explicit_time'],
                    threshold=cl_query['threshold'],
                    metric_name=cl_query['metric_name'],
                    seed=seeds[seed_idx],
                    acc=acc,
                    bwt=bwt,
                    bwt_plus=bwt_plus,
                    rem=rem,
                    ms=ms,  
                    te=time_efficiency, 
                    fs=fs,
                    sss=sss,
                    )
            cl_data.append(row)
    cl_df = pd.DataFrame(cl_data)
    return cl_df


def create_cl_table(cl_df, dataset, explicit_time):

    # METHOD ACC REM MS TE FS CLscore CLstability
    r_df = cl_df.query(f'(dataset=="{dataset}") and (explicit_time=={explicit_time})').groupby(['cl_type']).mean()
    data_dim = r_df['data_dim'].tolist()[0]
    #r_df = r_df[['acc', 'bwt_plus', 'rem', 'ms', 'te', 'fs', 'sss']]
    r_df = r_df[['acc', 'rem', 'ms', 'te', 'fs', 'sss']]
    r_df['cl_score'] = (r_df['acc'] + r_df['rem'] +r_df['ms'] + r_df['te'] + r_df['fs'] + r_df['sss'])/6
    r_df['cl_stability'] = 1.0 - r_df[['acc', 'rem', 'ms', 'te', 'fs', 'sss']].std(axis=1)

    # String name for saving
    table_str_name = f'latex_tbl_{dataset}_{data_dim}D_t{explicit_time}.tex'
    return r_df, table_str_name


def bold_extreme_values(data, data_max=-1, float_format=3):
    '''
    Source: https://blog.martisak.se/2021/04/10/publication_ready_tables/
    '''
    if data == data_max:
        fmt_str = '{:.'+str(float_format)+'f}'
        data = fmt_str.format(data)
        return "\\textbf{%s}" % data
    return data


def format_cl_table(r_df, 
                    table_str_name,
                    save_dir,
                    float_format=3, 
                    bold_headers=True,
                    col_names_display=['METHOD', 'ACC','REM','MS','TE','FS','SSS','CL$_{score}$','CL$_{stability}$'],
                    col_show_max=['acc','rem','ms','te','fs','sss','cl_score','cl_stability'],
                    sort_dict={'SG':0, 'FT':1, 'REP':2, 'SI':3, 'MAS':4, 'HN':5, 'CHN':6} 
                    ):

    # Number of decimal places
    r_df = r_df.round(float_format)
    format_str = f'{{:.{float_format}f}}'
    pd.set_option('display.float_format', format_str.format)

    # Bold format for max number in each column
    for k in col_show_max:
        r_df[k] = r_df[k].apply(lambda data: bold_extreme_values(data, data_max=r_df[k].max(), float_format=float_format))
    
    # All columns at same level
    r_df = r_df.reset_index(level='cl_type', col_level=0)

    # Sort rows
    r_df = r_df.sort_values(by=['cl_type'], key=lambda x: x.map(sort_dict))

    # Make column headers bold
    if bold_headers:
        col_names_display = ['\\textbf{'+h+'}' for h in col_names_display]

    # Assign new column names
    r_df.columns = col_names_display

    #latex_str = r_df.style.applymap_index(lambda v: "font-weight: bold;", axis="columns").to_latex(convert_css=True, index=False, escape=False)
    latex_str = r_df.to_latex(index=False, escape=False)

    # Save latex file
    save_path = os.path.join(save_dir, table_str_name)
    with open(save_path, "w") as latex_file:
        latex_file.write(latex_str)

    print(f'Table saved in {save_path}')
    print(latex_str)    


def wall_clock_table(database_df,
                     dataset, 
                     data_dim,
                     num_iters,
                     num_tasks,
                     selected_tasks,
                     save_dir, 
                     table_str_name,
                     sort_dict={'SG':0, 'FT':1, 'REP':2, 'SI':3, 'MAS':4, 'HN':5, 'CHN':6}
                     ):

    renamed_col_names = dict()
    renamed_col_names['cl_type'] = 'METHOD'
    for s in selected_tasks:
        renamed_col_names[s] = f'Task {s}'

    temp_df = database_df.query(f"(dataset=='{dataset}') and (data_dim=={data_dim}) and (traj_type=='NODE') and (explicit_time==1) and (num_iters=={num_iters}) and (num_tasks=={num_tasks}) and (obs_id==0) and (eval_task_id==0)")
    temp_df = temp_df.groupby(['cl_type', 'train_task_id']).agg({'train_time':np.median})
    temp_df = temp_df.pivot_table('train_time', ['cl_type'], 'train_task_id').reset_index()
    temp_df = temp_df[['cl_type', *selected_tasks]]
    temp_df = temp_df.sort_values(by=['cl_type'], key=lambda x: x.map(sort_dict))

    # Cast floats to int
    float_col = temp_df.select_dtypes(include=['float64'])
    for col in float_col.columns.values:
        temp_df[col] = temp_df[col].astype('int64')

    temp_df = temp_df.rename(columns=renamed_col_names)

    # Save latex file
    latex_str = temp_df.to_latex(index=False, escape=False)
    save_path = os.path.join(save_dir, table_str_name)
    with open(save_path, "w") as latex_file:
        latex_file.write(latex_str)

    print(f'Table saved in {save_path}')
    print(latex_str)    


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_cumulative_errors(database_df, ax, dataset, explicit_time, cl_type, traj_type, data_dim, metric_name, plot_log=True, label_type='combined', markersize=6.5):

    # Filter results for the dataset and explicit time
    filtered_database_df = database_df.loc[
        (database_df.dataset == dataset) &
        (database_df.explicit_time == explicit_time) &
        (database_df.data_dim == data_dim)
    ]

    filtered_database_df = filtered_database_df.reset_index()
    filtered_database_df.drop(['index'], axis=1, inplace=True)

    # Find the number of tasks
    num_tasks = np.unique(filtered_database_df.num_tasks.values)[0]

    # Create a dict to store metric values
    metric_list = list()

    for train_task_id in range(num_tasks):

        val = filtered_database_df.loc[
            (filtered_database_df.cl_type == cl_type) &
            (filtered_database_df.traj_type == traj_type) &
            (filtered_database_df.train_task_id == train_task_id) &
            (filtered_database_df.eval_task_id <= train_task_id)
        ][metric_name].values

        metric_list.append(val)

    # Create a dict to store medians and percentiles of metric values (each metric has a list containing arrays)
    metric_plot = list()
    metric_log10_plot = list()

    for val in metric_list:
        low = np.percentile(val, 25)
        mid = np.median(val)
        high = np.percentile(val, 75)
        metric_plot.append((low, mid, high))

        low_log10 = np.percentile(np.log10(val), 25)
        mid_log10 = np.median(np.log10(val))
        high_log10 = np.percentile(np.log10(val), 75)
        metric_log10_plot.append((low_log10, mid_log10, high_log10))

    color = colors_models_node[cl_type] if traj_type == 'NODE' else color_models_lsddm[cl_type]

    plot_kwargs = dict(color=color,
                       marker="o",
                       markersize=markersize,
                       markeredgecolor=adjust_lightness(color, 0.5),
                       markeredgewidth=0.5,
                       lw=lw_models[cl_type],
                       ls="-")

    plot_data = metric_log10_plot if plot_log else metric_plot

    ax.plot([m[1] for m in plot_data],
            label=f'{cl_type}_{traj_type}' if label_type=='combined' else f'{cl_type}',
            path_effects=[
                pe.Stroke(linewidth=lw_models[cl_type]*2, foreground=adjust_lightness(color, 0.8)), pe.Normal()],
            **plot_kwargs)

    x = np.arange(0, num_tasks)
    ax.fill_between(x, [m[0] for m in plot_data], [m[2]
                    for m in plot_data], color=color, alpha=0.15)


# All errors for selected_tasks
def plot_model_errors(a, database_df, dataset, explicit_time, data_dim, metric_name, plot_log=True, selected_tasks=None, traj_types=['NODE', 'LSDDM'],
                      cl_types='all'):

    order_map = {'SG': 0, 'REP': 1, 'FT': 2, 'SI': 3, 'MAS': 4, 'HN': 5, 'CHN': 6}

    # Filter results for the dataset and explicit time
    filtered_database_df = database_df.loc[
        (database_df.dataset == dataset) &
        (database_df.explicit_time == explicit_time) &
        (database_df.data_dim == data_dim) &
        (database_df.traj_type.isin(traj_types)) &
        (database_df.train_task_id.isin(selected_tasks))
        
    ]

    filtered_database_df = filtered_database_df.reset_index()
    filtered_database_df.drop(['index'], axis=1, inplace=True)

    if cl_types != 'all':
        filtered_database_df = filtered_database_df.loc[filtered_database_df.cl_type.isin(cl_types)]

    if plot_log:
        filtered_database_df['dtw'] = filtered_database_df['dtw'].map(lambda x: np.log10(x))

    # All tasks together
    order = list(np.unique(filtered_database_df[['cl_type', 'traj_type']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1).values))
    order.sort(key=lambda x: order_map[x.split('_')[0]])
    sns.boxplot(x=filtered_database_df[['cl_type', 'traj_type']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1), 
                y=metric_name, palette=color_palette, 
                data=filtered_database_df, 
                showfliers=False, 
                order=order,
                ax=a)


def plot_model_errors_old(a, database_df, dataset, explicit_times, data_dim, metric_name, plot_log=True, selected_tasks=None, traj_types=['NODE', 'LSDDM'],
                      cl_types='all'):

    order_map = {'IFLOW_0': 0, 'NODE_0': 1, 'NODE_1': 2}

    # Filter results for the dataset and explicit time
    filtered_database_df = database_df.loc[
        (database_df.dataset == dataset) &
        (database_df.explicit_time.isin(explicit_times)) &
        (database_df.data_dim == data_dim) &
        (database_df.traj_type.isin(traj_types)) &
        (database_df.train_task_id.isin(selected_tasks))
    ]

    filtered_database_df = filtered_database_df.reset_index()
    filtered_database_df.drop(['index'], axis=1, inplace=True)

    if cl_types != 'all':
        filtered_database_df = filtered_database_df.loc[filtered_database_df.cl_type.isin(cl_types)]

    if plot_log:
        filtered_database_df['dtw'] = filtered_database_df['dtw'].map(lambda x: np.log10(x))

    order = list(np.unique(filtered_database_df[['traj_type', 'explicit_time']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1).values))

    order.sort(key=lambda x: order_map[x])

    sns.boxplot(x=filtered_database_df[['traj_type', 'explicit_time']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1), 
                y=metric_name, palette=color_palette, 
                data=filtered_database_df, 
                showfliers=False, 
                order=order,
                ax=a)

def plot_iflow_errors(a, database_df, dataset, explicit_times, data_dim, metric_name, plot_log=True, selected_tasks=None, traj_types=['NODE', 'LSDDM'],
                      cl_types='all'):

    order_map = {'IFLOW_0': 0, 'NODE_0': 1, 'NODE_1': 2}

    # Filter results for the dataset and explicit time
    filtered_database_df = database_df.loc[
        (database_df.dataset == dataset) &
        (database_df.explicit_time.isin(explicit_times)) &
        (database_df.data_dim == data_dim) &
        (database_df.traj_type.isin(traj_types)) &
        (database_df.train_task_id.isin(selected_tasks))
    ]

    filtered_database_df = filtered_database_df.reset_index()
    filtered_database_df.drop(['index'], axis=1, inplace=True)

    if cl_types != 'all':
        filtered_database_df = filtered_database_df.loc[filtered_database_df.cl_type.isin(cl_types)]

    if plot_log:
        filtered_database_df['dtw'] = filtered_database_df['dtw'].map(lambda x: np.log10(x))

    order = list(np.unique(filtered_database_df[['traj_type', 'explicit_time']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1).values))

    order.sort(key=lambda x: order_map[x])

    sns.boxplot(x=filtered_database_df[['traj_type', 'explicit_time']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1), 
                y=metric_name, 
                palette=color_palette, 
                hue=filtered_database_df[['traj_type', 'explicit_time']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1),
                data=filtered_database_df, 
                showfliers=False, 
                order=order,
                ax=a,
                dodge=False)


# Qualitative plot - 1D plots fpr ground truth and predicted quaternions

import os
import matplotlib.pyplot as plt
import torch

from imitation_cl.model.hypernetwork import HyperNetwork, ChunkedHyperNetwork, TargetNetwork, str_to_ints, str_to_act, get_current_targets
from imitation_cl.model.node import NODE
from imitation_cl.data.robottasks import RobotTasksPosition, RobotTasksOrientation
from imitation_cl.train.utils import get_sequence

from imitation_cl.logging.utils import Dictobject, read_dict

def hn_position_orientation_predict(cl_type, save_dir, data_dir, seq_file, demo_id, train_task_id, eval_task_id, type='orientation'):

    args = Dictobject(read_dict(os.path.join(save_dir, 'commandline_args.json')))


    filenames = get_sequence(seq_file)

    device = torch.device('cpu')


    # Shapes of the target network parameters
    target_shapes = TargetNetwork.weight_shapes(n_in=args.tnet_dim+args.explicit_time, 
                                                n_out=args.tnet_dim, 
                                                hidden_layers=str_to_ints(args.tnet_arch), 
                                                use_bias=True)

    if cl_type=='hn':
        # Create the regular hypernetwork
        hnet = HyperNetwork(layers=str_to_ints(args.hnet_arch), 
                        te_dim=args.task_emb_dim, 
                        target_shapes=target_shapes,
                        dropout_rate=args.dropout,
                        device=device).to(device)
    elif cl_type=='chn':
        # Create the chunked hypernetwork
        hnet = ChunkedHyperNetwork(final_target_shapes=target_shapes,
                                    layers=str_to_ints(args.hnet_arch),
                                    chunk_dim=args.chunk_dim,
                                    te_dim=args.task_emb_dim,
                                    ce_dim=args.chunk_emb_dim,
                                    dropout_rate=args.dropout,
                                    device=device).to(device)
        
    # Create a target network without parameters
    # Parameters are supplied during the forward pass of the hypernetwork
    tnet = TargetNetwork(n_in=args.tnet_dim+args.explicit_time, 
                            n_out=args.tnet_dim, 
                            hidden_layers=str_to_ints(args.tnet_arch),
                            activation_fn=str_to_act(args.tnet_act), 
                            use_bias=True, 
                            no_weights=True,
                            init_weights=None, 
                            dropout_rate=-1,  # Dropout is only used for hnet 
                            use_batch_norm=False, 
                            bn_track_stats=False,
                            distill_bn_stats=False, 
                            out_fn=None,
                            device=device).to(device)

    # The NODE uses the target network as the RHS of its
    # differential equation
    # Apart from this, the NODE has no other trainable parameters
    node = NODE(tnet, explicit_time=args.explicit_time, method=args.int_method).to(device)

    hnet = torch.load(os.path.join(save_dir, 'models', f'hnet_{train_task_id}.pth'), map_location=device)
    tnet = torch.load(os.path.join(save_dir, 'models', f'tnet_{train_task_id}.pth'), map_location=device)
    node = torch.load(os.path.join(save_dir, 'models', f'node_{train_task_id}.pth'), map_location=device)

    hnet.device = device

    hnet.eval()
    tnet.eval()

    tnet = tnet.to(device)
    hnet = hnet.to(device)
    node = node.to(device)

    # Generate parameters of the target network for the current task
    weights = hnet.forward(eval_task_id)

    # Set the weights of the target network
    tnet.set_weights(weights)

    # Set the target network in the NODE
    node.set_target_network(tnet)
    node = node.float()
    node.eval()

    if type == 'orientation':
        data = RobotTasksOrientation(data_dir=data_dir, 
                                    datafile=filenames[eval_task_id], 
                                    device=device, 
                                    scale=100.0)
    elif type == 'position':
        data = RobotTasksPosition(data_dir=data_dir, datafile=filenames[eval_task_id], device=device)

    t = data.t[0].float()

    # The starting position 
    # (n,d-dimensional, where n is the num of demos and 
    # d is the dimension of each point)
    #y_start = torch.unsqueeze(dataset.pos[0,0], dim=0)
    # Use the translated trajectory (goal at origin)
    y_start = data.pos[:,0]
    y_start = y_start.float()

    # The entire demonstration trajectory
    y_all = data.pos.float()

    # Predicted trajectory
    y_hat = node(t, y_start) # forward simulation

    # Compute trajectory metrics
    y_all_np = y_all.cpu().detach().numpy()
    y_hat_np = y_hat.cpu().detach().numpy()

    # De-normalize the data before computing trajectories
    y_all_np = data.unnormalize(y_all_np)
    y_hat_np = data.unnormalize(y_hat_np)    

    # Prediction
    if type == 'orientation':
        q_hat_np = data.from_tangent_plane(y_hat_np)
    elif type == 'position':
        q_hat_np = y_hat_np

    # Ground truth
    if type == 'orientation':
        q_np = data.quat_data
    elif type == 'position':
        q_np = y_all_np

    return q_hat_np, q_np


def sg_position_orientation_predict(cl_type, save_dir, data_dir, seq_file, demo_id, train_task_id, eval_task_id, type='orientation'):

    args = Dictobject(read_dict(os.path.join(save_dir, 'commandline_args.json')))

    filenames = get_sequence(seq_file)

    device = torch.device('cpu')


    # Shapes of the target network parameters
    target_shapes = TargetNetwork.weight_shapes(n_in=args.tnet_dim+args.explicit_time, 
                                                n_out=args.tnet_dim, 
                                                hidden_layers=str_to_ints(args.tnet_arch), 
                                                use_bias=True)
        
    # Create a target network without parameters
    # Parameters are supplied during the forward pass of the hypernetwork
    tnet = TargetNetwork(n_in=args.tnet_dim+args.explicit_time, 
                            n_out=args.tnet_dim, 
                            hidden_layers=str_to_ints(args.tnet_arch),
                            activation_fn=str_to_act(args.tnet_act), 
                            use_bias=True, 
                            no_weights=True,
                            init_weights=None, 
                            dropout_rate=-1,  # Dropout is only used for hnet 
                            use_batch_norm=False, 
                            bn_track_stats=False,
                            distill_bn_stats=False, 
                            out_fn=None,
                            device=device).to(device)

    # The NODE uses the target network as the RHS of its
    # differential equation
    # Apart from this, the NODE has no other trainable parameters
    node = NODE(tnet, explicit_time=args.explicit_time, method=args.int_method).to(device)

    tnet = torch.load(os.path.join(save_dir, 'models', f'tnet_{eval_task_id}.pth'), map_location=device)
    node = torch.load(os.path.join(save_dir, 'models', f'node_{eval_task_id}.pth'), map_location=device)

    tnet.eval()

    tnet = tnet.to(device)
    node = node.to(device)

    # Set the target network in the NODE
    node.set_target_network(tnet)
    node = node.float()
    node.eval()

    if type == 'orientation':
        data = RobotTasksOrientation(data_dir=data_dir, 
                                    datafile=filenames[eval_task_id], 
                                    device=device, 
                                    scale=100.0)
    elif type == 'position':
        data = RobotTasksPosition(data_dir=data_dir, datafile=filenames[eval_task_id], device=device)

    t = data.t[0].float()

    # The starting position 
    # (n,d-dimensional, where n is the num of demos and 
    # d is the dimension of each point)
    #y_start = torch.unsqueeze(dataset.pos[0,0], dim=0)
    # Use the translated trajectory (goal at origin)
    y_start = data.pos[:,0]
    y_start = y_start.float()

    # The entire demonstration trajectory
    y_all = data.pos.float()

    # Predicted trajectory
    y_hat = node(t, y_start) # forward simulation

    # Compute trajectory metrics
    y_all_np = y_all.cpu().detach().numpy()
    y_hat_np = y_hat.cpu().detach().numpy()

    # De-normalize the data before computing trajectories
    y_all_np = data.unnormalize(y_all_np)
    y_hat_np = data.unnormalize(y_hat_np)    

    # Prediction
    if type == 'orientation':
        q_hat_np = data.from_tangent_plane(y_hat_np)
    elif type == 'position':
        q_hat_np = y_hat_np

    # Ground truth
    if type == 'orientation':
        q_np = data.quat_data
    elif type == 'position':
        q_np = y_all_np

    return q_hat_np, q_np