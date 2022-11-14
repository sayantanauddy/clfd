from operator import index
import os
import numpy as np
import glob
from copy import deepcopy
import shutil
from shutil import copyfile
import json
import re
import datetime
import warnings
import torch
from imitation_cl.logging.utils import read_dict, write_dict, count_lines, check_dirtree, Dictobject

import pandas as pd

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

def remove_processed_upperbaseline(base_logdir):
    # Remove any dirs with the _processed suffix
    del_list = list()
    for path, dirs, files in os.walk(base_logdir):
        if os.path.basename(path).endswith('_processed') and len(dirs)>0:
            del_list.append(path)

    for p_dir in del_list:
        shutil.rmtree(p_dir)

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

        copyfile(os.path.join(os.path.dirname(log), 'commandline_args.json'),
                 os.path.join(os.path.dirname(processed_log_path), 'commandline_args.json'))

        # Copy the trajectory plot files
        traj_plot = glob.glob(os.path.join(os.path.dirname(log), '*.pdf'))

        for tp in traj_plot:
            copyfile(os.path.join(tp),
                     os.path.join(os.path.dirname(processed_log_path), os.path.basename(tp)))


def find_upperbaselinelogs(base_logdir, upperbaseline_prefixes=['tr_node_', 'tr_lsddm_', 'tr_iflow']):
    """
    Finds log_multiseed_dirs of upper baselines (SG)
    """
    upper_baseline_log_multiseed_dirs = list()

    for path, dirs, files in os.walk(base_logdir):
        for prefix in upperbaseline_prefixes:
            if prefix in os.path.basename(path) and len(dirs)>0:
                upper_baseline_log_multiseed_dirs.append(path)

    return upper_baseline_log_multiseed_dirs

def find_logs(base_logdir, ignore_dirs, eval_file_name='eval_results.json', upperbaseline_prefixes=['tr_node_', 'tr_lsddm_', 'tr_iflow']):

    eval_log_paths = list()
    filtered_eval_log_paths = list()

    for path, dirs, files in os.walk(base_logdir):

        # Find the eval files
        if eval_file_name in files:
            eval_log_paths.append(os.path.join(path, eval_file_name))

    # Remove unprocessed upper baseline logs
    for d in eval_log_paths:
        remove_flag = False
        for prefix in upperbaseline_prefixes:
            if prefix in d and '_processed' not in d:
                remove_flag = True

        if not remove_flag:
            filtered_eval_log_paths.append(d)

    return filtered_eval_log_paths                

def find_training_time(log_path, task_id):
    
    regex = f'Training started for task_id: {task_id}'

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
                t_start = datetime.datetime.strptime(f"{date_str}:{line.split(' ')[0].replace('[','').replace(']','')}", '%y:%m:%d:%H:%M:%S')
                t_end = datetime.datetime.strptime(f"{date_str}:{next(file).split(' ')[0].replace('[','').replace(']','')}", '%y:%m:%d:%H:%M:%S')
                t_diff = (t_end - t_start).total_seconds()
                # If the training ended on the next day
                if t_diff < 0:
                    t_end = datetime.datetime.strptime(f"{next_date_str}:{next(file).split(' ')[0].replace('[','').replace(']','')}", '%y:%m:%d:%H:%M:%S')
                    t_diff = (t_end - t_start).total_seconds()
                # End time must be greater than the start
                assert t_diff > 0, f't_end:{t_end}<t_start:{t_start}, log_path={log_path}'

    return t_diff

def compute_parameter_size(cl_type, model_dir, task_id, last_task_id=25, verbose=False, split_size=False):

    if cl_type == 'SG':
        # Models are saved in the unprocessed directory
        model_dir = model_dir.replace('_processed','')

    if cl_type in ['HN','CHN']:

        model_path = os.path.join(f'{model_dir}', f'hnet_{last_task_id}.pth')

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

    elif cl_type in ['FT', 'SI', 'MAS', 'SG', 'REP']:
        
        # Parameter size does not vary with seed, just pick the model from the first log
        model_path = os.path.join(f'{model_dir}', f'node_{last_task_id}.pth')

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

        if cl_type == 'SG':
            total_size *= (task_id+1)

        if verbose:
            print(f'task_emb_size={task_emb_size}, theta_size={theta_size}, buffer_size={buffer_size}')

        if split_size:
            return total_size, task_emb_size, theta_size, buffer_size
        else:
            return total_size

    else:
        raise NotImplementedError(f'Unknown model_type {cl_type}')

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line

def check_success(log_file):
    """
    Checks the log file to see if the training run succeeded
    """
    with open(log_file) as f_in:
        for line in nonblank_lines(f_in):
            pass

    last_word = line.split(' ')[-1]

    if last_word=='Completed':
        return True
    else:
        return False

script_to_cl_map = {
    'tr_node': 'SG',
    'tr_rep_node': 'REP',
    'tr_ft_node': 'FT',
    'tr_si_node': 'SI',
    'tr_mas_node': 'MAS',
    'tr_hn_node': 'HN',
    'tr_chn_node': 'CHN',
    'tr_lsddm': 'SG',
    'tr_ft_lsddm': 'FT',
    'tr_rep_lsddm': 'REP',
    'tr_si_lsddm': 'SI',
    'tr_mas_lsddm': 'MAS',
    'tr_hn_lsddm': 'HN',
    'tr_chn_lsddm': 'CHN',
    'tr_iflow': 'SG'
}

script_to_traj_map = {
    'tr_node': 'NODE',
    'tr_rep_node': 'NODE',
    'tr_ft_node': 'NODE',
    'tr_si_node': 'NODE',
    'tr_mas_node': 'NODE',
    'tr_hn_node': 'NODE',
    'tr_chn_node': 'NODE',
    'tr_lsddm': 'LSDDM',
    'tr_rep_lsddm': 'LSDDM',
    'tr_ft_lsddm': 'LSDDM',
    'tr_si_lsddm': 'LSDDM',
    'tr_mas_lsddm': 'LSDDM',
    'tr_hn_lsddm': 'LSDDM',
    'tr_chn_lsddm': 'LSDDM',
    'tr_iflow': 'IFLOW'
}

def create_results_df(base_logdir, insert_comment='', verbose=False):

    experiment_data_list = list()

    # Remove existing processed upper baseline (SG) logs if any
    remove_processed_upperbaseline(base_logdir)

    # Find the upper baseline logs (unprocessed)
    upper_baseline_log_multiseed_dirs = find_upperbaselinelogs(base_logdir)

    # Process the upper baseline logs
    for d in upper_baseline_log_multiseed_dirs:
        process_upperbaseline_eval(log_multiseed_dir=d, 
                                processed_log_dir=f'{d}_processed')

    # For further processing, we need to ignore the logs in upper_baseline_log_multiseed_dirs
    # Find all log files and parent directories
    eval_paths = find_logs(base_logdir, ignore_dirs=upper_baseline_log_multiseed_dirs)

    for e in eval_paths:
        eval_path = e
        log_path = os.path.dirname(e)

        # If this training is not successful, skip and raise an alert
        success = check_success(os.path.join(log_path, 'log.log'))

        if not success:
            if verbose:
                warnings.warn(f'Log path {log_path} corresponds to an unsuccessful run')
            continue

        # Read the commandline args
        args_dict = read_dict(os.path.join(log_path, 'commandline_args.json'))

        # Patch for iFlow as it does not have all the necessary commandline args
        if 'tr_iflow' in args_dict['description']:
            args_dict['data_class'] = 'LASA'
            args_dict['num_iter'] = args_dict['nr_epochs']*27 # len(dataloader)=27
            args_dict['explicit_time'] = 0
            args_dict['tnet_dim'] = 2

        args = Dictobject(args_dict)

        # Get args as a string
        args_str = json.dumps(args_dict)

        # Fetch information from the command line arguments
        seed = args.seed
        dataset = args.data_class
        num_iters = args.num_iter
        explicit_time = args.explicit_time
        data_dim = args.tnet_dim

        # Extract script name, CL and traj learning type
        script_name = os.path.basename(log_path[:log_path.find(dataset)-1])
        cl_type = script_to_cl_map[script_name]
        traj_type = script_to_traj_map[script_name]

        # Extract the date and time
        log_path_basename = os.path.basename(log_path)
        date_and_time = log_path_basename[:log_path_basename.find('seed')-1]
        date = date_and_time[:date_and_time.find('_')]
        time = date_and_time[date_and_time.find('_')+1:]

        eval_results_dict = read_dict(eval_path)

        metric_errors = eval_results_dict['data']['metric_errors']
        num_tasks = len(metric_errors.keys())

        # Path to the trained models
        model_dir = os.path.join(log_path, 'models')

        # Any other custom information
        info = 'NA'
        if script_name == 'tr_rep_node' or script_name == 'tr_rep_lsddm':
            info = f'REP_ITER_MULTI={args.train_iter_multiplier}'

        for train_task in metric_errors.keys():

            train_task_id = int(train_task[train_task.rfind('_')+1:])

            # Find the time for training
            train_time = find_training_time(log_path=os.path.join(log_path,'log.log'), 
                                            task_id=train_task_id)

            # Find the parameter count
            try:
                model_param_cnt = compute_parameter_size(cl_type=cl_type,
                                                        model_dir=model_dir,
                                                        task_id=train_task_id,
                                                        last_task_id=num_tasks-1,
                                                        verbose=False,
                                                        split_size=False)
            except FileNotFoundError as e:
                #print(e)
                model_param_cnt = -1

            for eval_task in metric_errors[train_task].keys():

                eval_task_id = int(eval_task[eval_task.rfind('_')+1:])

                try:
                    dtw_errors = metric_errors[train_task][eval_task]['dtw']
                    frechet_errors = metric_errors[train_task][eval_task]['frechet']
                    swept_errors = metric_errors[train_task][eval_task]['swept']

                    # If there is no error till now, then we are dealing with Position data
                    quat_errors = [-1.0 for _ in dtw_errors]
                except Exception as e:
                    # If there is an error here, then we are dealing with Orientation data
                    quat_errors = metric_errors[train_task][eval_task]['quat_error']
                    dtw_errors = [-1.0 for _ in quat_errors]
                    frechet_errors = [-1.0 for _ in quat_errors]
                    swept_errors = [-1.0 for _ in quat_errors]

                for obs_id, errors in enumerate(zip(dtw_errors, frechet_errors, swept_errors, quat_errors)):

                    # Create a row (in the form of a dict)
                    row_data = {
                        'script_name': script_name,
                        'date': date,
                        'time': time,
                        'dataset': dataset,
                        'cl_type': cl_type,
                        'traj_type': traj_type,
                        'seed': seed,
                        'data_dim': data_dim,
                        'num_tasks': num_tasks,
                        'num_iters': num_iters,
                        'explicit_time': explicit_time,
                        'model_param_cnt': model_param_cnt,
                        'train_time': train_time,
                        'train_task_id': train_task_id,
                        'eval_task_id': eval_task_id,
                        'obs_id': obs_id,
                        'dtw': errors[0],
                        'frechet': errors[1],
                        'swept': errors[2],
                        'quat_error': errors[3],
                        'info': info,
                        'log_path': log_path,
                        'model_dir': model_dir,
                        'args_str': args_str,
                        'insert_comment': insert_comment,
                    }
                
                    experiment_data_list.append(row_data)

    # Convert to a dataframe
    experiment_data_df = pd.DataFrame.from_dict(experiment_data_list)

    return experiment_data_df

def concat_results(df_list):
    return pd.concat(df_list)

def load_database(database_path, index_col=False):
    return pd.read_csv(database_path, index_col=index_col)

def verify_database(database_df, script_names, dataset, explicit_times, num_obs, num_tasks, data_dim, seeds):

    print(f'Verifying for dataset: {dataset}')

    all_correct = True

    for script_name in script_names:
        for explicit_time in explicit_times:
            for seed in seeds:
                query = f'script_name=="{script_name}" and dataset=="{dataset}" and explicit_time=={explicit_time} and data_dim=={data_dim} and seed=={seed}'
                #if script_name=='tr_rep_node':
                #    print(f'Query: {query}')
                selection = database_df.query(query).reset_index()
                expected_cnt = int(num_obs*(num_tasks*(num_tasks+1)/2))
                actual_cnt = len(selection)
                try:
                    assert actual_cnt==expected_cnt                        
                except AssertionError as e:
                    all_correct = False
                    print(f'[script_name:"{script_name}",dataset:"{dataset}",explicit_time:{explicit_time},seed:{seed}], expected={expected_cnt}, found={actual_cnt} rows')

    if all_correct:
        print(f'Verification complete for dataset: {dataset}')
    else:
        print(f'Errors found for dataset: {dataset}')