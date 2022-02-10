import os
import numpy as np
from numba import jit
import glob
from imitation_cl.logging.utils import read_dict, count_lines

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

def accuracy(errors, threshold):
    """
    Calcuate how many of the errors are lower than the 
    `threshold`

    Args:
        errors (numpy array): Errors between individual trajectories (num_traj,)
        threshold (float): Threshold for determining if two trajectories are similar

    Returns:
        float: Percentage of predicted trajectories which are accurate (0.0 to 1.0)
    """
    errors = np.array(errors)
    return np.sum(errors<=threshold)/errors.shape[0]
    
def get_arr(json_path, local_home, acc_thresh_dict, agg_mode='mean'):
    """
    Reads a evaluation log file and converts it into a 
    dict of numpy arrays, with one array for each metric.

    Args:
        json_path (str): Path to the evaluation log file
        local_home (str): Path to the local home directory

    Returns:
        dict of numpy arrays: One np array for each metric
    """

    eval_dict = read_dict(json_path)

    args = eval_dict['args']
    data = eval_dict['data']

    seed = args['seed']
    description = args['description']

    num_tasks, cl_sequence = count_lines(os.path.join(local_home, '/'.join(args['seq_file'].split('/')[3:])))

    # How many metrics are measured?
    metrics = data['metric_errors']['train_task_0']['eval_task_0'].keys()

    # Create a numpy array for each metric
    metric_arr = dict()
    acc_arr = dict()

    for metric in metrics:
        metric_arr[metric] = np.zeros((num_tasks, num_tasks))
        acc_arr[metric] = np.zeros((num_tasks, num_tasks))

    for task_name in cl_sequence:
        train_index = cl_sequence.index(task_name)

        task_results = data['metric_errors'][f'train_task_{train_index}']

        for eval_index in range(train_index+1):
            eval_results = task_results[f'eval_task_{eval_index}']

            for metric_name, metric_errors in eval_results.items():
                
                metric_agg = None
                if agg_mode == 'mean':
                    metric_agg = np.mean(metric_errors)
                elif agg_mode == 'median':
                    metric_agg = np.median(metric_errors)
                else:
                    raise NotImplementedError(f'Unknown agg_mode: {agg_mode}')

                acc = accuracy(metric_errors, threshold=acc_thresh_dict[metric_name])

                metric_arr[metric_name][train_index, eval_index] = metric_agg
                acc_arr[metric_name][train_index, eval_index] = acc

    return metric_arr, acc_arr, seed, description, cl_sequence

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


def get_cl_metrics_multiseed(arr_list):

    acc_list, bwt_list, bwt_plus_list, rem_list = list(), list(), list(), list()
    for arr in arr_list:
        acc, bwt, bwt_plus, rem = get_cl_metrics(arr)
        acc_list.append(acc)
        bwt_list.append(bwt)
        bwt_plus_list.append(bwt_plus)
        rem_list.append(rem)

    cl_metrics_mean = {'acc': np.mean(acc_list), 'bwt': np.mean(bwt_list), 'bwt_plus': np.mean(bwt_plus_list), 'rem': np.mean(rem_list)}
    cl_metrics_std = {'acc': np.std(acc_list), 'bwt': np.std(bwt_list), 'bwt_plus': np.std(bwt_plus_list), 'rem': np.std(rem_list)}

    return cl_metrics_mean, cl_metrics_std
        

def get_arr_multiseed(log_multiseed_dir, acc_thresh_dict, calc_mode='mean'):

    local_home = os.path.expanduser('~')
    eval_files = get_eval_files(log_multiseed_dir)

    # Trajectory metrics
    metric_arrs_swept = list()
    metric_arrs_frechet = list()
    metric_arrs_dtw = list()

    # Log10 of trajectory metrics
    log_metric_arrs_swept = list()
    log_metric_arrs_frechet = list()
    log_metric_arrs_dtw = list()

    # Threshold-based accuracy computed from trajectory metrics
    acc_metric_arrs_swept = list()
    acc_metric_arrs_frechet = list()
    acc_metric_arrs_dtw = list()

    seeds = list()

    for eval_file in eval_files:
        metric_arr, acc_arr, seed, description, cl_sequence = get_arr(eval_file, local_home, acc_thresh_dict=acc_thresh_dict)

        metric_arrs_swept.append(metric_arr['swept'])
        metric_arrs_frechet.append(metric_arr['frechet'])
        metric_arrs_dtw.append(metric_arr['dtw'])

        log_metric_arrs_swept.append(np.log10(metric_arr['swept'] + 1e-8))
        log_metric_arrs_frechet.append(np.log10(metric_arr['frechet'] + 1e-8))
        log_metric_arrs_dtw.append(np.log10(metric_arr['dtw'] + 1e-8))

        acc_metric_arrs_swept.append(acc_arr['swept'])
        acc_metric_arrs_frechet.append(acc_arr['frechet'])
        acc_metric_arrs_dtw.append(acc_arr['dtw'])

        seeds.append(seed)

    if calc_mode == 'mean':
        metric_arr_swept = np.mean(metric_arrs_swept, axis=0)
        metric_arr_frechet = np.mean(metric_arrs_frechet, axis=0)
        metric_arr_dtw = np.mean(metric_arrs_dtw, axis=0)

        log_metric_arr_swept = np.mean(log_metric_arrs_swept, axis=0)
        log_metric_arr_frechet = np.mean(log_metric_arrs_frechet, axis=0)
        log_metric_arr_dtw = np.mean(log_metric_arrs_dtw, axis=0)

        cl_metric_arr_swept, cl_metric_arr_err_swept = get_cl_metrics_multiseed(acc_metric_arrs_swept)
        cl_metric_arr_frechet, cl_metric_arr_err_frechet = get_cl_metrics_multiseed(acc_metric_arrs_frechet)
        cl_metric_arr_dtw, cl_metric_arr_err_dtw = get_cl_metrics_multiseed(acc_metric_arrs_dtw)

        metric_arr_err_swept = np.std(metric_arrs_swept, axis=0)
        metric_arr_err_frechet = np.std(metric_arrs_frechet, axis=0)
        metric_arr_err_dtw = np.std(metric_arrs_dtw, axis=0)

        log_metric_arr_err_swept = np.std(log_metric_arrs_swept, axis=0)
        log_metric_arr_err_frechet = np.std(log_metric_arrs_frechet, axis=0)
        log_metric_arr_err_dtw = np.std(log_metric_arrs_dtw, axis=0)

    else:
        raise NotImplementedError(f'Unknown calc_mode: {calc_mode}')

    mid = {'swept': metric_arr_swept,
           'frechet': metric_arr_frechet,
           'dtw': metric_arr_dtw,
           }

    log10_mid = {'swept': log_metric_arr_swept,
                 'frechet': log_metric_arr_frechet,
                 'dtw': log_metric_arr_dtw,
                }

    cl_mid = {'swept': cl_metric_arr_swept,
              'frechet': cl_metric_arr_frechet,
              'dtw': cl_metric_arr_dtw
             }

    err = {'swept': metric_arr_err_swept,
           'frechet': metric_arr_err_frechet,
           'dtw': metric_arr_err_dtw,
          }

    log10_err = {'swept': log_metric_arr_err_swept,
                 'frechet': log_metric_arr_err_frechet,
                 'dtw': log_metric_arr_err_dtw
                }

    cl_err = {'swept': cl_metric_arr_err_swept,
              'frechet': cl_metric_arr_err_frechet,
              'dtw': cl_metric_arr_err_dtw
             }

    return mid, log10_mid, cl_mid, err, log10_err, cl_err, seeds, description

