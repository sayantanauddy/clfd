'''
Evaluation functions for evaluating Hypernetwork+NODE models.

The eval function can be called (independent of the training) using the following snippet:

```
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from imitation_cl.test.hnet_node import eval_all
from imitation_cl.logging.utils import Dictobject

task_names_map = {'Angle': 'Ang',
                  'BendedLine': 'bLin',
                  'CShape': 'C',
                  'DoubleBendedLine': '2bLin',
                  'GShape': 'G',
                  'heee': 'he',
                  'JShape_2': 'J_2',
                  'JShape': 'J',
                  'Khamesh': 'Kh',
                  'Leaf_1': 'L_1',
                  'Leaf_2': 'L_2',
                  'Line': 'Lin',
                  'LShape': 'L',
                  'NShape': 'N',
                  'PShape': 'P',
                  'RShape': 'R',
                  'Saeghe': 'Sae',
                  'Sharpc': 'Sh_C',
                  'Sine': 'Sin',
                  'Snake': 'Snk',
                  'Spoon': 'Spn',
                  'Sshape': 'S',
                  'Trapezoid': 'Trp',
                  'Worm': 'Wrm',
                  'WShape': 'W',
                  'Zshape': 'Z'}

plot_args = Dictobject({'figw':16, 'figh':4, 'nrows':2, 'ncols':13, 'plot_fs':8})


log_hnet_node_5 = 'logs/hnet_node_lasa_all/210930_191842_seed100'
log_hnet_node_10 = 'logs/hnet_node_lasa_all/211003_133203_seed100'
log_hnet_node_64 = 'logs/hnet_node_lasa_all/211004_100328_seed100'
log_hnet_node_128 = 'logs/hnet_node_lasa_all/211004_225815_seed100'

logs = [log_hnet_node_5, log_hnet_node_10, log_hnet_node_64, log_hnet_node_128]

for i, log_dir in enumerate(logs):
    logging.info(f'############### Starting log {i+1}/{len(logs)} ###############')
    eval_all(log_dir=log_dir, 
            eval_mode='median', 
            train_task_id=None, 
            save_results=True, 
            save_plots=False, 
            plot_args=plot_args, 
            task_names_map=task_names_map)
```
By setting train_task_id to a specific number, we can test the trained model for that task. Setting train_task_id to None 
evaluates all the networks saved after each task.
'''

import os
from numpy.lib.npyio import save
from tqdm import trange
from argparse import ArgumentParser
import logging
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim

from imitation_cl.train.utils import check_cuda, set_seed, get_sequence
from imitation_cl.model.hypernetwork import HyperNetwork, TargetNetwork, str_to_ints, str_to_act, get_current_targets, calc_delta_theta, calc_fix_target_reg
from imitation_cl.model.node import NODE
from imitation_cl.data.lasa import LASA
from imitation_cl.data.utils import get_minibatch
from imitation_cl.plot.utils import plot_ode_simple
from imitation_cl.metrics.traj_metrics import mean_swept_error, mean_frechet_error_fast as mean_frechet_error, dtw_distance_fast as dtw_distance
from imitation_cl.logging.utils import custom_logging_setup, write_dict, read_dict, Dictobject, count_lines

#TODO Remove later
# Warning is a PyTorch bug
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def eval_task(args, task_id, hnet, tnet, node, device, eval_mode='mean'):

    hnet.eval()
    tnet.eval()

    filenames = get_sequence(args.seq_file)
    lasa = LASA(data_dir=args.data_dir, filename=filenames[task_id])

    # Generate parameters of the target network for the current task
    weights = hnet.forward(task_id)

    # Set the weights of the target network
    tnet.set_weights(weights)

    # Set the target network in the NODE
    node.set_target_network(tnet)
    node = node.float()
    node.eval()

    # The time steps
    t = torch.from_numpy(lasa.t[0]).float().to(device)

    # The starting position 
    # (n,d-dimensional, where n is the num of demos and 
    # d is the dimension of each point)
    y_start = torch.from_numpy(lasa.pos[:,0]).float().to(device)

    # The entire demonstration trajectory
    y_all = torch.from_numpy(lasa.pos).float().to(device)

    # Predicted trajectory
    y_hat = node(t, y_start) # forward simulation

    # Compute trajectory metrics
    y_all_np = y_all.cpu().detach().numpy()
    y_hat_np = y_hat.cpu().detach().numpy()

    metric_swept = mean_swept_error(y_all_np, y_hat_np, eval_mode)
    metric_frechet = mean_frechet_error(y_all_np, y_hat_np, eval_mode)
    metric_dtw = dtw_distance(y_all_np, y_hat_np, eval_mode)

    eval_traj_metrics = {'metric_swept': metric_swept, 
                         'metric_frechet': metric_frechet, 
                         'metric_dtw': metric_dtw}

    # Data that is used for creating a plot of demonstration
    # trajectories and predicted trajectories
    plot_data = [t, y_all, node.ode_rhs, y_hat.detach()]        

    return eval_traj_metrics, plot_data

def eval_all(log_dir, eval_mode='mean', train_task_id=None, save_results=True, save_plots=True, plot_args=None, task_names_map=None):

    # Check if cuda is available
    cuda_available, device = check_cuda()
    logging.info(f'cuda_available: {cuda_available}')

    # Read the command line arguments for the training from the log directory
    args_dict = read_dict(os.path.join(log_dir, 'commandline_args.json'))
    args = Dictobject(args_dict)

    # Dict for storing evaluation results
    # This will be written to a json file in the log folder
    eval_results = dict()

    # For storing command line arguments for this run
    eval_results['args'] = args_dict

    # For storing the evaluation results
    eval_results['data'] = dict()

    # Set the seed for reproducability
    set_seed(args.seed)      

    # Shapes of the target network parameters
    target_shapes = TargetNetwork.weight_shapes(n_in=args.tnet_dim, 
                                                n_out=args.tnet_dim, 
                                                hidden_layers=str_to_ints(args.tnet_arch), 
                                                use_bias=True)

    # Create the hypernetwork
    hnet = HyperNetwork(layers=str_to_ints(args.hnet_arch), 
                        te_dim=args.task_emb_dim, 
                        target_shapes=target_shapes,
                        device=device).to(device)
        
    # Create a target network without parameters
    # Parameters are supplied during the forward pass of the hypernetwork
    tnet = TargetNetwork(n_in=args.tnet_dim, 
                         n_out=args.tnet_dim, 
                         hidden_layers=str_to_ints(args.tnet_arch),
                         activation_fn=str_to_act(args.tnet_act), 
                         use_bias=True, 
                         no_weights=True,
                         init_weights=None, 
                         dropout_rate=-1, 
                         use_batch_norm=False, 
                         bn_track_stats=False,
                         distill_bn_stats=False, 
                         out_fn=None,
                         device=device).to(device)

    # The NODE uses the target network as the RHS of its
    # differential equation
    # Apart from this, the NODE has no other trainable parameters
    node = NODE(tnet).to(device)

    # Extract the list of demonstrations from the text file 
    # containing the sequence of demonstrations
    seq = get_sequence(args.seq_file)

    num_tasks = len(seq)

    # After the last task has been trained, we create a plot
    # showing the performance on all the tasks
    figw, figh = plot_args.figw, plot_args.figh
    plt.subplots_adjust(left=1/figw, right=1-1/figw, bottom=1/figh, top=1-1/figh)
    fig, axes = plt.subplots(figsize=(figw, figh), 
                             sharey=True, 
                             ncols=plot_args.ncols, 
                             nrows=plot_args.nrows,
                             subplot_kw={'aspect': 1})


    # If we want to evaluate trained models for all tasks
    if train_task_id is None:

        for task_id in range(num_tasks):

            logging.info(f'#### Evaluation started for task_id: {task_id} (task {task_id+1} out of {num_tasks}) ###')

            eval_results['data'][f'train_task_{task_id}'] = dict()

            # Load the networks for the current task_id
            hnet = torch.load(os.path.join(log_dir, 'models', f'hnet_{task_id}.pth'))
            tnet = torch.load(os.path.join(log_dir, 'models', f'tnet_{task_id}.pth'))
            node = torch.load(os.path.join(log_dir, 'models', f'node_{task_id}.pth'))

            r, c = 0, 0

            # Evaluate on all the past and current task_ids
            for eval_task_id in range(task_id+1):
                logging.info(f'Loaded network trained on task {task_id}, evaluating on task {eval_task_id}')

                # Figure is plotted only for the last task
                
                eval_traj_metrics, plot_data = eval_task(args, eval_task_id, hnet, tnet, node, device, eval_mode)

                # Plot the trajectories
                if task_id == (num_tasks-1) and save_plots:
                    r = eval_task_id//plot_args.ncols
                    c = eval_task_id%plot_args.ncols
                    if plot_args.nrows == 1:
                        ax=axes[c]
                    else:
                        ax=axes[r][c]
                    t, y_all, ode_rhs, y_hat = plot_data
                    handles, labels = plot_ode_simple(t, y_all, ode_rhs, y_hat, ax=ax)
                    if task_names_map is None:
                        title = f'Eval task {eval_task_id}'
                    else:
                        title = list(task_names_map.values())[eval_task_id]
                    ax.set_title(title, fontsize=plot_args.plot_fs)
                    fig.legend(handles, labels, loc='lower center', fontsize=plot_args.plot_fs, ncol=len(handles))
                
                logging.info(f'Evaluated trajectory metrics: {eval_traj_metrics}')

                # Store the evaluated metrics
                eval_results['data'][f'train_task_{task_id}'][f'eval_task_{eval_task_id}'] = eval_traj_metrics

    # If we want to evaluate the model for a specific task
    else:
        task_id = train_task_id

        logging.info(f'#### Evaluation started for task_id: {task_id} (task {task_id+1} out of {num_tasks}) ###')

        eval_results['data'][f'train_task_{task_id}'] = dict()

        # Load the networks for the current task_id
        hnet = torch.load(os.path.join(log_dir, 'models', f'hnet_{task_id}.pth'))
        tnet = torch.load(os.path.join(log_dir, 'models', f'tnet_{task_id}.pth'))
        node = torch.load(os.path.join(log_dir, 'models', f'node_{task_id}.pth'))

        r, c = 0, 0

        # Evaluate on all the past and current task_ids
        for eval_task_id in range(task_id+1):
            logging.info(f'Loaded network trained on task {task_id}, evaluating on task {eval_task_id}')

            # Figure is plotted only for the last task
            
            eval_traj_metrics, plot_data = eval_task(args, eval_task_id, hnet, tnet, node, device)

            # Plot the trajectories
            if save_plots:
                r = eval_task_id//plot_args.ncols
                c = eval_task_id%plot_args.ncols
                if plot_args.nrows == 1:
                    ax=axes[c]
                else:
                    ax=axes[r][c]  
                t, y_all, ode_rhs, y_hat = plot_data
                handles, labels = plot_ode_simple(t, y_all, ode_rhs, y_hat, ax=ax)
                if task_names_map is None:
                    title = f'Eval task {eval_task_id}'
                else:
                    title = list(task_names_map.values())[eval_task_id]
                ax.set_title(title, fontsize=plot_args.plot_fs)                
                fig.legend(handles, labels, loc='lower center', fontsize=plot_args.plot_fs, ncol=len(handles))
            
            logging.info(f'Evaluated trajectory metrics: {eval_traj_metrics}')

            # Store the evaluated metrics
            eval_results['data'][f'train_task_{task_id}'][f'eval_task_{eval_task_id}'] = eval_traj_metrics

    # Save the evaluation plot
    if save_plots:
        plt.savefig(os.path.join(log_dir, f'plot_trajectories_{args.description}.pdf'), bbox_inches='tight')

    # Write the evaluation results to a file in the log dir
    if save_results:
        write_dict(os.path.join(log_dir, f'eval_results_{eval_mode}.json'), eval_results)

    logging.info('Done')