import os
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter
from copy import deepcopy

class HelloWorld():
    def __init__(self, data_dir, filename, norm=True, device=torch.device('cpu'), record_freq=60):
        """
        Loads the HelloWorld dataset

        Args:
            data_dir (str): Directory where the LASA mat files are located
            filename (str): Which mat file to load (with or without .mat extension)
            norm (bool): Whether to normalize the trajectories across the demonstrations
            device (torch device): Device on which to train
            record_freq (int): Frequency of recording during demonstration in Hz
        """

        # Define Variables
        self.data_dir = data_dir
        self.filename = filename
        self.norm = norm
        self.device = device
        self.record_freq = record_freq

        # Load the .npy file
        self.pos = np.load(os.path.join(self.data_dir, filename))
        self.pos = self.pos[:,:,0:2]
        self.num_demos = self.pos.shape[0]
        self.traj_dim = self.pos.shape[2]
        self.traj_len = self.pos.shape[1]

        self.t = [np.linspace(0.0, self.traj_len, self.traj_len)/record_freq]

        # Normalize the trajectories if needed
        if self.norm:
            self.pos, self.pos_mean, self.pos_std = self.normalize(self.pos)

    def normalize(self, arr):
        """
        Normalizes the input array
        """
        # Compute the mean and std for x,y across all demonstration trajectories
        mean = np.expand_dims(np.mean(np.reshape(arr, (self.num_demos*self.traj_len, self.traj_dim)), axis=0), axis=0)
        std = np.expand_dims(np.std(np.reshape(arr, (self.num_demos*self.traj_len, self.traj_dim)), axis=0), axis=0)
        arr = (arr - mean)/std
        return arr, mean, std

    def unnormalize(self, arr, type='pos'):
        """
        Denormalizes the input array
        """
        if not self.norm:
            return arr
            
        if type == 'pos':
            arr = arr*self.pos_std + self.pos_mean
        else:
            raise NotImplementedError(f'Unknown type={type}')
        return arr

### Functions for processing the helloworld dataset

def plot_demo(pos):

    marker = 'o'
    markersize = 2

    num_demos = pos.shape[0]
    num_points = pos.shape[1]

    fig = plt.figure(constrained_layout=True, figsize=(20,5))
    gs = fig.add_gridspec(2, 8, hspace=0.05, wspace=0.05)
    ax00 = fig.add_subplot(gs[0, 0:6])
    ax10 = fig.add_subplot(gs[1, 0:6])
    ax_1 = fig.add_subplot(gs[:, 6:])

    for i in range(num_demos):
        timesteps_ticks = np.arange(0, num_points, 20)

        ax00.plot(pos[i,:,0], marker=marker, markersize=markersize)
        ax00.set_xlabel('timestep')
        ax00.set_ylabel('State x0')
        ax00.set_xticks(timesteps_ticks)
        ax00.grid(True)

        ax10.plot(pos[i,:,1], marker=marker, markersize=markersize)
        ax10.set_xlabel('timestep')
        ax10.set_ylabel('State x1')
        ax10.set_xticks(timesteps_ticks)
        ax10.grid(True)
        
        ax_1.plot(pos[i,:,0], pos[i,:,1], marker=marker, markersize=markersize)
        ax_1.set_xlabel('State x0')
        ax_1.set_ylabel('State x1')
        ax_1.grid(True)

def load_raw(dir_path, file_wildcard='*.txt', dim=2):

    demo_files = glob.glob(os.path.join(dir_path, file_wildcard))

    demos = list()
    for demo_file in demo_files:
        demo = np.loadtxt(demo_file)[:,:dim]
        demos.append(demo)

    return np.array(demos)

def align_end_points(demos):

    for k,v in demos.items():
        
        num_demo_per_letter = demos[k].shape[0]

        # For each letter, set the end points of all demos to the same point
        # by shifting each trajectory
        # Find end point of the last demo
        last_demo_end_point = demos[k][-1,-1,:]

        # Pick each demo at a time
        for demo_idx in range(num_demo_per_letter):

            demo = demos[k][demo_idx]
            demo_end_point = demo[-1,:]
 
            end_point_diff = last_demo_end_point - demo_end_point

            demos[k][demo_idx] += end_point_diff

    return demos

def remove_deadzones(demos, deadzones):

    for k,v in demos.items():
        
        # Fetch the manually annotated deadzones for this letter
        deadzone = deadzones[k]

        # Cut off the deadzones from the start and the end
        demos[k] = v[:, deadzone[0]:deadzone[1]+1, :]

    return demos

def interpolate_traj(demos, orig_traj_len):

    for k,v in demos.items():
        
        num_demo_per_letter = demos[k].shape[0]

        # Pick each demo at a time
        letter_demos = list()
        for demo_idx in range(num_demo_per_letter):

            # Time steps forms the x axis
            x = np.arange(0, v[demo_idx].shape[0])

            # Treat the 2 state dimensions as 1D functions
            y0 = v[demo_idx,:,0]
            y1 = v[demo_idx,:,1]

            # Create the interpolation functions for the two state dimensions
            f0 = interpolate.interp1d(x, y0)
            f1 = interpolate.interp1d(x, y1)

            # Interpolate the time steps
            x_new = np.linspace(x[0], x[-1], num=orig_traj_len)
            y0_new = f0(x_new)
            y1_new = f1(x_new)
            y = np.vstack([y0_new,y1_new]).T

            letter_demos.append(y)

        demos[k] = np.array(letter_demos)

    return demos

def smooth(demos, window_size=51, poly_order=3):

    for k,v in demos.items():
        
        num_demo_per_letter = demos[k].shape[0]

        # Pick each demo at a time
        letter_demos = list()
        for demo_idx in range(num_demo_per_letter):

            # Treat the 2 state dimensions as 1D functions
            y0 = v[demo_idx,:,0]
            y1 = v[demo_idx,:,1]

            # Smooth using the savgol filter
            y0 = savgol_filter(y0, window_size, poly_order)
            y1 = savgol_filter(y1, window_size, poly_order)

            y = np.vstack([y0,y1]).T

            letter_demos.append(y)

        demos[k] = np.array(letter_demos)

    return demos

def save(demos, processed_data_dir):
    for k,v in demos.items():
        np.save(os.path.join(processed_data_dir,f'{k}.npy'), demos[k])

def check(demos, shape=(10,1000,2)):
    for k,v in demos.items():
        assert v.shape==shape, f'Wrong shape {v.shape} for {k}'

def remove_demos(demos, keep_list=None):
    if keep_list is not None:
        for k,v in demos.items():
            demos[k] = v[keep_list]
    return demos

def convert_m_to_cm(demos):
    for k,v in demos.items():
        v *= 100.0

    return demos

def process_demos(raw_data_dir, processed_data_dir, deadzones):

    letters = deadzones.keys()

    # Load the raw data
    demos = dict()
    for letter in letters:
        demo = load_raw(dir_path=os.path.join(raw_data_dir, letter), 
                        file_wildcard=f'{letter}*.txt', 
                        dim=2)
        demos[letter] = demo

    raw_demos = deepcopy(demos)

    # Smooth the trajectories
    demos = smooth(demos, window_size=201, poly_order=2)

    # Remove deadzones
    demos = remove_deadzones(demos=demos, deadzones=deadzones)

    # Interpolate to have same number of points in each trajectory
    demos = interpolate_traj(demos, orig_traj_len=1000)

    # Align end points
    demos = align_end_points(demos)

    # Remove unwanted demos
    demos = remove_demos(demos, keep_list=[0,1,2,3,4,5,6,7])

    # Convert units from m to cm
    demos = convert_m_to_cm(demos)

    # Check that processed demos have the correct shape
    check(demos, shape=(8,1000,2))

    # Save the processed demos
    save(demos, processed_data_dir)

    return demos, raw_demos

