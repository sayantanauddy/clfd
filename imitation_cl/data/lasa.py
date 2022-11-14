import os, sys, time
import numpy as np
import scipy.io as spio
import torch

class LASA():
    def __init__(self, data_dir, filename, norm=True, replicate_num=0, device=torch.device('cpu')):
        """
        Loads the LASA dataset

        From the README of the LASA dataset: 
        https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt

        DataSet: contains a library of 2D handwriting motions recorded from 
        Tablet-PC. For each motion, the user was asked to draw 7 demonstrations 
        of a desired pattern, by starting from different initial positions 
        (but fairly close to each other) and ending to the same final point. 
        These demonstrations may intersect each other. In total a library of 
        30 human handwriting motions were collected, of which 26 each correspond 
        to one single pattern, the remaining four motions each include more than 
        one pattern (called Multi Models). Without loss of generality, for all 
        handwriting motions (shapes), the target is by definition set at (0, 0). 
        Demonstrations are saved as '.mat' file and contains two variables:

            * dt: the average time steps across all demonstrations
            * demos: A structure variable containing ncessary informations
              about all demonstrations. The variable 'demos' has the following
              format:
              - demos{n}:     Information related to the n-th demonstration.
              - demos{n}.pos: 2 x 1000 matrix representing the motion in 2D
                              space. The first and second rows correspond to
                              x and y axes in the Cartesian space, respectively.
              - demons{n}.t:  1 x 1000 vector indicating the corresponding time
                              for each datapoint (i.e. each column of demos{n}.pos).
              - demos{n}.vel: 2 x 1000 matrix representing the velocity of the motion.
              - demos{n}.acc: 2 x 1000 matrix representing the acceleration of the motion.

        Accordingly, the data from the mat file is loaded into an object of the LASA 
        class has the following (important) members:
            pos (numpy array): Array of shape 7x1000x2
            vel (numpy array): Array of shape 7x1000x2
            acc (numpy array): Array of shape 7x1000x2
            t (numpy array): Array of shape 7x1000

        If replicate_num is not None, then the shape of the pos,vel,acc arrays 
        becomes 7x(1000+replicate_num)x2 and the shape of t is 7x(1000+replicate_num)
        This can be used to strengthen the effect of the point attractor.

        Args:
            data_dir (str): Directory where the LASA mat files are located
            filename (str): Which mat file to load (with or without .mat extension)
            norm (bool): Whether to normalize the trajectories across the demonstrations
            replicate_num (int): How many times the end points of the trajectory is repeated
            device (torch device): Device on which to train
        """

        # Define Variables
        self.data_dir = data_dir
        self.filename = filename
        self.norm = norm
        self.device = device

        # Load the mat file
        mat = spio.loadmat(os.path.join(self.data_dir,filename), 
                           simplify_cells=True,
                           appendmat=True)

        # Contents of the dict mat:
        # mat['dt'] - The average time steps across all demonstrations
        # mat['demos'][i] - Information about the i'th demonstration
        #   mat['demos'][i]['pos'] - 2 x 1000 matrix of 2D positions in Cartesian space
        #   mat['demos'][i]['t']   - 1 x 1000 matrix of time of each datapoint
        #   mat['demos'][i]['vel'] - 2 x 1000 matrix of 2D velocity
        #   mat['demos'][i]['acc'] - 2 x 1000 matrix of 2D acceleration

        self.num_demos = len(mat['demos'])
        self.traj_dim = mat['demos'][0]['pos'].shape[0]
        self.traj_len = mat['demos'][0]['pos'].shape[1]

        # Scalar value of the average time step
        self.dt = mat['dt']

        # Load the pos,t,vel and acc information
        self.pos, self.t, self.vel, self.acc = list(), list(), list(), list()
        for i in range(self.num_demos):
            self.pos.append(mat['demos'][i]['pos'].T)
            self.t.append(mat['demos'][i]['t'].T)
            self.vel.append(mat['demos'][i]['vel'].T)
            self.acc.append(mat['demos'][i]['acc'].T)

        # Convert trajectories to numpy arrays
        self.pos = np.array(self.pos)
        self.t = np.array(self.t)
        self.vel = np.array(self.vel)
        self.acc = np.array(self.acc)

        # Normalize the trajectories if needed
        if self.norm:
            self.pos, self.pos_mean, self.pos_std = self.normalize(self.pos)
            self.vel, self.vel_mean, self.vel_std = self.normalize(self.vel)
            self.acc, self.acc_mean, self.acc_std = self.normalize(self.acc)

        if replicate_num > 0:
            # Replicate the last point of the trajectory `replicate_num` times
            self.pos = np.hstack((self.pos, np.repeat(self.pos[:,[-1],:], replicate_num, axis=1)))
            self.vel = np.hstack((self.vel, np.repeat(self.vel[:,[-1],:], replicate_num, axis=1)))
            self.acc = np.hstack((self.acc, np.repeat(self.acc[:,[-1],:], replicate_num, axis=1)))

            # For each demonstration, the difference of the last and second-last time point
            # is used to increase the time points in an even manner
            # For example, consider t has the shape (2,10)
            '''
            t = np.array([[0.0, 1.3, 4.5, 4.7, 5.2, 5.7, 7.1, 7.6, 8.1, 9.2], 
                          [0.1, 1.4, 4.6, 4.8, 5.1, 5.6, 7.0, 7.5, 8.0, 9.4]])

            t_new = np.hstack((t, t[:,[-1]] + (t[:,[-1]]-t[:,[-2]])*np.arange(1,num_repeat+1)))
            '''
            # Then t_new becomes
            '''
            [[ 0.   1.3  4.5  4.7  5.2  5.7  7.1  7.6  8.1  9.2 10.3 11.4 12.5 13.6 14.7 15.8 16.9 18.  19.1 20.2]
             [ 0.1  1.4  4.6  4.8  5.1  5.6  7.   7.5  8.   9.4 10.8 12.2 13.6 15.  16.4 17.8 19.2 20.6 22.  23.4]]
            '''
            self.t = np.hstack((self.t, self.t[:,[-1]] + (self.t[:,[-1]]-self.t[:,[-2]])*np.arange(1,replicate_num+1)))

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

class LASAExtended():

    def __init__(self, datafile, seq_len, norm=True, device=torch.device('cpu')):
        """
        Loads the extended LASA trajectories. 
        Can be used for trajectories of 2 or more dimensions.

        All the data is first moved to the GPU (if available).
        Since the entire dataset is small, moving it to the GPU may be more efficient.

        Args:
            datafile (str): Path of the numpy archive (npz) with the trajectory data.
            seq_len (int): Length of sampled trajectory subsequence.
            norm (bool, optional): Whether to normalize the trajectories. Defaults to True.
            device (torch.device, optional): Which device to move the data to. Defaults to torch.device('cpu').
        """
        self.datafile = datafile
        self.seq_len = seq_len
        self.norm = norm
        self.device = device

        # Read the numpy data and store all the trajectories
        data = np.load(self.datafile)

        # The trajectores and time stamps
        self.pos = data['pos']
        self.t = data['t']

        # Find the number of sequences N and the length of each sequence T
        [self.num_demos, self.num_timesteps, self.data_dim] = self.pos.shape

        # Normalize the trajectories if needed
        # Convert to tensors and move to the required device
        if self.norm:
            self.pos, self.pos_mean, self.pos_std = self.normalize(self.pos)
            self.pos, self.pos_mean, self.pos_std = torch.from_numpy(self.pos), torch.from_numpy(self.pos_mean), torch.from_numpy(self.pos_std)
            self.pos, self.pos_mean, self.pos_std = self.pos.to(device), self.pos_mean.to(device), self.pos_std.to(device)
        else:
            self.pos_mean, self.pos_std = None, None
            self.pos = torch.from_numpy(self.pos)
            self.pos = self.pos.to(device)
        self.t = torch.from_numpy(self.t).to(device)

        # Translating the goal to the origin
        self.pos_offset = self.pos[:,-1,:]

    def normalize(self, arr):
        """
        Normalizes the input array
        """
        # Compute the mean and std for x,y across all demonstration trajectories
        mean = np.expand_dims(np.mean(np.reshape(arr, (self.num_demos*self.num_timesteps, self.data_dim)), axis=0), axis=0)
        std = np.expand_dims(np.std(np.reshape(arr, (self.num_demos*self.num_timesteps, self.data_dim)), axis=0), axis=0)
        arr = (arr - mean)/std
        return arr, mean, std

    def denormalize(self, arr, type='pos'):
        """
        Denormalizes the input array
        """
        if not self.norm:
            return arr
            
        if type == 'pos':
            arr = arr*self.pos_std.cpu().detach().numpy() + self.pos_mean.cpu().detach().numpy()
        else:
            raise NotImplementedError(f'Unknown type={type}')
        return arr

    def unnormalize(self, arr, type='pos'):
        return self.denormalize(arr, type)

    def zero_center(self):

        # Find the goal position
        self.goal = self.pos[:,-1,:]

        # Translate the goal position to the origin
        # self.pos_goal_origin should be used for training
        self.pos_goal_origin = self.pos - torch.stack([self.goal]*self.pos.shape[1], axis=1)

    def unzero_center(self, arr):

        # Translate the prediction away from the origin
        return arr + torch.stack([self.goal]*self.pos.shape[1], axis=1)
