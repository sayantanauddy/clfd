import os, sys, time
import numpy as np
import scipy.io as spio
from pyquaternion import Quaternion
import torch

class RobotTasksExtended():

    def __init__(self, data_dir, datafile, norm=True, pos_idx={'x': 0, 'y': 1, 'z': 2}, rot_idx={'w': 3, 'x': 4, 'y': 5, 'z': 6}, record_freq=60, device=torch.device('cpu')):
        """
        Loads trajectories from the robot tasks dataset. 
        Can be used for trajectories of 2 or more dimensions.

        All the data is first moved to the GPU (if available).
        Moving the data to the GPU may be more efficient.

        Args:
            datafile (str): Path of the numpy archive (npz) with the trajectory data.
            seq_len (int): Length of sampled trajectory subsequence.
            norm (bool, optional): Whether to normalize the trajectories. Defaults to True.
            device (torch.device, optional): Which device to move the data to. Defaults to torch.device('cpu').
        """
        self.data_dir = data_dir
        self.datafile = datafile
        self.norm = norm
        self.pos_idx = pos_idx
        self.rot_idx = rot_idx
        self.record_freq = record_freq
        self.device = device

        # The trajectores and time stamps
        # Load the .npy file
        self.data = np.load(os.path.join(self.data_dir, self.datafile))
        pos_idx_list = [self.pos_idx[i] for i in ['x','y','z']]
        rot_idx_list = [self.rot_idx[i] for i in ['w','x','y','z']]
        self.pos = self.data[:,:,pos_idx_list]
        self.ori = self.data[:,:,rot_idx_list]

        # Find the number of sequences N and the length of each sequence T
        [self.num_demos, self.num_timesteps, self.pos_dim] = self.pos.shape
        [_, _, self.ori_dim] = self.ori.shape

        self.t = [np.linspace(0.0, self.num_timesteps, self.num_timesteps)/record_freq]

        # Normalize the position trajectories if needed
        # Convert to tensors and move to the required device
        if self.norm:
            self.pos, self.pos_mean, self.pos_std = self.normalize(self.pos)
            self.pos, self.pos_mean, self.pos_std = torch.from_numpy(self.pos), torch.from_numpy(self.pos_mean), torch.from_numpy(self.pos_std)
            self.pos, self.pos_mean, self.pos_std = self.pos.to(device), self.pos_mean.to(device), self.pos_std.to(device)
        else:
            self.pos_mean, self.pos_std = None, None
            self.pos = torch.from_numpy(self.pos)
            self.pos = self.pos.to(device)
        self.t = torch.from_numpy(np.array(self.t)).to(device)

        # Translating the goal to the origin
        self.pos_offset = self.pos[:,-1,:]

    def normalize(self, arr):
        """
        Normalizes the input array
        """
        # Compute the mean and std for x,y across all demonstration trajectories
        mean = np.expand_dims(np.mean(np.reshape(arr, (self.num_demos*self.num_timesteps, self.pos_dim)), axis=0), axis=0)
        std = np.expand_dims(np.std(np.reshape(arr, (self.num_demos*self.num_timesteps, self.pos_dim)), axis=0), axis=0)
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


class RobotTasksOrientationSimple():

    def __init__(self, data_dir, datafile, rot_idx={'w': 3, 'x': 4, 'y': 5, 'z': 6}, record_freq=60, device=torch.device('cpu'), scale=1.0):
        """
        Loads trajectories from the robot tasks dataset. 
        Can be used for trajectories of 2 or more dimensions.

        All the data is first moved to the GPU (if available).
        Moving the data to the GPU may be more efficient.
        """
        self.data_dir = data_dir
        self.datafile = datafile
        self.rot_idx = rot_idx
        self.record_freq = record_freq
        self.device = device
        # Multiply data by this factor to rescale it
        self.scale = scale

        # The trajectores and time stamps
        # Load the .npy file
        self.data = np.load(os.path.join(self.data_dir, self.datafile))
        rot_idx_list = [self.rot_idx[i] for i in ['w','x','y','z']]

        # self.pos actually stores the orientations
        # Ignore the wrong variable name
        self.pos = self.data[:,:,rot_idx_list]

        self.pos *= self.scale

        # Find the number of sequences N and the length of each sequence T
        [self.num_demos, self.num_timesteps, self.pos_dim] = self.pos.shape

        self.t = [np.linspace(0.0, self.num_timesteps, self.num_timesteps)/record_freq]

        self.pos = torch.from_numpy(self.pos)
        self.pos = self.pos.to(device)

        self.t = torch.from_numpy(np.array(self.t)).to(device)

    def unnormalize(self, arr, type='pos'):
        # Dummy method - in place to avoid errors from training script
        return arr


class RobotTasksOrientation():

    def __init__(self, data_dir, datafile, rot_idx={'w': 3, 'x': 4, 'y': 5, 'z': 6}, record_freq=60, device=torch.device('cpu'), scale=1.0):
        """
        All the data is first moved to the GPU (if available).
        Moving the data to the GPU may be more efficient.
        """
        self.data_dir = data_dir
        self.datafile = datafile
        self.rot_idx = rot_idx
        self.record_freq = record_freq
        self.device = device
        self.scale = scale

        # The trajectores and time stamps
        # Load the .npy file
        self.data = np.load(os.path.join(self.data_dir, self.datafile))
        rot_idx_list = [self.rot_idx[i] for i in ['w','x','y','z']]

        self.quat_data = self.data[:,:,rot_idx_list]

        # Find the number of sequences N and the length of each sequence T
        [self.num_demos, self.num_timesteps, self.quat_dim] = self.quat_data.shape

        # Must load only quaternions
        assert self.quat_dim == 4

        self.t = [np.linspace(0.0, self.num_timesteps, self.num_timesteps)/record_freq]

        # Project the quaternion trajectories to the tangent plane
        self.pos = torch.from_numpy(self.to_tangent_plane())
        # Scale the values if needed
        self.pos *= self.scale
        self.pos = self.pos.to(device)

        self.t = torch.from_numpy(np.array(self.t)).to(device)

    def to_tangent_plane(self):
        """
        Projects the demonstration quaternion trajectories to the Eucliden tangent plane
        """

        # The goal orientation
        q_goal = self.quat_data[:,-1,:]
        assert q_goal.shape == (self.num_demos, self.quat_dim)

        # Project quat_data to tangent plane using log map
        quat_data_projection = list()

        for demo in range(self.num_demos):
            # For each demo
            r_list = list()
            for step in range(self.num_timesteps):
                # Project a quaternion in each step
                q = Quaternion(q_goal[demo])
                p = Quaternion(self.quat_data[demo,step])

                r = Quaternion.log_map(q=q, p=p).elements

                # First value of r must be 0.0
                assert np.allclose(r[0], 0.0)

                # Remove the first element
                r = r[1:]
                r_list.append(r)
            quat_data_projection.append(r_list)

        quat_data_projection = np.array(quat_data_projection)

        # Check the shape
        assert quat_data_projection.shape == (self.num_demos, self.num_timesteps, 3)

        # The final point in the projected trajectory should be (0.0,0.0,0.0)
        # for each demonstration trajectory
        assert np.allclose(quat_data_projection[:,-1], np.zeros((self.num_demos, 3)))

        return quat_data_projection

    def from_tangent_plane(self, tangent_vector_data):
        """
        Projects the Eucliden tangent plane trajectories back to quaternion trajectories 
        """
        # The goal orientation
        q_goal = self.quat_data[:,-1,:]
        assert q_goal.shape == (self.num_demos, self.quat_dim)

        # Downscale if needed
        tangent_vector_data /= self.scale
        
        quat_data = list()
        for demo in range(self.num_demos):
            # For each demo
            q_list = list()
            for step in range(self.num_timesteps):
                q = Quaternion(q_goal[demo])
                r = tangent_vector_data[demo, step]
                assert r.shape == (3,)

                # r is 3-dimensional, insert 0.0 as the first element
                r = np.r_[np.zeros((1,)), r]
                assert r.shape == (4,)
                assert r[0] == 0.0
                r = Quaternion(r)
                
                try:
                    # Tangent vector projected back to quaternion
                    q_ = Quaternion.exp_map(q=q, eta=r)
                    # Enforce unit quaternions
                    if q_.norm > 0.0:
                        q_ /= q_.norm
                    q_ = q_.elements
                except Exception as e:
                    print(f'Exception: {e}')
                    print(f'q={q}')
                    print(f'r={r}')
                    print(f'q_={q_}')
                    q_ = q_list[-1]
                q_list.append(q_)

            quat_data.append(q_list)

        quat_data = np.array(quat_data)
        assert quat_data.shape == (self.num_demos, self.num_timesteps, 4)

        return quat_data

    def unnormalize(self, arr, type='pos'):
        # Dummy method - in place to avoid errors from training script
        return arr


class RobotTasksPosition():

    def __init__(self, data_dir, datafile, norm=True, pos_idx={'x': 0, 'y': 1, 'z': 2}, record_freq=60, device=torch.device('cpu')):
        """
        Loads position trajectories from the robot tasks dataset. 

        All the data is first moved to the GPU (if available).
        Moving the data to the GPU may be more efficient.

        Args:
            datafile (str): Path of the numpy archive (npz) with the trajectory data.
            seq_len (int): Length of sampled trajectory subsequence.
            norm (bool, optional): Whether to normalize the trajectories. Defaults to True.
            device (torch.device, optional): Which device to move the data to. Defaults to torch.device('cpu').
        """
        self.data_dir = data_dir
        self.datafile = datafile
        self.norm = norm
        self.pos_idx = pos_idx
        self.record_freq = record_freq
        self.device = device

        # The trajectores and time stamps
        # Load the .npy file
        self.data = np.load(os.path.join(self.data_dir, self.datafile))
        pos_idx_list = [self.pos_idx[i] for i in ['x','y','z']]
        self.pos = self.data[:,:,pos_idx_list]

        # Find the number of sequences N and the length of each sequence T
        [self.num_demos, self.num_timesteps, self.pos_dim] = self.pos.shape

        self.t = [np.linspace(0.0, self.num_timesteps, self.num_timesteps)/record_freq]

        # Normalize the position trajectories if needed
        # Convert to tensors and move to the required device
        if self.norm:
            self.pos, self.pos_mean, self.pos_std = self.normalize(self.pos)
            self.pos, self.pos_mean, self.pos_std = torch.from_numpy(self.pos), torch.from_numpy(self.pos_mean), torch.from_numpy(self.pos_std)
            self.pos, self.pos_mean, self.pos_std = self.pos.to(device), self.pos_mean.to(device), self.pos_std.to(device)
        else:
            self.pos_mean, self.pos_std = None, None
            self.pos = torch.from_numpy(self.pos)
            self.pos = self.pos.to(device)
        self.t = torch.from_numpy(np.array(self.t)).to(device)

        # Translating the goal to the origin
        self.pos_offset = self.pos[:,-1,:]

    def normalize(self, arr):
        """
        Normalizes the input array
        """
        # Compute the mean and std for x,y across all demonstration trajectories
        mean = np.expand_dims(np.mean(np.reshape(arr, (self.num_demos*self.num_timesteps, self.pos_dim)), axis=0), axis=0)
        std = np.expand_dims(np.std(np.reshape(arr, (self.num_demos*self.num_timesteps, self.pos_dim)), axis=0), axis=0)
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