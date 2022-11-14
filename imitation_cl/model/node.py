
import torch
import torch.nn as nn
from torchdiffeq import odeint
from tqdm import trange

def integrate(ode_rhs,x0,t,rtol=1e-6,atol=1e-7, method='dopri5'):
    ''' Performs forward integration with Dopri5 (RK45) solver with rtol=1e-6 & atol=1e-7.
        Higher-order solvers as well as smaller tolerances would give more accurate solutions.
        Inputs:
            ode_rhs    time differential function with signature ode_rhs(t,s)
            x0 - [N,d] initial values
            t  - [T]   integration time points
        Retuns:
            xt - [N,T,d] state trajectory computed at t
    '''
    return odeint(ode_rhs, x0, t, method=method, rtol=rtol, atol=atol).permute(1,0,2)

class NODE(nn.Module):
    def __init__(self, target_network, explicit_time=0, method='dopri5'):
        ''' d - ODE dimensionality '''
        super().__init__()
        self.set_target_network(target_network)
        self.explicit_time = explicit_time
        self.method = method

    def set_target_network(self, target_network):
        self.target_network = target_network
    
    @property
    def ode_rhs(self):
        ''' returns the differential function '''
        if self.explicit_time == 1:
            return lambda t,x: self.target_network(torch.cat([x, t.repeat(*(list(x.shape[0:-1])+[1]))], dim=-1))
        elif self.explicit_time == 0:
            return lambda t,x: self.target_network(x)
        else:
            raise NotImplementedError(f'Invalid value of explicit_time={self.explicit_time} (only 0 or 1 allowed)')
    
    def forward(self, t, x0):
        ''' Forward integrates the NODE system and returns state solutions
            Input
                t  - [T]   time points
                x0 - [N,d] initial value
            Returns
                X  - [N,T,d] forward simulated states
        '''
        return integrate(self.ode_rhs, x0, t, method=self.method).float()

class NODETaskEmbedding(NODE):
    def __init__(self, target_network, te_dim, explicit_time=0, method='dopri5'):
        ''' d - ODE dimensionality '''
        super().__init__(target_network, explicit_time, method=method)

        # Empty parameter list of task embeddings
        # Before learning each task, an embedding is
        # to be created for that task
        self.task_embs = nn.ParameterList()
        self.te_dim = te_dim
        self.task_id = None

    def set_task_id(self, task_id):
        self.task_id = task_id

    def gen_new_task_emb(self):
        """
        Creates a new task embedding before learning a task
        """
        self.task_embs.append(nn.Parameter(data=torch.Tensor(self.te_dim), requires_grad=True))
        torch.nn.init.normal_(self.task_embs[-1], mean=0., std=1.)
        
    def get_task_embs(self):
        """Return a list of all task embeddings.
        Returns:
            A list of Parameter tensors.
        """
        return self.task_embs

    def get_task_emb(self, task_id):
        """Return the task embedding corresponding to a given task id.
        Args:
            task_id: Determines the task for which the embedding should be
                returned.
        Returns:
            A list of Parameter tensors.
        """
        return self.task_embs[task_id]

    def set_target_network(self, target_network):
        self.target_network = target_network

    def embedded_forward(self, x):
        """
        Forward function using the task embedding vector as an additional input

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Get the task embedding for the current task
        task_emb = self.get_task_emb(self.task_id)

        # Find the dimensions of the input
        # This is used to find how much the task embedding
        # should be repeated so that the input and the
        # task embedding can be concated in the last dimension
        repeat_shape = list(x.shape)
        repeat_shape[-1] = 1

        # Repeat the task embedding
        task_emb_repeat = task_emb.repeat(*repeat_shape)

        # Stack the repeated task embedding with the input
        x_stacked = torch.cat((task_emb_repeat, x), dim=-1)

        # Forward the stacked input through the network
        return self.target_network(x_stacked)
    
    @property
    def ode_rhs(self):
        ''' returns the differential function '''        
        if self.explicit_time == 1:
            return lambda t,x: self.embedded_forward(torch.cat([x, t.repeat(*(list(x.shape[0:-1])+[1]))], dim=-1))
        elif self.explicit_time == 0:
            return lambda t,x: self.embedded_forward(x)
        else:
            raise NotImplementedError(f'Invalid value of explicit_time={self.explicit_time} (only 0 or 1 allowed)')


class SINODETaskEmbedding(NODETaskEmbedding):
    def __init__(self, target_network, te_dim, si_c, si_epsilon, explicit_time=0, method='dopri5'):
        super().__init__(target_network, te_dim, explicit_time, method)

        # Set the SI hyperparameters
        self.si_c = si_c           
        self.si_epsilon = si_epsilon

        # Setting buffers
        # Register starting parameter values
        for n, p in self.target_network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer(f'{n}_SI_prev_task', p.data.detach().clone())
        
        # Prepare buffers to store running importance estimates
        # and param-values before update
        for n, p in self.target_network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer(f'W_{n}', p.data.detach().clone().zero_())
                self.register_buffer(f'p_old_{n}', p.data.detach().clone())

    def check(self):

        for n,p in self.named_buffers():
            print(n,p.sum().item())

        for n,p in self.target_network.named_parameters():
            print(n,p.sum().item())


    def update_omega(self):
        """
        Updates the per-parameter importance once training is complete for a task.
        Adapted from https://github.com/GMvandeVen/continual-learning 
        (using buffers instead of Python variables)           
        """
        # Loop over all parameters
        for n, p in self.target_network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, f'{n}_SI_prev_task')
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                W_n = getattr(self, f'W_{n}')
                omega_add = W_n/(p_change**2 + self.si_epsilon)
                try:
                    omega = getattr(self, f'{n}_SI_omega')
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer(f'{n}_SI_prev_task', p_current)
                self.register_buffer(f'{n}_SI_omega', omega_new)

    def update_running_importance(self):

        # Update W (running importance estimate) after each iteration
        for n, p in self.target_network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if p.grad is not None:
                    W_n = getattr(self, f'W_{n}')
                    p_old_n = getattr(self, f'p_old_{n}')
                    W_n.add_(-p.grad*(p.detach()-p_old_n))
                    self.register_buffer(f'W_{n}', W_n)
                p_old_n = p.detach().clone()
                self.register_buffer(f'p_old_{n}', p_old_n)

    def surrogate_loss(self):
        """
        Compute the surrogate loss in SI using the per-parameter importance and
        the current and previous parameter values.
        
        Adapted from https://github.com/GMvandeVen/continual-learning 
        (using buffers instead of Python variables)
        """
        try:
            losses = []
            for n, p in self.target_network.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, f'{n}_SI_prev_task')
                    omega = getattr(self, f'{n}_SI_omega')
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0.)

class MASNODETaskEmbedding(NODETaskEmbedding):
    def __init__(self, target_network, te_dim, mas_lambda, explicit_time=0, method='dopri5'):
        super().__init__(target_network, te_dim, explicit_time, method)

        # Set the MAS hyperparameter
        self.mas_lambda = mas_lambda           

        # Setting buffers
        # Register starting parameter values
        for n, p in self.target_network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer(f'{n}_theta_prev', p.data.detach().clone().zero_())
                self.register_buffer(f'{n}_omega_curr', p.data.detach().clone().zero_())
                self.register_buffer(f'{n}_omega_prev', p.data.detach().clone().zero_())
        
    def update_omega(self, args, data, get_minibatch_fn, device):
        """
        Updates the per-parameter importance once training is complete for a task.
        
        Here, we compute the per-sample gradient by iterating over a dataloader
        with a bach size of 1, but this is painfully slow. 
        More efficient implements are possible. For example, 
        see: [BackPACK](https://github.com/f-dangel/backpack). 
        
        Adapted from https://github.com/GMvandeVen/continual-learning 
        (using buffers instead of Python variables)           
        """

        # Set the target network to eval mode
        self.target_network.eval()

        # Start training iterations
        for training_iters in trange(args.num_iter):

            # Reset all gradients of the model
            self.target_network.zero_grad()

            # Get a mini batch with batch_size=1
            t, y_all = get_minibatch_fn(data.t[0], data.pos, nsub=2, tsub=args.tsub)

            # At this point y_all has the shape [7, tsub, 2]
            # Since we need per sample gradients, we need to iterate
            # over the first dimension of y_all
            y_all_shape = list(y_all.shape)

            for sample_idx in range(y_all_shape[0]):

                y_all_ = y_all[sample_idx, :, :]

                # Make the number of dimensions same as y_all
                y_all_ = y_all_.unsqueeze(0)

                # The time steps
                t = t.to(device).float()

                # Subsequence trajectories
                y_all_ = y_all_.to(device)

                # Starting points
                y_start = y_all_[:,0].float()
                y_start.requires_grad = True

                # Compute the output of the target network
                # Using the last time stamp to find the sensitivity of the target network's output
                # TODO Check if we need to iterate over all time steps as well
                if self.explicit_time == 1:
                    target_net_output = self.embedded_forward(torch.cat([y_start, t[-1].repeat(*(list(y_start.shape[0:-1])+[1]))], dim=-1))
                elif self.explicit_time == 0:
                    target_net_output = self.embedded_forward(y_start)
                else:
                    raise NotImplementedError(f'Invalid value of explicit_time={self.explicit_time} (only 0 or 1 allowed)')

                # L2 norm of the output
                l2_norm = torch.norm(target_net_output, p=2, dim=1)
                
                # Square the L2 norm and sum
                sum_sq_l2_norm = torch.sum(l2_norm**2)

                # Compute the gradients (backward pass)
                sum_sq_l2_norm.backward()

                # Loop over all parameters
                for n, p in self.target_network.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        
                        # Retrieve current omega for this parameter
                        try:
                            omega_curr = getattr(self, f'{n}_omega_curr')
                        except AttributeError:
                            omega_curr = p.detach().clone().zero_()
                        
                        # Retrieve the absolute current gradient for this parameter
                        p_abs_grad = torch.abs(p.grad.detach().clone())
                                            
                        # Update the current omega for this parameter
                        # The gradient is normalized by the number of batches (of size 1)
                        omega_curr = torch.add(omega_curr, p_abs_grad, alpha=1.0/(args.num_iter*y_all_shape[0]))

                        # Update the current omega in the buffer
                        self.register_buffer(f'{n}_omega_curr', omega_curr)

    def merge_omega(self):     
        """
        Once omega for the current task has been computed, merge this 
        with the omega computed for the previous tasks
        """ 
        
        # Loop over all parameters
        for n, p in self.target_network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Retrieve previous omega for this parameter
                try:
                    omega_prev = getattr(self, f'{n}_omega_prev')
                except AttributeError:
                    omega_prev = p.detach().clone().zero_()

                # Retrieve current omega for this parameter
                omega_curr = getattr(self, f'{n}_omega_curr')

                # Update the current omega
                omega_curr = torch.add(omega_prev, omega_curr)
                self.register_buffer(f'{n}_omega_prev', omega_curr)

    def update_theta_prev(self):
        """
        Once training is over for the current task, the current
        parameter values need to be saved
        """
        for n, p in self.target_network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer(f'{n}_theta_prev', p.data.detach().clone())
                
    def regularization_loss(self):
        """
        Computes the regularization loss in MAS using the per-parameter importance and
        the current and previous parameter values.
        """
        try:
            losses = []
            for n, p in self.target_network.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and the corresponding omega
                    n = n.replace('.', '__')
                    prev_values = getattr(self, f'{n}_theta_prev')
                    omega_prev = getattr(self, f'{n}_omega_prev')
                    # Calculate MAS's surrogate loss, sum over all parameters
                    losses.append((omega_prev * (p-prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # Regularization loss is 0 if there is no stored omega yet
            return torch.tensor(0.)