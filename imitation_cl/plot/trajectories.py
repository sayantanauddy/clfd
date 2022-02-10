import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

#TODO remove
def get_quiver_data(t, y_all, ode_rhs, y_hat, x_lim=[-2.0,2.0], y_lim=[-2.0,2.0], L=1):

    N_ = 10

    min_x,min_y = x_lim[0], y_lim[0]
    max_x,max_y = x_lim[1], y_lim[1]

    t = t.detach().cpu().numpy()

    xs1_,xs2_ = np.meshgrid(np.linspace(min_x, max_x, N_),np.linspace(min_y, max_y, N_))
    Z = np.array([xs1_.T.flatten(), xs2_.T.flatten()]).T
    Z = torch.from_numpy(Z).float().to(y_all.device)
    Z = torch.stack([Z]*L)
    F = ode_rhs(None,Z).detach().cpu().numpy()
    F /= ((F**2).sum(-1,keepdims=True))**(0.25)
    Z  = Z.detach().cpu().numpy()
    y_all = y_all.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()

    return (F, Z, xs1_, xs2_, N_, t, y_all, y_hat, x_lim, y_lim)

#TODO remove
def plot_ode_simple2(t, y_all, F, Z, xs1_, xs2_, N_, y_hat, L=1, ax=None, fontsize=10, x_lim=[-2.0,2.0], y_lim=[-2.0,2.0]):
    """[summary]

    Args:
        t (torch.Tensor): Time steps
        y_all (torch.Tensor): Demonstrated trajectories
        ode_rhs (function): ODE RHS of the NODE
        y_hat (torch.Tensor, optional): Predicted trajectories. Defaults to None.
        L (int, optional): [description]. Defaults to 1.
        return_fig (bool, optional): Whether to return the fig. Defaults to False.

    Returns:
        [type]: [description]
    """
    linewidth = 1.5
    markersize = 5.0
    alpha = 0.7

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('State $x_1$',fontsize=fontsize)
    ax.set_ylabel('State $x_2$',fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    for F_ in F:
        h1 = ax.quiver(xs1_, xs2_, 
                       F_[:,0].reshape(N_,N_).T, F_[:,1].reshape(N_,N_).T, 
                       cmap=plt.cm.Blues)

    if y_hat is None: # only plotting data
        for y_ in y_all:
            h2, = ax.plot(y_[0,0],
                          y_[0,1],
                         'o', 
                         fillstyle='none',
                         markersize=markersize, 
                         linewidth=linewidth)
            h3, = ax.plot(y_[:,0],
                          y_[:,1],
                          '-',
                          color=h2.get_color(),
                          linewidth=linewidth)

    else: # plotting data and fits, set the color correctly!
        # The initial position
        h2, = ax.plot(y_all[:,0,0],
                      y_all[:,0,1],
                      'o',
                      color='firebrick', 
                      fillstyle='none',
                      markersize=markersize, 
                      linewidth=linewidth)
        # The demonstrations
        num_demos = y_all.shape[0]
        for demo_idx in range(num_demos):
            h3, = ax.plot(y_all[demo_idx,:,0],
                          y_all[demo_idx,:,1],
                          '-',color='firebrick',
                          alpha=alpha,
                          linewidth=linewidth)

    
    for y_hat_ in y_hat:
        h4, = ax.plot(y_hat_[:,0],
                        y_hat_[:,1],
                        '-',
                        color='royalblue',
                        alpha=alpha,
                        linewidth=linewidth)
    if y_hat.shape[0]>1:
        ax.plot(y_all[0,:,0],
                y_all[0,:,1],
                '-',
                color='firebrick',
                alpha=alpha,
                linewidth=linewidth)

    # Return handles for creating legend
    return [h1,h2,h3,h4], ['Vector field','Initial value','Demonstration', 'Prediction']
    

def plot_ode_simple(t, y_all, ode_rhs, y_hat=None, L=1, ax=None, fontsize=10, explicit_time=0):
    """[summary]

    Args:
        t (torch.Tensor): Time steps
        y_all (torch.Tensor): Demonstrated trajectories
        ode_rhs (function): ODE RHS of the NODE
        y_hat (torch.Tensor, optional): Predicted trajectories. Defaults to None.
        L (int, optional): [description]. Defaults to 1.
        return_fig (bool, optional): Whether to return the fig. Defaults to False.

    Returns:
        [type]: [description]
    """
    N_ = 10

    linewidth = 1.0
    markersize = 5.0
    alpha = 0.7

    # To show the vector field, we need to evaluate the ODE at
    # different starting points
    min_x,min_y = y_all.min(dim=0)[0].min(dim=0)[0].detach().cpu().numpy()
    max_x,max_y = y_all.max(dim=0)[0].max(dim=0)[0].detach().cpu().numpy()

    limit = 3.0
    min_x,min_y = -limit, -limit
    max_x,max_y = limit, limit

    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    xs1_,xs2_ = np.meshgrid(np.linspace(min_x, max_x, N_),np.linspace(min_y, max_y, N_))
    Z = np.array([xs1_.T.flatten(), xs2_.T.flatten()]).T
    Z = torch.from_numpy(Z).float().to(y_all.device)
    Z = torch.stack([Z]*L)

    if explicit_time == 1:
        # Use the last time stamp for the vector field
        F = ode_rhs(t[-1],Z).detach().cpu().numpy()
    elif explicit_time == 0:
        F = ode_rhs(None,Z).detach().cpu().numpy()
    else:
        raise NotImplementedError(f'Invalid value of explicit_time={explicit_time} (only 0 or 1 allowed)')

    F /= ((F**2).sum(-1,keepdims=True))**(0.25)
    Z  = Z.detach().cpu().numpy()

    t = t.detach().cpu().numpy()
    y_all = y_all.detach().cpu().numpy()

    ax.set_xlabel('State $x_1$',fontsize=fontsize)
    ax.set_ylabel('State $x_2$',fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    for F_ in F:
        h1 = ax.quiver(xs1_, xs2_, 
                       F_[:,0].reshape(N_,N_).T, F_[:,1].reshape(N_,N_).T, 
                       cmap=plt.cm.Blues)

    if y_hat is None: # only plotting data
        for y_ in y_all:
            h2, = ax.plot(y_[0,0],
                          y_[0,1],
                         'o', 
                         fillstyle='none',
                         markersize=markersize, 
                         linewidth=linewidth)
            h3, = ax.plot(y_[:,0],
                          y_[:,1],
                          '-',
                          color=h2.get_color(),
                          linewidth=linewidth)

    else: # plotting data and fits, set the color correctly!
        # The initial position
        h2, = ax.plot(y_all[:,0,0],
                      y_all[:,0,1],
                      'o',
                      color='firebrick', 
                      fillstyle='none',
                      markersize=markersize, 
                      linewidth=linewidth)
        # The demonstrations
        num_demos = y_all.shape[0]
        for demo_idx in range(num_demos):
            h3, = ax.plot(y_all[demo_idx,:,0],
                          y_all[demo_idx,:,1],
                          '-',color='firebrick',
                          alpha=alpha,
                          linewidth=linewidth)

    if y_hat is None:
        #ax.set_aspect('equal')

        # Return handles for creating legend
        # Quiver plot legend does not work as expected, the below line fixes this
        h1 = ax.scatter([],[],marker=r'$\rightarrow$', label='Vector Field', color='black', s=100)
        return [h1,h2,h3], ['Vector field','Initial value','Demonstration']
    else:
        #ax.set_aspect('equal')
        y_hat = y_hat.detach().cpu()
        for y_hat_ in y_hat:
            h4, = ax.plot(y_hat_[:,0],
                          y_hat_[:,1],
                          '-',
                          color='royalblue',
                          alpha=alpha,
                          linewidth=linewidth)
        if y_hat.shape[0]>1:
            ax.plot(y_all[0,:,0],
                    y_all[0,:,1],
                    '-',
                    color='firebrick',
                    alpha=alpha,
                    linewidth=linewidth)

        # Return handles for creating legend
        # Quiver plot legend does not work as expected, the below line fixes this
        h1 = ax.scatter([],[],marker=r'$\leftarrow$', label='Vector Field', color='black', s=100)
        return [h1,h2,h3,h4], ['Vector field','Initial value','Demonstration', 'Prediction']

    
    
def plot_ode(t, X, ode_rhs, Xhat=None, L=1, return_fig=False):
    print(t.shape, X.shape, Xhat.shape)
    N_ = 10
    min_x,min_y = X.min(dim=0)[0].min(dim=0)[0].detach().cpu().numpy()
    max_x,max_y = X.max(dim=0)[0].max(dim=0)[0].detach().cpu().numpy()
    xs1_,xs2_ = np.meshgrid(np.linspace(min_x, max_x, N_),np.linspace(min_y, max_y, N_))
    Z = np.array([xs1_.T.flatten(), xs2_.T.flatten()]).T
    Z = torch.from_numpy(Z).float().to(X.device)
    Z = torch.stack([Z]*L)
    F = ode_rhs(t[-1],Z).detach().cpu().numpy()
    F /= ((F**2).sum(-1,keepdims=True))**(0.25)
    Z  = Z.detach().cpu().numpy()

    t = t.detach().cpu().numpy()
    X = X.detach().cpu().numpy()
    fig = plt.figure(1,[15,7.5],constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[:, 0])

    ax1.set_xlabel('State $x_1$',fontsize=17)
    ax1.set_ylabel('State $x_2$',fontsize=17)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    for F_ in F:
        h1 = ax1.quiver(xs1_, xs2_, F_[:,0].reshape(N_,N_).T, F_[:,1].reshape(N_,N_).T, \
                    cmap=plt.cm.Blues)
    if Xhat is None: # only plotting data
        for X_ in X:
            h2, = ax1.plot(X_[0,0],X_[0,1],'o', fillstyle='none', \
                     markersize=11.0, linewidth=2.0)
            h3, = ax1.plot(X_[:,0],X_[:,1],'-',color=h2.get_color(),linewidth=3.0)
    else: # plotting data and fits, set the color correctly!
        h2, = ax1.plot(X[0,0,0],X[0,0,1],'o',color='firebrick', fillstyle='none', \
                 markersize=11.0, linewidth=2.0)
        h3, = ax1.plot(X[0,:,0],X[0,:,1],'-',color='firebrick',linewidth=3.0)
    if Xhat is not None and Xhat.ndim==3:
        Xhat = Xhat.unsqueeze(0)
    if Xhat is None:
        plt.legend([h1,h2,h3],['Vector field','Initial value','State trajectory'],
            loc='lower right', fontsize=20, bbox_to_anchor=(1.5, 0.05))
    else:
        Xhat = Xhat.detach().cpu()
        for xhat in Xhat:
            h4, = ax1.plot(xhat[0,:,0],xhat[0,:,1],'-',color='royalblue',linewidth=3.0)
        if Xhat.shape[0]>1:
            ax1.plot(X[0,:,0],X[0,:,1],'-',color='firebrick',linewidth=5.0)
        plt.legend([h1,h2,h3,h4],['Vector field','Initial value','Data sequence', 'Forward simulation'],
            loc='lower right', fontsize=20, bbox_to_anchor=(1.5, 0.05))

    ax2 = fig.add_subplot(gs[0, 1:])
    if Xhat is None: # only plotting data
        for X_ in X:
            h4, = ax2.plot(t,X_[:,0],linewidth=3.0)
    else: # plotting data and fits, set the color correctly!
        h4, = ax2.plot(t,X[0,:,0],color='firebrick',linewidth=3.0)
    if Xhat is not None:
        for xhat in Xhat:
            ax2.plot(t,xhat[0,:,0],color='royalblue',linewidth=3.0)
        if Xhat.shape[0]>1:
            ax2.plot(t,X[0,:,0],color='firebrick',linewidth=5.0)
    ax2.set_xlabel('time',fontsize=17)
    ax2.set_ylabel('State $x_1$',fontsize=17)

    ax3 = fig.add_subplot(gs[1, 1:])
    
    if Xhat is None: # only plotting data
        for X_ in X:
            h5, = ax3.plot(t,X_[:,1],linewidth=3.0)
    else: # plotting data and fits, set the color correctly!
        h5, = ax3.plot(t,X[0,:,1],color='firebrick',linewidth=3.0)
    if Xhat is not None:
        for xhat in Xhat:
            ax3.plot(t,xhat[0,:,1],color='royalblue',linewidth=3.0)
        if Xhat.shape[0]>1:
            ax3.plot(t,X[0,:,1],color='firebrick',linewidth=5.0)
    ax3.set_xlabel('time',fontsize=17)
    ax3.set_ylabel('State $x_2$',fontsize=17)
    
    if return_fig:
        return fig,ax1,h3,h4,h5
    else:
        import uuid
        filename = str(uuid.uuid4())
        #plt.savefig(filename)