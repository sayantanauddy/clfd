"""
Adapted from https://github.com/TheCamusean/iflow/blob/main/iflow/test_measures/trajectory_metrics.py
"""

import numpy as np
import similaritymeasures
from numba import jit


def seds_metric(vel_ref_trj_l, vel_pred_tr_l, r=0.6, q=0.4, epsilon = 0.0001):

    n_trj = len(vel_ref_trj_l)

    error = 0
    for vel_pred_tr, vel_ref_trj in zip(vel_pred_tr_l,vel_ref_trj_l):


        total_value = 0
        for i in range(0,vel_pred_tr.shape[0]):
            norm_pred = np.sum(np.abs(vel_pred_tr[i,0]))
            norm_real = np.sum(np.abs(vel_ref_trj[i,0]))

            vel_p = vel_pred_tr[i,:]
            vel_r = vel_ref_trj[i,:]

            dist_x = vel_r - vel_p
            dist_mean = np.matmul(dist_x.T,dist_x)

            q_value = dist_mean/(norm_pred*norm_real+epsilon)

            dist_ang = np.matmul(vel_r.T,vel_p)
            r_value = (1 - dist_ang/((norm_pred*norm_real+epsilon)))**2
            value_t = r*r_value + q*q_value

            total_value += np.sqrt(value_t)

        total_value_n = total_value/vel_pred_tr.shape[0]
        error += total_value_n

    error_n = error/ n_trj
    return error_n


def squared_mean_error(ref_trj_l, pred_tr_l):

    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):

        l2_norm_tr = 0
        length = pred_tr.shape[0]
        for i in range(0,length):
            x_pred = pred_tr[i,:]
            x_real = ref_trj[i,:]

            dist = x_pred - x_real
            l2_norm = np.linalg.norm(dist)
            l2_norm_tr += l2_norm

        l2_norm_tr_n = l2_norm_tr/length

        error += l2_norm_tr_n

    error_n = error/ n_trj
    return error_n


def area(X):
    n_points = len(X)

    d = 0
    for i in range(0,n_points):
        x0 = X[i]
        if i== (n_points-1):
            x1 = X[0]
        else:
            x1 = X[i+1]

        d_point = x0[0]*x1[1] - x0[1]*x1[0]
        d += d_point

    A = np.abs(d)/2
    return A


def mean_swept_error(ref_trj_l ,pred_tr_l, eval_mode='mean'):
    n_trj = len(pred_tr_l)

    errors = np.zeros(n_trj)

    idx = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        ar = 0
        lenght = ref_trj.shape[0]
        for i in range(0,lenght-1):
            X = []
            X.append(pred_tr[i, :])
            X.append(pred_tr[i + 1, :])
            X.append(ref_trj[i+1,:])
            X.append(ref_trj[i,:])

            A = area(X)
            ar += A

        errors[idx] = ar
        idx += 1

    error = 0
    if eval_mode == 'mean':
        error = np.mean(errors)
    elif eval_mode == 'median':
        error = np.median(errors)
    else:
        raise NotImplementedError(f'Unknown eval_mode')
    return error, errors


def area_between_error(ref_trj_l, pred_tr_l):
    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        area = similaritymeasures.area_between_two_curves(ref_trj,pred_tr)
        error += area
    error_n = error/n_trj
    return error_n


def mean_frechet_error(ref_trj_l, pred_tr_l):
    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        frecht_d = similaritymeasures.frechet_dist(ref_trj,pred_tr)
        error += frecht_d
    error_n = error/n_trj
    return error_n

def dtw_distance(ref_trj_l, pred_tr_l):
    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        dtw_dist, d = similaritymeasures.dtw(ref_trj,pred_tr)
        error+= dtw_dist
    error_n = error/n_trj
    return error_n

############### Optimized metrics ###############

## Frechet Distance

@jit(nopython=True)
def frechet_loop(traj_A, traj_B):

    dist = lambda a,b: np.linalg.norm(a-b)

    p = traj_A
    q = traj_B

    n_p = p.shape[0]
    n_q = q.shape[0]

    arr = np.zeros((n_p, n_q))

    for i in range(n_p):
        for j in range(n_q):

            d = dist(p[i], q[j])

            if i>0 and j>0:
                arr[i,j] = max(min(arr[i-1,j], 
                                   arr[i-1,j-1],
                                   arr[i,j-1]), 
                               d)
            elif i==0 and j>0:
                arr[i,j] = max(arr[i,j-1], d)
            elif i>0 and j==0:
                arr[i,j] = max(arr[i-1,j], d)
            else:
                arr[i,j] = d

    return arr[-1,-1]

@jit(nopython=True)
def mean_frechet_error_fast(ref_trj_l, pred_tr_l, eval_mode='mean'):
    n_trj = len(pred_tr_l)

    errors = np.zeros(ref_trj_l.shape[0])

    i = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        frecht_d = frechet_loop(ref_trj,pred_tr)
        errors[i] = frecht_d
        i += 1

    error = 0
    if eval_mode == 'mean':
        error = np.mean(errors)
    elif eval_mode == 'median':
        error = np.median(errors)
    else:
        raise NotImplementedError(f'Unknown eval_mode')
    return error, errors

## Dynamic Time Warping 

@jit(nopython=True)
def dtw(traj_A, traj_B):

    dist = lambda a,b: np.linalg.norm(a-b)

    n, m = len(traj_A), len(traj_B)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist(traj_A[i-1], traj_B[j-1])
            # take last min from a square box
            last_min = min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[-1,-1]

@jit(nopython=True)
def dtw_distance_fast(ref_trj_l, pred_tr_l, eval_mode='mean'):
    dists = np.zeros(ref_trj_l.shape[0])
    for i in range(ref_trj_l.shape[0]):
        distance = dtw(ref_trj_l[i], pred_tr_l[i])
        dists[i] = distance

    dist = 0
    if eval_mode == 'mean':
        dist = np.mean(dists)
    elif eval_mode == 'median':
        dist = np.median(dists)
    else:
        raise NotImplementedError(f'Unknown eval_mode')
    return dist, dists
