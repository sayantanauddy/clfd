import numpy as np
from pyquaternion import Quaternion

def convert_to_unit_quat(quat_traj):
    """Converts a trajectory of quaternions to a trajectory of unit quaternions"""

    assert len(quat_traj.shape) == 3

    num_demos, num_steps, data_dim = quat_traj.shape

    for demo in range(num_demos):
        for step in range(num_steps):
            quat = Quaternion(quat_traj[demo,step])
            #if not np.allclose(quat.norm, 1.0):
            #    print(quat.norm)
            quat /= quat.norm
            quat_traj[demo,step] = quat.elements

    return quat_traj

def quat_traj_distance(quat_traj_true, quat_traj_pred):
    """Finds the difference between 2 sets of quaternion trajectories of equal length"""
    num_demos, num_steps, data_dim = quat_traj_true.shape

    # Ensure unit quaternions
    quat_traj_true = convert_to_unit_quat(quat_traj_true)
    quat_traj_pred = convert_to_unit_quat(quat_traj_pred)

    metric_errors = list()

    for demo in range(num_demos):
        demo_errors = list()
        for step in range(num_steps):
            err = quat_distance(quat_traj_true[demo, step], quat_traj_pred[demo, step])
            demo_errors.append(err)

        # Mean error for the whole trajectory
        metric_errors.append(np.mean(demo_errors))

    metric_errors = np.array(metric_errors)

    # metric_errors should be a list with num_demos elements, 1 for each trajectory
    return np.median(metric_errors), metric_errors


def quat_distance(quat_true, quat_pred):
    """ Finds the difference between 2 quaternions"""
    # The elements [a, b, c, d] of the array correspond the the real, and each 
    # imaginary component respectively in the order a + bi + cj + dk.

    # The distance is computed as norm_2(r)
    # where r = 2*log(q_true * conjugate(quat_pred))
    # where log is the log map
    # This metric is defined in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907291
    
    quat_true = Quaternion(quat_true)
    quat_pred = Quaternion(quat_pred)

    quat_pred_conj = quat_pred.conjugate

    quat_prod = quat_true * quat_pred_conj

    scalar_quat_prod = quat_prod.scalar
    vector_quat_prod = quat_prod.vector
    vector_quat_prod_norm = np.linalg.norm(vector_quat_prod)

    # Round off scalar_quat_prod to take care of errors due to numerical errors
    scalar_quat_prod = np.round(scalar_quat_prod, 4)

    if scalar_quat_prod < -1.0 or scalar_quat_prod > 1.0: 
        print(f'scalar_quat_prod={scalar_quat_prod}')
        
    log_quat_prod = np.arccos(scalar_quat_prod)*(vector_quat_prod/vector_quat_prod_norm) if vector_quat_prod_norm > 0 else np.array([0.0,0.0,0.0])

    return np.linalg.norm(log_quat_prod)