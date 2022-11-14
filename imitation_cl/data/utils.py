import torch

def get_minibatch(t, y, nsub=None, tsub=None, dtype=torch.float64):
    """
    Extract nsub sequences each of lenth tsub from the original dataset y.

    Args:
        t (np array [T]): T integration time points from the original dataset.
        y (np array [N,T,d]): N observed sequences from the original dataset, 
                              each with T datapoints where d is the dimension 
                              of each datapoint.
        nsub (int): Number of sequences to be selected from.
                    If Nsub is None, then all N sequences are considered.
        tsub (int): Length of sequences to be returned.
                    If tsub is None, then sequences of length T are returned.

    Returns:
        tsub (torch tensor [tsub]): Integration time points (in this minibatch)
        ysub (torch tensor [nsub, tsub, d]): Observed (sub)sequences (in this minibatch)
    """
    # Find the number of sequences N and the length of each sequence T
    [N,T] = y.shape[:2]

    # If nsub is None, then consider all sequences
    # Else select nsub sequences randomly
    y_   = y if nsub is None else y[torch.randperm(N)[:nsub]]

    # Choose the starting point of the sequences
    # If tsub is None, then start from the beginning
    # Else find a random starting point based on tsub
    t0   = 0 if tsub is None else torch.randint(0,1+len(t)-tsub,[1]).item()  # pick the initial value
    tsub = T if tsub is None else tsub

    # Set the data to be returned
    tsub, ysub = torch.from_numpy(t[t0:t0+tsub]).type(dtype), torch.from_numpy(y_[:,t0:t0+tsub]).type(dtype)
    return tsub, ysub 

def get_minibatch_extended(t, y, nsub=None, tsub=None, dtype=torch.float64):
    """
    Extract nsub sequences each of lenth tsub from the original dataset y.

    Args:
        t (np array [T]): T integration time points from the original dataset.
        y (np array [N,T,d]): N observed sequences from the original dataset, 
                              each with T datapoints where d is the dimension 
                              of each datapoint.
        nsub (int): Number of sequences to be selected from.
                    If Nsub is None, then all N sequences are considered.
        tsub (int): Length of sequences to be returned.
                    If tsub is None, then sequences of length T are returned.

    Returns:
        tsub (torch tensor [tsub]): Integration time points (in this minibatch)
        ysub (torch tensor [nsub, tsub, d]): Observed (sub)sequences (in this minibatch)
    """
    # Find the number of sequences N and the length of each sequence T
    [N,T] = y.shape[:2]

    # If nsub is None, then consider all sequences
    # Else select nsub sequences randomly
    y_   = y if nsub is None else y[torch.randperm(N)[:nsub]]

    # Choose the starting point of the sequences
    # If tsub is None, then start from the beginning
    # Else find a random starting point based on tsub
    t0   = 0 if tsub is None else torch.randint(0,1+len(t)-tsub,[1]).item()  # pick the initial value
    tsub = T if tsub is None else tsub

    # Set the data to be returned
    tsub, ysub = t[t0:t0+tsub], y_[:,t0:t0+tsub]

    if not torch.is_tensor(tsub):
        tsub = torch.from_numpy(tsub)

    if not torch.is_tensor(ysub):
        ysub = torch.from_numpy(ysub)

    return tsub, ysub 