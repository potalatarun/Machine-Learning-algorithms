import numpy as np
from typing import List, Tuple
import torch

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting
    
    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """
    # Your implementation here
    # samples = torch.rand(n_samples) 
    samples = list(range(0,n_samples))
#    print(samples)
    
    size_of_splits = n_samples//k 
    first_split_size = size_of_splits + (n_samples % k)
    splits = []
    temp = 0
    for i in range(k-1):
        if i==0:
            splits.append(samples[temp: first_split_size])
            temp = first_split_size
        splits.append(samples[temp:temp+size_of_splits])
        temp += size_of_splits
#        print(splits)
#    print(splits)
    ans = []
    for i in range(k):
        
        temp = splits[i]
        temp2 = splits[0:i]
        temp3 = splits[i+1:]
        #print(len(temp), len(temp2+temp3))
        ans.append((temp2+temp3, temp))
    return ans
    pass

#print(k_fold_cross_validation(n_samples= 11, k = 5, shuffle=True))
