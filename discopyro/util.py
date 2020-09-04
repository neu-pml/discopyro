import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_softmax(logprobs, dim1=-2, dim2=-1):
    logprobs = list(logprobs.unbind(dim=dim1))
    for i, logprob in enumerate(logprobs):
        logprobs[i] = F.softmax(logprob, dim2)
    return torch.stack(logprobs, dim=dim1)

def optional_log(tensor):
    nonzeros = tensor.nonzero(as_tuple=True)
    return torch.index_put(tensor, nonzeros, torch.log(tensor[nonzeros]))

def optional_energies(probs):
    return -optional_log(probs)

def soften_logprobs(energies, temperature, dim1=-2, dim2=-1):
    if dim2 is None:
        return F.softmax(energies / temperature, dim1)
    return batch_softmax(energies / temperature, dim1, dim2)

def soften_probabilities(probs, temperature, dim1=-2, dim2=-1):
    logprobs = optional_log(probs)
    return soften_logprobs(logprobs, temperature + 1e-10, dim1, dim2)
