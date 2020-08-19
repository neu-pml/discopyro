import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_logprobs(logprobs, dim1=-2, dim2=-1):
    logprobs = list(logprobs.unbind(dim=dim1))
    for i, logprob in enumerate(logprobs):
        nonzeros = logprob.nonzero(as_tuple=True)
        logprobs[i] = torch.index_put(logprob, nonzeros,
                                      F.softmax(logprob[nonzeros], dim=dim2))
    return torch.stack(logprobs, dim=dim1)

def optional_log(tensor):
    nonzeros = tensor.nonzero(as_tuple=True)
    return torch.index_put(tensor, nonzeros, torch.log(tensor[nonzeros]))

def optional_energies(probs):
    return -optional_log(probs)

def soften_logprobs(energies, temperature, dim1=-2, dim2=-1):
    return softmax_logprobs(energies / temperature, dim1, dim2)

def soften_probabilities(probs, temperature, dim1=-2, dim2=-1):
    logprobs = optional_log(probs)
    return soften_logprobs(logprobs, temperature + 1e-5, dim1, dim2)
