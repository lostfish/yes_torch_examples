import numpy as np
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn

def set_manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def dist_init(ip, rank, local_rank, world_size, port):
    host_addr_full = 'tcp://' + ip + ':' + str(port)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(
                2*hidden_size)] = 1.0
