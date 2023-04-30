import torch

if __name__ == '__main__':
    state_dict = torch.load('hydrogen_mace_state_dict.model')
    for key in state_dict.keys():
        print(key)