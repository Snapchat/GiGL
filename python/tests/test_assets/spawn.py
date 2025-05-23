import torch.multiprocessing as mp


def spawn_new_process(func, *args):
    mp.spawn(fn=func, args=args)
