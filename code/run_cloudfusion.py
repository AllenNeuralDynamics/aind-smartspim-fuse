import multiprocessing as mp
import os

import torch
from aind_smartspim_fuse import cloudfusion_fusion


def run():
    """Executes the capsule"""
    # Some configurations helpful for GPU processing.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print("Multiprocessing start method: ", mp.get_start_method(allow_none=False))
    print(
        "Multiprocessing start forkserver: ",
        mp.set_start_method("forkserver", force=True),
    )
    print("Multiprocessing start method: ", mp.get_start_method(allow_none=False))
    torch.cuda.empty_cache()

    cloudfusion_fusion.execute_job()


if __name__ == "__main__":
    run()
