from model.falcon_gate_modeling import FalconMambaForCausalLM_Gate
from model.falcon_ssm_modeling import FalconMambaForCausalLM_SSM
import torch
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import argparse
from transformers import FalconMambaConfig

def setup_distributed(devices, rank=0, world_size=2):
    device_str = devices[rank]
    print(f"[Rank {rank}] Setting up distributed on device {device_str}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12233"

    # Set the CUDA device BEFORE init_process_group
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    # Initialize distributed process group (NCCL)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id = device
    )

    print(f"[Rank {rank}] Distributed process group initialized on {device_str}")
    return rank, world_size, device

def cleanup(rank):
    print(f"[Rank {rank}] Cleaning up distributed process group")
    dist.destroy_process_group()

def worker(rank, devices, input_ids):
    try:
        print(f"[Rank {rank}] Worker started")
        rank, world_size, device = setup_distributed(devices, rank=rank)
        
        dist.barrier()

        input_ids = input_ids.to(device)
        print(f"[Rank {rank}] input_ids moved to device {device}")

        config = model = None
        if rank == 0:
            print(f"[Rank {rank}] Loading FalconMambaForCausalLM_Gate")
            config = FalconMambaConfig.from_pretrained("HMasarani/falcon-mamba-gate")
            model = FalconMambaForCausalLM_Gate(config).to(device)
            print(f"[Rank {rank}] FalconMambaForCausalLM_Gate loaded")
        else:
            print(f"[Rank {rank}] Loading FalconMambaForCausalLM_SSM")
            config = FalconMambaConfig.from_pretrained("HMasarani/falcon-mamba-ssm")
            model = FalconMambaForCausalLM_SSM(config).to(device)
            print(f"[Rank {rank}] FalconMambaForCausalLM_SSM loaded")

        model.eval()
        print(f"[Rank {rank}] Model loaded")

        dist.barrier()
        print(f"[Rank {rank}] Starting forward pass, on device {device}")
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
        print(f"[Rank {rank}] Forward pass done")


        if rank == 0:
            logits = outputs.logits
            torch.save(logits.cpu(), "output/logits.pt")

        dist.barrier()
        cleanup(rank)
        print(f"[Rank {rank}] Worker finished successfully")

    except Exception as e:
        print(f"[Rank {rank}] Exception occurred: {e}")
        dist.barrier()
        cleanup(rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev0", type=str, default="cuda:2")
    parser.add_argument("-dev1", type=str, default="cuda:3")
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-seq_len", type=int, default=1)
    parser.add_argument("-num_iter", type=int, default=1)

    args = parser.parse_args()

    devices = [args.dev0, args.dev1]
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_iter = args.num_iter

    input_ids = torch.randint(0, 4096, (batch_size, seq_len))

    mp.set_start_method("spawn", force=True)

    print("[Main] Spawning workers")
    mp.spawn(worker, args=(devices, input_ids), nprocs=2, join=True)
    print("[Main] Workers joined")

    logits = torch.load("output/logits.pt")
    print("[Main] model output: ", logits)