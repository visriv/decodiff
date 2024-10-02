import torch.distributed as dist



# export MASTER_ADDR='localhost'  # Or the address of the master node
# export MASTER_PORT='12355'      # An open port on the master node


world_size = 1
rank = 0

# Initialize the process group with the specified backend
dist.init_process_group(
    backend='nccl',         # Replace 'nccl' with your chosen backend
    init_method='env://',
    world_size=world_size,
    rank=rank
)

# Check the current backend
current_backend = dist.get_backend()
print(f"The current backend is: {current_backend}")
