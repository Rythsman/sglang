#!/usr/bin/env python3
"""
Example code to demonstrate DCP (Decode Context Parallel) rank allocation.
This script shows how ranks are assigned to different DCP groups based on
the logic in parallel_state.py.
"""

def calculate_dcp_group_ranks(world_size: int, decode_context_model_parallel_size: int):
    """
    Calculate DCP group ranks based on the logic in parallel_state.py.
    
    Args:
        world_size: Total number of GPUs/processes
        decode_context_model_parallel_size: Size of each DCP group
    
    Returns:
        List of lists, where each inner list contains the ranks in one DCP group
    """
    num_decode_context_model_parallel_groups: int = (
        world_size // decode_context_model_parallel_size
    )
    
    group_ranks = []
    for i in range(num_decode_context_model_parallel_groups):
        ranks = list(
            range(
                i * decode_context_model_parallel_size,
                (i + 1) * decode_context_model_parallel_size,
            )
        )
        group_ranks.append(ranks)
    
    return group_ranks


def print_rank_allocation(world_size: int, decode_context_model_parallel_size: int):
    """
    Print detailed rank allocation information.
    """
    print("=" * 80)
    print(f"DCP Rank Allocation Analysis")
    print("=" * 80)
    print(f"World Size: {world_size}")
    print(f"Decode Context Model Parallel Size: {decode_context_model_parallel_size}")
    print()
    
    group_ranks = calculate_dcp_group_ranks(world_size, decode_context_model_parallel_size)
    num_groups = len(group_ranks)
    
    print(f"Number of DCP Groups: {num_groups}")
    print()
    
    # Print group information
    for group_idx, ranks in enumerate(group_ranks):
        print(f"DCP Group {group_idx}:")
        print(f"  Ranks: {ranks}")
        print(f"  Size: {len(ranks)}")
        print()
    
    # Print per-rank information
    print("Per-Rank Information:")
    print("-" * 80)
    print(f"{'Global Rank':<15} {'DCP Group':<15} {'Rank in Group':<15}")
    print("-" * 80)
    
    for rank in range(world_size):
        for group_idx, ranks in enumerate(group_ranks):
            if rank in ranks:
                rank_in_group = ranks.index(rank)
                print(f"{rank:<15} {group_idx:<15} {rank_in_group:<15}")
                break
    
    print("=" * 80)


if __name__ == "__main__":
    # Example: world_size=16, decode_context_model_parallel_size=8
    world_size = 16
    decode_context_model_parallel_size = 8
    
    print_rank_allocation(world_size, decode_context_model_parallel_size)
    
    print("\n" + "=" * 80)
    print("Code Logic Explanation:")
    print("=" * 80)
    print("""
The rank allocation follows this logic (from parallel_state.py):

    num_decode_context_model_parallel_groups = world_size // decode_context_model_parallel_size
    group_ranks = []
    for i in range(num_decode_context_model_parallel_groups):
        ranks = list(
            range(
                i * decode_context_model_parallel_size,
                (i + 1) * decode_context_model_parallel_size,
            )
        )
        group_ranks.append(ranks)

For world_size=16, decode_context_model_parallel_size=8:
    - num_decode_context_model_parallel_groups = 16 // 8 = 2
    - Group 0: ranks = range(0, 8) = [0, 1, 2, 3, 4, 5, 6, 7]
    - Group 1: ranks = range(8, 16) = [8, 9, 10, 11, 12, 13, 14, 15]

Result:
    - DCP Group 0 contains ranks [0, 1, 2, 3, 4, 5, 6, 7]
    - DCP Group 1 contains ranks [8, 9, 10, 11, 12, 13, 14, 15]
    """)
    print("=" * 80)
