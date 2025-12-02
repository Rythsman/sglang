#!/usr/bin/env python3
"""
Example code to demonstrate DCP (Decode Context Parallel) rank assignment.
This script shows how ranks are assigned to different DCP groups when
world_size=16 and decode_context_model_parallel_size=8.
"""

def calculate_dcp_ranks(world_size: int, decode_context_model_parallel_size: int):
    """
    Calculate DCP group ranks based on world_size and decode_context_model_parallel_size.
    
    Args:
        world_size: Total number of GPUs/processes
        decode_context_model_parallel_size: Size of each DCP group
    
    Returns:
        List of lists, where each inner list contains ranks for one DCP group
    """
    num_decode_context_model_parallel_groups = (
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


def print_rank_assignment(world_size: int, decode_context_model_parallel_size: int):
    """
    Print detailed rank assignment information.
    """
    group_ranks = calculate_dcp_ranks(world_size, decode_context_model_parallel_size)
    
    print(f"World Size: {world_size}")
    print(f"Decode Context Model Parallel Size: {decode_context_model_parallel_size}")
    print(f"Number of DCP Groups: {len(group_ranks)}")
    print("\n" + "="*60)
    print("DCP Group Rank Assignment:")
    print("="*60)
    
    for group_idx, ranks in enumerate(group_ranks):
        print(f"\nDCP Group {group_idx}:")
        print(f"  Ranks: {ranks}")
        print(f"  Size: {len(ranks)}")
        print(f"  Rank Range: {ranks[0]} to {ranks[-1]}")
    
    print("\n" + "="*60)
    print("Per-Card Rank Assignment:")
    print("="*60)
    
    for rank in range(world_size):
        # Find which group this rank belongs to
        group_idx = None
        rank_in_group = None
        for g_idx, ranks in enumerate(group_ranks):
            if rank in ranks:
                group_idx = g_idx
                rank_in_group = ranks.index(rank)
                break
        
        print(f"Rank {rank:2d}: DCP Group {group_idx}, Rank in Group: {rank_in_group}")


if __name__ == "__main__":
    # Example: world_size=16, decode_context_model_parallel_size=8
    world_size = 16
    decode_context_model_parallel_size = 8
    
    print_rank_assignment(world_size, decode_context_model_parallel_size)
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("With world_size=16 and decode_context_model_parallel_size=8:")
    print("- There are 2 DCP groups")
    print("- Group 0 contains ranks [0, 1, 2, 3, 4, 5, 6, 7]")
    print("- Group 1 contains ranks [8, 9, 10, 11, 12, 13, 14, 15]")
    print("- Each group has 8 ranks")
    print("- Ranks 0-7 belong to DCP Group 0")
    print("- Ranks 8-15 belong to DCP Group 1")
