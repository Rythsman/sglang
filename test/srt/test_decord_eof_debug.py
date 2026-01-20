"""
Debug script for decord EOF error analysis.

This script helps diagnose and reproduce the EOF error:
    DECORDError: Unable to handle EOF because it takes too long to retrieve last few frames

Usage:
    # Test a specific video
    python test_decord_eof_debug.py --video /path/to/video.mp4

    # Test with specific indices
    python test_decord_eof_debug.py --video /path/to/video.mp4 --test-last-frames

    # Test in multiprocess mode (simulates keye.py behavior)
    python test_decord_eof_debug.py --video /path/to/video.mp4 --multiprocess
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np


def get_video_info(video_path: str) -> dict:
    """Get video information using decord."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    info = {
        "path": video_path,
        "total_frames": len(vr),
        "fps": vr.get_avg_fps(),
        "duration": len(vr) / vr.get_avg_fps(),
    }
    return info


def test_frame_access(video_path: str, indices: List[int], timeout: float = 30.0) -> Tuple[bool, str]:
    """Test accessing specific frames."""
    from decord import VideoReader, cpu

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        start = time.time()
        frames = vr.get_batch(indices)
        elapsed = time.time() - start
        return True, f"Success: {len(indices)} frames in {elapsed:.2f}s"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_last_frames(video_path: str) -> None:
    """Test accessing the last few frames progressively."""
    from decord import VideoReader, cpu

    print(f"\n{'='*60}")
    print("Testing last frames access")
    print(f"{'='*60}")

    vr = VideoReader(video_path, ctx=cpu(0))
    nframes = len(vr)
    print(f"Total frames: {nframes}")

    # Test different offsets from the end
    for offset in [0, 1, 2, 3, 5, 10, 20, 50]:
        if offset >= nframes:
            continue

        idx = nframes - 1 - offset
        try:
            start = time.time()
            frame = vr[idx]
            elapsed = time.time() - start
            print(f"  ✅ Frame {idx} (offset -{offset}): OK ({elapsed:.3f}s)")
        except Exception as e:
            print(f"  ❌ Frame {idx} (offset -{offset}): {e}")

    # Test batch access to last N frames
    print(f"\nTesting batch access to last N frames:")
    for n in [1, 5, 10, 20]:
        if n > nframes:
            continue
        indices = list(range(nframes - n, nframes))
        try:
            start = time.time()
            frames = vr.get_batch(indices)
            elapsed = time.time() - start
            print(f"  ✅ Last {n} frames: OK ({elapsed:.3f}s)")
        except Exception as e:
            print(f"  ❌ Last {n} frames: {e}")


def simulate_keye_indices(video_path: str, target_fps: float = 2.0) -> List[int]:
    """Simulate the index generation in keye.py."""
    from decord import VideoReader, cpu
    import torch

    vr = VideoReader(video_path, ctx=cpu(0))
    nframes = len(vr)
    video_fps = vr.get_avg_fps()

    # Simulate smart_nframes logic
    fps = min(target_fps, video_fps)
    max_frames = 8192  # VIDEO_TOTAL_PIXELS / VIDEO_MIN_PIXELS approximation
    fps_nframes = int(nframes / video_fps * fps)
    final_nframes = min(fps_nframes, max_frames)
    final_nframes = max(1, final_nframes)

    # Generate indices (this is the problematic line in keye.py)
    indices = torch.linspace(0, nframes - 1, final_nframes).round().long()

    return indices.tolist(), nframes


def test_keye_simulation(video_path: str) -> None:
    """Simulate keye.py frame extraction."""
    print(f"\n{'='*60}")
    print("Simulating keye.py frame extraction")
    print(f"{'='*60}")

    indices, nframes = simulate_keye_indices(video_path)
    print(f"Total frames: {nframes}")
    print(f"Extracted frames: {len(indices)}")
    print(f"First 5 indices: {indices[:5]}")
    print(f"Last 5 indices: {indices[-5:]}")
    print(f"Max index: {max(indices)} (nframes-1 = {nframes-1})")

    # Check if last frame is included
    if nframes - 1 in indices:
        print(f"⚠️  Last frame (index {nframes-1}) IS included - potential EOF issue!")
    else:
        print(f"✅ Last frame not included")

    # Test the actual access
    print(f"\nTesting get_batch with simulated indices...")
    success, msg = test_frame_access(video_path, indices)
    if success:
        print(f"  ✅ {msg}")
    else:
        print(f"  ❌ {msg}")

    # Test with safer indices (skip last frame)
    safe_indices = [i for i in indices if i < nframes - 2]
    if len(safe_indices) < len(indices):
        print(f"\nTesting with safe indices (skip last 2 frames)...")
        success, msg = test_frame_access(video_path, safe_indices)
        if success:
            print(f"  ✅ {msg}")
        else:
            print(f"  ❌ {msg}")


def worker_process(args):
    """Worker function for multiprocess testing."""
    video_path, indices = args
    return test_frame_access(video_path, indices)


def test_multiprocess(video_path: str, num_workers: int = 4) -> None:
    """Test frame access in multiprocess environment."""
    print(f"\n{'='*60}")
    print(f"Testing multiprocess access ({num_workers} workers)")
    print(f"{'='*60}")

    indices, nframes = simulate_keye_indices(video_path)

    # Submit same task multiple times to simulate concurrent access
    tasks = [(video_path, indices) for _ in range(num_workers)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_process, task) for task in tasks]

        for i, future in enumerate(as_completed(futures)):
            success, msg = future.result()
            status = "✅" if success else "❌"
            print(f"  Worker {i}: {status} {msg}")


def suggest_fix() -> None:
    """Print suggested fix for the EOF issue."""
    print(f"\n{'='*60}")
    print("SUGGESTED FIX")
    print(f"{'='*60}")

    print("""
1. **Environment Variable** (Quick fix):
   export DECORD_EOF_RETRY_MAX=20480

2. **Code Fix** (keye.py line 373):

   # Before (problematic):
   indices = torch.linspace(0, nframes - 1, final_nframes).round().long()

   # After (safer - skip last 2 frames):
   max_idx = max(0, nframes - 3)  # Leave 2-frame margin
   indices = torch.linspace(0, max_idx, final_nframes).round().long()

3. **Add Error Handling** (Robust):

   try:
       frames_hwc = vr.get_batch(indices.tolist())
   except DECORDError as e:
       if "EOF" in str(e):
           # Retry with safer indices
           safe_indices = [i for i in indices.tolist() if i < nframes - 3]
           frames_hwc = vr.get_batch(safe_indices)
       else:
           raise

4. **Video Pre-check** (Best for training):
   - Pre-validate all videos in dataset
   - Mark problematic videos for exclusion
""")


def main():
    parser = argparse.ArgumentParser(description="Debug decord EOF errors")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--test-last-frames", action="store_true",
                        help="Test accessing last frames progressively")
    parser.add_argument("--simulate-keye", action="store_true",
                        help="Simulate keye.py frame extraction")
    parser.add_argument("--multiprocess", action="store_true",
                        help="Test in multiprocess environment")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers for multiprocess test")
    parser.add_argument("--suggest-fix", action="store_true",
                        help="Show suggested fixes")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Show video info
    print(f"{'='*60}")
    print("VIDEO INFO")
    print(f"{'='*60}")
    info = get_video_info(args.video)
    for k, v in info.items():
        if k == "duration":
            print(f"  {k}: {v:.2f} seconds")
        elif k == "fps":
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Run requested tests
    if args.test_last_frames:
        test_last_frames(args.video)

    if args.simulate_keye:
        test_keye_simulation(args.video)

    if args.multiprocess:
        test_multiprocess(args.video, args.workers)

    if args.suggest_fix:
        suggest_fix()

    # Default: run all tests
    if not any([args.test_last_frames, args.simulate_keye, args.multiprocess, args.suggest_fix]):
        test_last_frames(args.video)
        test_keye_simulation(args.video)
        suggest_fix()


if __name__ == "__main__":
    main()
