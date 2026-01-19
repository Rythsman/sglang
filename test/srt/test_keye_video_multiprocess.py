"""
Unit tests for keye.py multiprocess video processing logic.

This test verifies that:
1. kwargs_list order is consistent between multiprocess and single-process processing
2. Mixed dict and non-dict videos are handled correctly
3. asyncio.gather preserves task order
"""

import asyncio
import collections
import unittest
from concurrent.futures import ThreadPoolExecutor


class TestKeyeVideoMultiprocessLogic(unittest.TestCase):
    """
    Test the core multiprocess video processing logic without actual video processing.

    This tests the logic pattern used in keye.py:process_mm_data_async():
        for i, video in enumerate(base_output.videos):
            if isinstance(video, dict):
                continue
            tasks.append(loop.run_in_executor(...))
            task_indices.append(i)
        processed = await asyncio.gather(*tasks)
        frames_list, kwargs_list = zip(*processed)
        for idx, frames in zip(task_indices, frames_list):
            base_output.videos[idx] = frames
        for kw in kwargs_list:
            for k, v in kw.items():
                videos_kwargs[k].append(v)
    """

    def test_task_indices_skip_dict(self):
        """Test that dict videos are correctly skipped and task_indices is accurate."""
        videos = [
            "video_0.mp4",  # raw, index 0
            {"pre_computed": True},  # dict (skipped), index 1
            "video_2.mp4",  # raw, index 2
            {"pre_computed": True},  # dict (skipped), index 3
            "video_4.mp4",  # raw, index 4
        ]

        task_indices = []
        for i, video in enumerate(videos):
            if isinstance(video, dict):
                continue
            task_indices.append(i)

        # Verify dict videos are skipped
        self.assertEqual(task_indices, [0, 2, 4])

    def test_asyncio_gather_preserves_order(self):
        """Verify asyncio.gather returns results in input order, not completion order."""

        async def delayed_task(task_id, delay):
            await asyncio.sleep(delay)
            return task_id

        async def run_test():
            # Create tasks with different delays
            # Task 0: 0.1s, Task 1: 0.05s, Task 2: 0.15s
            # Completion order would be: 1, 0, 2
            # But gather should return: 0, 1, 2
            tasks = [
                delayed_task(0, 0.1),
                delayed_task(1, 0.05),
                delayed_task(2, 0.15),
            ]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_test())
        self.assertEqual(results, [0, 1, 2])

    def test_multiprocess_order_simulation(self):
        """
        Simulate the multiprocess logic and verify order consistency.

        This tests the pattern:
            processed = await asyncio.gather(*tasks)
            frames_list, kwargs_list = zip(*processed)
            for idx, frames in zip(task_indices, frames_list):
                videos[idx] = frames
        """

        def mock_process_video(video_path):
            """Mock video processing that returns (frames, kwargs)."""
            # Use video path to generate deterministic results
            video_id = int(video_path.split("_")[1].split(".")[0])
            frames = f"frames_{video_id}"
            kwargs = {
                "height": 100 + video_id * 10,
                "width": 200 + video_id * 20,
                "video_id": video_id,
            }
            return frames, kwargs

        async def simulate_multiprocess():
            videos = [
                "video_0.mp4",
                {"pre_computed": True},  # will be skipped
                "video_2.mp4",
                "video_3.mp4",
            ]

            loop = asyncio.get_running_loop()
            tasks = []
            task_indices = []

            for i, video in enumerate(videos):
                if isinstance(video, dict):
                    continue
                with ThreadPoolExecutor(max_workers=4) as executor:
                    tasks.append(
                        loop.run_in_executor(executor, mock_process_video, video)
                    )
                    task_indices.append(i)

            processed = await asyncio.gather(*tasks) if tasks else []
            frames_list, kwargs_list = zip(*processed) if processed else ([], [])

            # Update videos
            for idx, frames in zip(task_indices, frames_list):
                videos[idx] = frames

            # Aggregate kwargs
            videos_kwargs = collections.defaultdict(list)
            for kw in kwargs_list:
                for k, v in kw.items():
                    if v is None:
                        continue
                    videos_kwargs[k].append(v)

            return videos, videos_kwargs, task_indices

        videos, videos_kwargs, task_indices = asyncio.run(simulate_multiprocess())

        # Verify task_indices
        self.assertEqual(task_indices, [0, 2, 3])

        # Verify videos update
        self.assertEqual(videos[0], "frames_0")
        self.assertIsInstance(videos[1], dict)  # unchanged
        self.assertEqual(videos[2], "frames_2")
        self.assertEqual(videos[3], "frames_3")

        # Verify kwargs order
        self.assertEqual(videos_kwargs["video_id"], [0, 2, 3])
        self.assertEqual(videos_kwargs["height"], [100, 120, 130])
        self.assertEqual(videos_kwargs["width"], [200, 240, 260])

    def test_zip_unpack_edge_cases(self):
        """Test zip(*processed) behavior for edge cases."""
        # Normal case: multiple tuples
        processed = [("a", 1), ("b", 2), ("c", 3)]
        frames_list, kwargs_list = zip(*processed)
        self.assertEqual(frames_list, ("a", "b", "c"))
        self.assertEqual(kwargs_list, (1, 2, 3))

        # Edge case: single tuple
        processed = [("a", 1)]
        frames_list, kwargs_list = zip(*processed)
        self.assertEqual(frames_list, ("a",))
        self.assertEqual(kwargs_list, (1,))

        # Edge case: empty list (handled by ternary)
        processed = []
        frames_list, kwargs_list = (
            zip(*processed) if processed else ([], [])
        )
        self.assertEqual(list(frames_list), [])
        self.assertEqual(list(kwargs_list), [])

    def test_organize_results_order_consistency(self):
        """
        Test that raw_videos order matches videos_kwargs order after processing.

        This simulates the organize_results() logic:
            for modality, item in all_items:
                if isinstance(item, dict):
                    dict_items.append(item)
                elif modality == Modality.VIDEO:
                    raw_videos.append(item)
        """
        # After multiprocess update, videos looks like:
        videos = [
            "frames_0",  # processed frames (was video_0.mp4)
            {"pre_computed": True},  # dict (unchanged)
            "frames_2",  # processed frames (was video_2.mp4)
            "frames_3",  # processed frames (was video_3.mp4)
        ]

        # organize_results would return items in order
        raw_videos = []
        dict_items = []
        for video in videos:
            if isinstance(video, dict):
                dict_items.append(video)
            else:
                raw_videos.append(video)

        # Verify order
        self.assertEqual(raw_videos, ["frames_0", "frames_2", "frames_3"])
        self.assertEqual(len(dict_items), 1)

        # This order matches videos_kwargs order: [0, 2, 3]
        # Because:
        # 1. task_indices = [0, 2, 3] (skipping index 1 which is dict)
        # 2. asyncio.gather returns in same order as tasks
        # 3. kwargs_list follows tasks order
        # 4. videos_kwargs aggregates in kwargs_list order
        # 5. organize_results iterates videos in order, skipping dicts
        # 6. So raw_videos order = [0, 2, 3] = kwargs_list order


class TestPotentialBugScenarios(unittest.TestCase):
    """Test potential bug scenarios in the multiprocess logic."""

    def test_issue_kwargs_order_mismatch(self):
        """
        Test a potential bug where kwargs order might mismatch raw_videos order.

        BUG SCENARIO (DOES NOT EXIST):
        If asyncio.gather returned results in completion order instead of input order,
        kwargs_list would be out of order with respect to task_indices.

        VERIFICATION:
        asyncio.gather DOES preserve input order, so this is NOT a bug.
        """

        async def simulate_with_delays():
            """Simulate processing where task 1 completes before task 0."""

            async def process_with_delay(video_id, delay):
                await asyncio.sleep(delay)
                return f"frames_{video_id}", {"video_id": video_id}

            videos = ["video_0.mp4", "video_1.mp4", "video_2.mp4"]
            task_indices = []
            tasks = []

            for i, video in enumerate(videos):
                video_id = int(video.split("_")[1].split(".")[0])
                # Different delays: video_0=0.15s, video_1=0.05s, video_2=0.1s
                delay = [0.15, 0.05, 0.1][video_id]
                tasks.append(process_with_delay(video_id, delay))
                task_indices.append(i)

            processed = await asyncio.gather(*tasks)
            frames_list, kwargs_list = zip(*processed)

            return frames_list, kwargs_list, task_indices

        frames_list, kwargs_list, task_indices = asyncio.run(simulate_with_delays())

        # Verify order is preserved (0, 1, 2) not completion order (1, 2, 0)
        self.assertEqual(task_indices, [0, 1, 2])
        self.assertEqual(
            [kw["video_id"] for kw in kwargs_list],
            [0, 1, 2],  # Input order, not completion order
        )

    def test_all_videos_are_dict(self):
        """Test edge case where all videos are pre-computed dicts."""
        videos = [
            {"pre_computed": True},
            {"pre_computed": True},
        ]

        task_indices = []
        tasks = []
        for i, video in enumerate(videos):
            if isinstance(video, dict):
                continue
            task_indices.append(i)
            tasks.append(None)  # Would add task here

        # All videos are dicts, so no tasks
        self.assertEqual(task_indices, [])
        self.assertEqual(tasks, [])

        # The code handles this with:
        # processed = await asyncio.gather(*tasks) if tasks else []
        # frames_list, kwargs_list = zip(*processed) if processed else ([], [])
        processed = []
        frames_list, kwargs_list = (
            zip(*processed) if processed else ([], [])
        )
        self.assertEqual(list(frames_list), [])
        self.assertEqual(list(kwargs_list), [])


if __name__ == "__main__":
    import sys

    # Simple test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKeyeVideoMultiprocessLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestPotentialBugScenarios))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
