# MIT License

# Copyright (c) 2024 The HuggingFace Team

import gc

import pytest


def _log_gpu_memory(stage: str, test_name: str = ""):
    """Print GPU memory statistics for debugging slow tests."""
    try:
        import torch

        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GiB
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        total = props.total_memory / (1024**3)
        free = total - reserved

        test_info = f" [{test_name}]" if test_name else ""
        print(f"\n{'=' * 80}")
        print(f"GPU Memory {stage}{test_info}")
        print(f"{'=' * 80}")
        print(f"  Device: {props.name}")
        print(f"  Total:     {total:.2f} GiB")
        print(f"  Allocated: {allocated:.2f} GiB")
        print(f"  Reserved:  {reserved:.2f} GiB")
        print(f"  Free:      {free:.2f} GiB")
        print(f"{'=' * 80}\n")
    except ImportError:
        pass


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True, scope="function")
def cleanup_gpu_memory(request):
    """Cleanup GPU memory before and after each test to prevent OOM errors."""
    # Cleanup before test (especially important for tests that run after other GPU-heavy tests)
    if "slow" in request.keywords:
        test_name = request.node.name

        # Log memory BEFORE cleanup
        _log_gpu_memory("BEFORE cleanup", test_name)

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        gc.collect()

        # Log memory AFTER cleanup (before test starts)
        _log_gpu_memory("AFTER cleanup (test starting)", test_name)

    yield

    # Cleanup after test
    if "slow" in request.keywords:
        test_name = request.node.name

        # Log memory AFTER test (before cleanup)
        _log_gpu_memory("AFTER test (before cleanup)", test_name)

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        gc.collect()

        # Log memory AFTER cleanup
        _log_gpu_memory("AFTER cleanup", test_name)
