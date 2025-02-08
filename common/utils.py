import torch
import time


def compute_time_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")

    return result


def compare_outputs(output1, output2, tolerance=1e-5):
    assert isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor)
    # Compare two torch tensors
    if torch.allclose(output1, output2, atol=tolerance):
        print("Outputs match !")
    else:
        print("Outputs mismatch !")
