## Learning summary

* ``CrossEntropy Loss`` forward and backward with PyTorch bindings.
* Optimized for batch parallelization.
* Supports reduction = none and mean (see parameterized tests in ``test.py``).


To run:

```bash
pip install torch pytest
python setup.py install

# Test
pytest test.py

# Benchmark
python benchmark.py
```