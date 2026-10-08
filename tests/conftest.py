import os

# Before the test modules load numpy: OpenBLAS threads only spin on the tests' small
# matrix products (one follower test: 31 s wall and 394 s CPU with them, 22 s and 18 s
# without).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
