# common/threading.py
import os, ctypes
def force_singlethread() -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:    # MKL
        import mkl
        mkl.set_num_threads(1)
    except (ImportError, AttributeError):
        pass
    try:  # OpenBLAS
        ctypes.CDLL("libopenblas.so").openblas_set_num_threads(1)
    except OSError:
        pass
