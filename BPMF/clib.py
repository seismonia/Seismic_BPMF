import os

import ctypes as C
import numpy as np
import datetime as dt
import warnings

from .config import cfg

cpu_loaded = False

libpath = os.path.join(cfg.PACKAGE, "lib")

try:
    _libc = C.cdll.LoadLibrary(os.path.join(libpath, "libc.so"))
    cpu_loaded = True
except:
    print(
        "Missing libc.so! You won"
        "t be able to run multiplet/template searches on the CPU"
    )
    print("Should be at {}libc.so".format(os.path.join(libpath, "")))


if cpu_loaded:
    _libc.kurtosis.argtypes = [
        C.POINTER(C.c_float),
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_int,
        C.POINTER(C.c_float),
    ]

    _libc.find_similar_moveouts.argtypes = [
        C.POINTER(C.c_float),  # moveouts
        C.POINTER(C.c_float),  # source_longitude
        C.POINTER(C.c_float),  # source_latitude
        C.POINTER(C.c_float),  # cell_longitude
        C.POINTER(C.c_float),  # cell_latitude
        C.c_float,  # rms time difference threshold
        C.c_size_t,  # number of grid points
        C.c_size_t,  # number of stations
        C.c_size_t,  # number of cells in longitude
        C.c_size_t,  # number of cells in latitude
        C.c_size_t,  # number of stations for diff
        C.c_int, # num threads
        C.POINTER(C.c_int),  # output pointer: redundant sources
    ]

    _libc.find_similar_moveouts2.argtypes = [
        C.POINTER(C.c_float),  # moveouts
        C.POINTER(C.c_float),  # source_longitude
        C.POINTER(C.c_float),  # source_latitude
        C.POINTER(C.c_float),  # cell_longitude
        C.POINTER(C.c_float),  # cell_latitude
        C.c_float,  # rms time difference threshold
        C.c_size_t,  # number of grid points
        C.c_size_t,  # number of stations
        C.c_size_t,  # number of cells in longitude
        C.c_size_t,  # number of cells in latitude
        C.c_size_t,  # number of stations for diff
        C.c_int, # num threads
        C.POINTER(C.c_int),  # output pointer: redundant sources
    ]

    _libc.select_cc_indexes.argtypes = [
        C.POINTER(C.c_float),  # CCs
        C.POINTER(C.c_float),  # threshold
        C.c_size_t,  # search window
        C.c_size_t,  # length of CCs
        C.POINTER(C.c_int),  # selected corr
    ]

    _libc.time_dependent_threshold.argtypes = [
            C.POINTER(C.c_float),  # input time series
            C.POINTER(C.c_float),  # gaussian sample
            C.c_float,             # num_dev
            C.c_size_t,            # num_samples
            C.c_size_t,            # half_window_samp
            C.c_size_t,            # shift_samp
            C.c_int, # num threads
            C.POINTER(C.c_float),  # output threshold
            ]


def kurtosis(signal, W):
    n_stations = signal.shape[0]
    n_components = signal.shape[1]
    length = signal.shape[-1]
    Kurto = np.zeros(n_stations * n_components * length, dtype=np.float32)
    signal = np.float32(signal.flatten())
    _libc.kurtosis(
        signal.ctypes.data_as(C.POINTER(C.c_float)),
        np.int32(W),
        np.int32(n_stations),
        np.int32(n_components),
        np.int32(length),
        Kurto.ctypes.data_as(C.POINTER(C.c_float)),
    )
    return Kurto.reshape(n_stations, n_components, length)


def find_similar_sources(
        moveouts,
        source_longitude,
        source_latitude,
        cell_longitude,
        cell_latitude,
        threshold,
        num_threads=None,
        num_stations_for_diff=None,
        method="closest"
        ):
    """
    Find sources with similar moveouts so that users can discard
    some of them during the computation of the network response
    and thus speedup the process.

    Parameters
    -------------
    moveouts : numpy.ndarray
        The (n_sources, n_stations) moveout 2-D array, in seconds.
        Note: It makes more sense to give the relative travel times (w.r.t. earliest arrival)
        rather than the absolute travel times.
    source_longitude : array_like
        The (n_sources,) list or 1-D array of source longitudes.
    source_latitude : array_like
        The (n_sources,) list or 1-D array of source latitudes.
    cell_longitude : array_like
        The (n_cells_longitude,) list or 1-D array of the vortex longitudes
        defining the geographic cells used to sub-divide the problem.
    cell_latitude : array_like
        The (n_cells_latitude,) list or 1-D array of the vortex latitudes
        defining the geographic cells used to sub-divide the problem.
    threshold: float
        The station average time difference tolerance to consider
        two sources as being redundant.
    num_threads : int or None, optional
        The number of threads over which the computation is parallelized. If None or -1,
        spaws one thread per available CPU. Defaults to None.
    num_stations_for_diff : int or None, optional
        The number of stations over which the sum of the squared differences is computed.
        See `method` for more info. If None, uses all of the stations. Defaults to None.
    method : str, optional
        Either of 'closest' or 'smallest'.
        - 'closest': Find the `num_stations_for_diff` closest stations to every source
          in the grid and restrict the sum to those.
        - 'smallest': Compute the differences at every station but use only the
          `num_stations_for_diff` smallest differences in the sum.

    Returns
    -------------
    redundant_sources: (n_sources,) boolean numpy.ndarray
        Boolean numpy array with True elements for sources that
        share similar moveouts with other sources.
    """
    n_sources = moveouts.shape[0]
    n_stations = moveouts.shape[1]
    n_cells_longitude = len(cell_longitude) - 1
    n_cells_latitude = len(cell_latitude) - 1

    if num_stations_for_diff is None:
        num_stations_for_diff = n_stations

    if moveouts.dtype in (np.int32, np.int64):
        print("Integer typed moveouts detected. Are you sure these are in" " seconds?")

    if num_threads is None:
        # set num_threads to -1 so that the C routine
        # understands to use all CPUs
        #num_threads = os.cpu_count()
        num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))

    assert method in {"closest", "smallest"}, "method should be either of 'closest' or 'smallest'"

    # format input arrays
    moveouts = np.float32(moveouts.flatten())
    source_longitude = np.float32(source_longitude)
    source_latitude = np.float32(source_latitude)
    cell_longitude = np.float32(cell_longitude)
    cell_latitude = np.float32(cell_latitude)

    # initialize the output pointer
    redundant_sources = np.zeros(n_sources, dtype=np.int32)

    # call the C function:
    if method == "closest":
        _libc.find_similar_moveouts2(
            moveouts.ctypes.data_as(C.POINTER(C.c_float)),
            source_longitude.ctypes.data_as(C.POINTER(C.c_float)),
            source_latitude.ctypes.data_as(C.POINTER(C.c_float)),
            cell_longitude.ctypes.data_as(C.POINTER(C.c_float)),
            cell_latitude.ctypes.data_as(C.POINTER(C.c_float)),
            np.float32(threshold),
            int(n_sources),
            int(n_stations),
            int(n_cells_longitude),
            int(n_cells_latitude),
            int(num_stations_for_diff),
            int(num_threads),
            redundant_sources.ctypes.data_as(C.POINTER(C.c_int)),
        )
    elif method == "smallest":
        _libc.find_similar_moveouts(
            moveouts.ctypes.data_as(C.POINTER(C.c_float)),
            source_longitude.ctypes.data_as(C.POINTER(C.c_float)),
            source_latitude.ctypes.data_as(C.POINTER(C.c_float)),
            cell_longitude.ctypes.data_as(C.POINTER(C.c_float)),
            cell_latitude.ctypes.data_as(C.POINTER(C.c_float)),
            np.float32(threshold),
            int(n_sources),
            int(n_stations),
            int(n_cells_longitude),
            int(n_cells_latitude),
            int(num_stations_for_diff),
            int(num_threads),
            redundant_sources.ctypes.data_as(C.POINTER(C.c_int)),
        )

    return redundant_sources.astype(bool)


def select_cc_indexes(ccs, threshold, search_win):
    """Select new event detection's correlation indexes.

    Parameters
    -----------
    ccs: (n_corr,) `numpy.ndarray`
        Time series of correlation coefficients.
    threshold (n_corr,) `numpy.ndarray` or `float` scalar
        Time series or scalar detection threshold.
    search_win: `int` scalar
        Size of the time window, in number of consecutive correlations, defining
        grouped detections.

    Returns
    --------
    selection: (n_corr,) bool `numpy.ndarray`
        Vector of `n_corr` booleans that are true if the corresponding CC index
        is a new event detection.
    """
    n_corr = len(ccs)
    if isinstance(threshold, float) or isinstance(threshold, int):
        threshold = threshold * np.ones(n_corr, dtype=np.float32)
    threshold = np.float32(threshold)
    selection = np.zeros(n_corr, dtype=np.int32)
    _libc.select_cc_indexes(
        ccs.ctypes.data_as(C.POINTER(C.c_float)),
        threshold.ctypes.data_as(C.POINTER(C.c_float)),
        int(search_win),
        int(n_corr),
        selection.ctypes.data_as(C.POINTER(C.c_int)),
    )
    return selection.astype(bool)

def time_dependent_threshold(
        time_series, sliding_window_samp, num_dev, overlap=0.66, threshold_type="rms",
        white_noise=None, num_threads=None
        ):
    """
    Time dependent detection threshold.

    Parameters
    -----------
    time_series : (n_correlations) array_like
        The array of correlation coefficients calculated by
        FMF (float 32).
    sliding_window_samp : scalar integer
        The size of the sliding window, in samples, used
        to calculate the time dependent central tendency
        and deviation of the time series.
    overlap : scalar float, default to 0.75
    threshold_type : string, default to 'rms'
        Either rms or mad, depending on which measure
        of deviation you want to use.

    Returns
    ----------
    threshold: (n_correlations) array_like
        Returns the time dependent threshold, with same
        size 
    """
    GAUSSIAN_SAMPLE_LEN = 500
    time_series = (time_series.copy())
    num_samples = len(time_series)
    if white_noise is None:
        white_noise = np.random.normal(size=GAUSSIAN_SAMPLE_LEN).astype("float32")

    if num_threads is None:
        num_threads = os.cpu_count()

    threshold_type = threshold_type.lower()
    half_window_samp = sliding_window_samp // 2
    shift_samp = int((1.0 - overlap) * sliding_window_samp)
    threshold = np.zeros(num_samples, dtype=np.float32)

    _libc.time_dependent_threshold(
            time_series.astype("float32").ctypes.data_as(C.POINTER(C.c_float)),
            white_noise.astype("float32").ctypes.data_as(C.POINTER(C.c_float)),
            float(num_dev),
            int(num_samples),
            int(half_window_samp),
            int(shift_samp),
            int(num_threads),
            threshold.ctypes.data_as(C.POINTER(C.c_float))
            )

    return threshold
