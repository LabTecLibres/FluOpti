# ==============================================================================
# 1. GENERAL PACKAGES AND LIBRARIES
# ==============================================================================
import os
import random
import time
import types
import signal
import atexit
import logging
import regex as re
from math import log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import emcee
import corner

from numba import njit, cfunc, types, carray, prange

# Import List from numba to use typed lists
from numba.typed import List
from numba.experimental import jitclass
from numbalsoda import lsoda, address_as_void_pointer
from scipy.stats import truncnorm

# import local/project-specific modules after standard libraries
import analysis as flua
import plotting as flup

# ==============================================================================
# 2. GLOBAL CONFIGURATIONS AND CONSTANTS
# ==============================================================================

plt.rcParams["figure.dpi"] = 300

# --- Constants for Data Handling ---
# Timelapse information mapping experiment IDs to light conditions
TIMELAPSE_INFORMATION = {
    "TL240521": ["48HRS_GREEN_20PER", [0.20, 0.00]],
    "TL240523": ["48HRS_GREEN_40PER", [0.40, 0.00]],
    "TL240528": ["48HRS_GREEN_80PER", [0.80, 0.00]],
    "TL240530": ["48HRS_DARK", [0.00, 0.00]],
    "TL240607": ["48HRS_GREEN_RED_100PER", [1.00, 1.00]],
    "TL240610": ["48HRS_RED_50PER", [0.00, 0.50]],
    "TL240615": ["48HRS_RED_75PER", [0.00, 0.75]],
    "TL240716": ["48HRS_RED_10PER_GREEN_100PER", [1.00, 0.10]],
    "TL240719": ["48HRS_RED_10PER_GREEN_50PER", [0.50, 0.10]],
    "TL240727": ["48HRS_RED_10PER_GREEN_25PER", [0.25, 0.10]],
    "TL240801": ["48HRS_RED_20PER_GREEN_50PER", [0.50, 0.20]],
    "TL240804": ["48HRS_RED_30PER_GREEN_50PER", [0.50, 0.30]],
}

# absolute key order produced by `exportRoisData`
ROW_NAMES = [
    "roi",
    "time",
    "exp_data_means",
    "exp_data_stds",
    "t0_tf",
    "data_tpoints",
    "IA_0",
    "schedule_times",
    "schedule_R",
    "schedule_G",
    "schedule_R_COMPLETE",
    "schedule_G_COMPLETE",
    "firstswitch",
    "secondswitch",  # <- original 14 rows
    "dataset_name",  # <- 15th row we add
]

# --- Constants for MCMC Priors ---
PRIOR_UNIFORM = 0
PRIOR_GAUSS = 1

# ==============================================================================
# 3. FUNCTION DEFINITIONS
#
# All function definitions are now at the global level to avoid scoping
# issues and to allow for proper pickling by multiprocessing.
# ==============================================================================


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX."""
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = width_pt * fraction * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


# ─────────────────────────────────────────────────────────────
# Robust: works for whitespace-, comma-, or ellipsis-separated
# lists like “[ 0.0  0.25 … 71.8 ]”  or “[0.0,0.25,71.8]”.
# ─────────────────────────────────────────────────────────────
def parse_array_string(s: str) -> np.ndarray:
    """
    Convert a bracketed string of numbers into a 1-D float array.
    Ignores any non-numeric junk such as Unicode ellipses (…).
    Returns an empty array if *no* numbers are found.
    """
    # 1. Trim outer brackets if present
    core = s.strip()
    if core.startswith("[") and core.endswith("]"):
        core = core[1:-1]

    # 2. Use regex to grab every float or int (handles “1”, “-2.3”, “4e-5”)
    number_regex = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    nums = re.findall(number_regex, core)

    # 3. Convert → float64 array
    return np.asarray(nums, dtype=np.float64)


def load_and_patch_dataset(
    csv_path, green_pct, red_pct, max_rois=50, drop_leading_zeros=False
):
    """
    Clean one ROI table.

    A surviving ROI must satisfy **all** of these QC rules
    1. no negative values in exp_data_means
    2. at least one data-point in exp_data_means
    3. not flat-lined at 0 beyond 20 h (index ≥ 80)
    The routine then trims every trace to ≤ 48 h (192 points),
    rebuilds the green/red schedules, and returns the patched
    DataFrame—or None if nothing survives.

    Parameters
    ----------
    drop_leading_zeros : bool, optional
        If True, skip leading zeros and start from the first non-zero value.
        If False (default), keep all data including leading zeros.
    """
    TMP_MAX_LEN = 192  # 48 h × 4 samples/h
    ZERO_PLATEAU = 80  # 20 h × 4 samples/h

    tmp = pd.read_csv(csv_path)
    tmp.index = ROW_NAMES[:-1]  # reserve last row for dataset_name

    valid_cols = []
    for c in (col for col in tmp.columns if str(col).isdigit()):
        try:
            means_arr = parse_array_string(tmp.at["exp_data_means", c])
        except Exception as err:
            print(f"⚠️  {csv_path} ROI {c}: parse error ({err}) — skipped.")
            continue

        # ── QC-1: negatives? ───────────────────────────────────────────────
        if np.any(means_arr < 0):
            continue

        # ── QC-2: empty trace? ─────────────────────────────────────────────
        if means_arr.size == 0:
            continue

        # ── QC-3: zero plateau beyond 20 h? ────────────────────────────────
        if means_arr.size > ZERO_PLATEAU and np.any(means_arr[ZERO_PLATEAU:] == 0):
            continue

        valid_cols.append(c)

    if not valid_cols:
        print(f"⚠️  {csv_path}: no ROI passed the QC filters — skipped.")
        return None

    # take ≤ max_rois of the "good" columns
    sampled_cols = random.sample(valid_cols, min(max_rois, len(valid_cols)))
    tmp = tmp[sampled_cols]

    # we'll re-patch schedule rows column-by-column once we know n_tpts
    for c in sampled_cols:
        # --------------------
        #  (1)  Trim *all* per-tp vectors to 192 if needed
        # --------------------
        time_arr = parse_array_string(tmp.at["time", c])
        means_arr = parse_array_string(tmp.at["exp_data_means", c])
        stds_arr = parse_array_string(tmp.at["exp_data_stds", c])
        data_tpts = parse_array_string(tmp.at["data_tpoints", c])

        # Handle leading zeros based on parameter
        if drop_leading_zeros:
            non_zero_mask = means_arr > 0
            if np.any(non_zero_mask):
                first_valid = np.argmax(non_zero_mask)
                time_arr = time_arr[first_valid:]
                means_arr = means_arr[first_valid:]
                stds_arr = stds_arr[first_valid:]
                data_tpts = data_tpts[first_valid:]
                fs_arr = parse_array_string(tmp.at["firstswitch", c])[first_valid:]
                ss_arr = parse_array_string(tmp.at["secondswitch", c])[first_valid:]
                tmp.at["firstswitch", c] = str(list(fs_arr))
                tmp.at["secondswitch", c] = str(list(ss_arr))

        if time_arr.size > TMP_MAX_LEN:
            slice_ = slice(0, TMP_MAX_LEN)
            time_arr = time_arr[slice_]
            means_arr = means_arr[slice_]
            stds_arr = stds_arr[slice_]
            data_tpts = data_tpts[slice_]

            # update firstswitch / secondswitch too
            fs_arr = parse_array_string(tmp.at["firstswitch", c])[:TMP_MAX_LEN]
            ss_arr = parse_array_string(tmp.at["secondswitch", c])[:TMP_MAX_LEN]
            tmp.at["firstswitch", c] = str(list(fs_arr))
            tmp.at["secondswitch", c] = str(list(ss_arr))

            # keep t0_tf[0] unchanged but set t0_tf[1] = new last time-point
            t0_tf = parse_array_string(tmp.at["t0_tf", c])
            t0_tf[1] = time_arr[-1]
            tmp.at["t0_tf", c] = str(list(t0_tf))

        # overwrite the rows with (possibly) truncated data
        tmp.at["time", c] = str(list(time_arr))
        tmp.at["exp_data_means", c] = str(list(means_arr))
        tmp.at["exp_data_stds", c] = str(list(stds_arr))
        tmp.at["data_tpoints", c] = str(list(data_tpts))

        # --------------------
        #  (2)  Patch schedules so every ROI has vectors of the
        #       right length and the correct green/red %s
        # --------------------
        n_tpts = time_arr.size
        red_vec = str(list(np.full(n_tpts, red_pct)))
        green_vec = str(list(np.full(n_tpts, green_pct)))

        tmp.at["schedule_R_COMPLETE", c] = red_vec
        tmp.at["schedule_G_COMPLETE", c] = green_vec

        # coarse (2-step) schedule rows are the same for every column
        tmp.at["schedule_R", c] = str([red_pct])
        tmp.at["schedule_G", c] = str([green_pct])

    # --------------------
    #  (3)  Annotate provenance
    # --------------------
    dataset_key = os.path.basename(csv_path).split("_")[-1].split(".")[0]
    tmp.loc["dataset_name"] = dataset_key

    return tmp


def lookup_light_schedule(exp_id: str, t: np.ndarray):
    """Return the green and red-light intensity vectors for an experiment."""
    try:
        green_frac, red_frac = TIMELAPSE_INFORMATION[exp_id][1]
    except KeyError:
        raise ValueError(f"Unknown experiment ID '{exp_id}'.")
    I_G = np.full_like(t, green_frac, dtype=np.float64)
    I_R = np.full_like(t, red_frac, dtype=np.float64)
    return I_G, I_R


# --- Numba-jitted helper functions ---
# @njit(inline="always")
# def _bin_search(t_arr, t_val):
#    left, right = 0, t_arr.size - 1
#    while left < right:
#        mid = (left + right) // 2
#        if t_arr[mid] < t_val:
#            left = mid + 1
#        else:
#            right = mid
#    return left


# @njit(inline="always")
# def light_interp_arr(t_arr, g_arr, r_arr, t_val):
#    idx = _bin_search(t_arr, t_val)
#    return g_arr[idx], r_arr[idx]


@njit(inline="always")
def faster_light_interp(t_val, t_start, time_step, g_arr, r_arr):
    """Calculates light intensity using fast O(1) arithmetic."""
    idx = int((t_val - t_start) / time_step)
    safe_idx = min(max(0, idx), g_arr.size - 1)
    return g_arr[safe_idx], r_arr[safe_idx]


# --- Data Structures for Numba-accelerated loops ---

# This struct holds the data passed to the LSODA ODE solver.
# Using int64 for pointers and explicit casting is the most robust approach.
# user_data_dtype = types.Record.make_c_struct(
#    [
#        ("p_p", types.int64),
#        ("len_p", types.int64),
#        ("t_light_p", types.int64),
#        ("len_t_light", types.int64),
#        ("g_light_p", types.int64),
#        ("len_g_light", types.int64),
#        ("r_light_p", types.int64),
#        ("len_r_light", types.int64),
#    ]
# )

user_data_dtype = types.Record.make_c_struct(
    [
        ("p_p", types.int64),
        ("len_p", types.int64),
        ("g_light_p", types.int64),
        ("len_g_light", types.int64),
        ("r_light_p", types.int64),
        ("len_r_light", types.int64),
        ("t_start", types.float64),
        ("time_step", types.float64),
    ]
)

# A jitclass to hold all data for a single ROI trace in a Numba-friendly format.
# spec = [
#    ("t", types.float64[:]),
#    ("y", types.float64[:]),
#    ("u0", types.float64[:]),
#    ("I_G", types.float64[:]),
#    ("I_R", types.float64[:]),
# ]


# @jitclass(spec)
# class ROIData:
#    def __init__(self, t, y, u0, I_G, I_R):
#        self.t = t
#        self.y = y
#        self.u0 = u0
#        self.I_G = I_G
#        self.I_R = I_R

spec = [
    ("t", types.float64[:]),
    ("y", types.float64[:]),
    ("u0", types.float64[:]),
    ("I_G", types.float64[:]),
    ("I_R", types.float64[:]),
    ("t_start", types.float64),  # Add starting time
    ("time_step", types.float64),  # Add the specific time step
]


@jitclass(spec)
class ROIData:
    def __init__(self, t, y, u0, I_G, I_R, t_start, time_step):
        self.t = t
        self.y = y
        self.u0 = u0
        self.I_G = I_G
        self.I_R = I_R
        self.t_start = t_start
        self.time_step = time_step


# def create_wrapped_rhs(rhs_func, args_dtype):
#    """Creates a C-callable wrapper for a Numba JIT-compiled RHS function."""
#    jitted_rhs = njit(rhs_func)
#    cfunc_sig = types.void(
#        types.double,
#        types.CPointer(types.double),
#        types.CPointer(types.double),
#        types.CPointer(args_dtype),
#    )
#
#    @cfunc(cfunc_sig, nopython=True)
#    def wrapped_rhs(t, u, du, user_data_p):
#        user_data = carray(user_data_p, 1)[0]
#        p = carray(
#            address_as_void_pointer(user_data.p_p), (user_data.len_p,), dtype=np.float64
#        )
#        t_light = carray(
#            address_as_void_pointer(user_data.t_light_p),
#            (user_data.len_t_light,),
#            dtype=np.float64,
#        )
#        g_light = carray(
#            address_as_void_pointer(user_data.g_light_p),
#            (user_data.len_g_light,),
#            dtype=np.float64,
#        )
#        r_light = carray(
#            address_as_void_pointer(user_data.r_light_p),
#            (user_data.len_r_light,),
#            dtype=np.float64,
#        )
#        jitted_rhs(t, u, du, p, t_light, g_light, r_light)
#
#    return wrapped_rhs


def create_wrapped_rhs(rhs_func, args_dtype):
    """Creates a C-callable wrapper for a Numba JIT-compiled RHS function."""
    jitted_rhs = njit(rhs_func)
    cfunc_sig = types.void(
        types.double,
        types.CPointer(types.double),
        types.CPointer(types.double),
        types.CPointer(args_dtype),
    )

    @cfunc(cfunc_sig, nopython=True)
    def wrapped_rhs(t, u, du, user_data_p):
        user_data = carray(user_data_p, 1)[0]
        p = carray(
            address_as_void_pointer(user_data.p_p), (user_data.len_p,), dtype=np.float64
        )
        g_light = carray(
            address_as_void_pointer(user_data.g_light_p),
            (user_data.len_g_light,),
            dtype=np.float64,
        )
        r_light = carray(
            address_as_void_pointer(user_data.r_light_p),
            (user_data.len_r_light,),
            dtype=np.float64,
        )
        # Pass the new start time and time step to the ODE function
        jitted_rhs(
            t, u, du, p, g_light, r_light, user_data.t_start, user_data.time_step
        )

    return wrapped_rhs


# def dIAdt_reduced(t, u, du, p, t_light, g_light, r_light):
#    """Reduced ODE model."""
#    I_G, I_R = light_interp_arr(t_light, g_light, r_light, t)
#    a1, d_p, mu_max, t_half, a3, a4, a5, a6 = p
#    a2 = d_p + mu_max / (1 + np.exp(mu_max * (t - t_half)))
#    prod = (I_G * a3 + 1) / (a3 * a5 * I_G + a4 * a6 + a4 * I_R + a5)
#    du[0] = a1 + prod - a2 * u[0]


def dIAdt_reduced(t, u, du, p, g_light, r_light, t_start, time_step):
    """Reduced ODE model with faster interpolation."""
    I_G, I_R = faster_light_interp(t, t_start, time_step, g_light, r_light)
    a1, d_p, mu_max, t_half, a3, a4, a5, a6 = p
    a2 = d_p + mu_max / (1 + np.exp(mu_max * (t - t_half)))
    prod = (I_G * a3 + 1) / (a3 * a5 * I_G + a4 * a6 + a4 * I_R + a5)
    du[0] = a1 + prod - a2 * u[0]


funcptr_reduced = create_wrapped_rhs(dIAdt_reduced, user_data_dtype)


def solve_model(theta_lin, t_vec, u0, I_G_vec, I_R_vec, model="reduced"):
    """Packs data and solves the ODE system."""
    if model != "reduced":
        raise ValueError("Only 'reduced' model is currently configured.")

    p = np.ascontiguousarray(theta_lin, dtype=np.float64)
    t_light = np.ascontiguousarray(t_vec, dtype=np.float64)
    g_light = np.ascontiguousarray(I_G_vec, dtype=np.float64)
    r_light = np.ascontiguousarray(I_R_vec, dtype=np.float64)

    user_data = np.array(
        [
            (
                p.ctypes.data,
                p.size,
                t_light.ctypes.data,
                t_light.size,
                g_light.ctypes.data,
                g_light.size,
                r_light.ctypes.data,
                r_light.size,
            )
        ],
        dtype=user_data_dtype,
    )

    sol, _ = lsoda(
        funcptr_reduced.address, u0, t_vec, data=user_data, atol=1e-6, rtol=1e-6
    )
    return sol.flatten()


# --- MCMC Probability Functions ---
@njit(fastmath=True)
def convert_to_log_bounds_nb(lower, upper):
    return np.log10(lower), np.log10(upper)


@njit(fastmath=True)
def log_prior_mixed_nb(p_log, lower_log, upper_log, modes, mu_log, sig_log):
    logp = 0.0
    for i in range(p_log.size):
        if not (lower_log[i] <= p_log[i] <= upper_log[i]):
            return -np.inf
        if modes[i] == PRIOR_GAUSS:
            diff = p_log[i] - mu_log[i]
            logp += -0.5 * (diff / sig_log[i]) ** 2
    return logp


@njit(fastmath=True)
def log_likelihood_nb_jit(y_model, y_data, log_sigma):
    sigma_sq = (10**log_sigma) ** 2
    return -0.5 * np.sum(
        (y_model - y_data) ** 2 / sigma_sq + np.log(2 * np.pi * sigma_sq)
    )


@njit(parallel=True)
def parallel_log_prob(
    theta_log, lower_log, upper_log, prior_modes, mu_log, sig_log, big_dataset, funcptr
):
    """
    A fully JIT-compiled and parallelized log-probability function.
    This is the new high-performance core of the MCMC.
    """
    # 1. Log Prior Calculation
    logp = 0.0
    for i in range(theta_log.size):
        if not (lower_log[i] <= theta_log[i] <= upper_log[i]):
            return -np.inf
        if prior_modes[i] == PRIOR_GAUSS:
            diff = theta_log[i] - mu_log[i]
            logp += -0.5 * (diff / sig_log[i]) ** 2

    # 2. Log Likelihood Calculation (Parallelized over all ROIs)
    theta_lin = 10.0 ** theta_log[:-1]
    log_sigma = theta_log[-1]
    sigma_sq = (10**log_sigma) ** 2
    log_sigma_term = np.log(2 * np.pi * sigma_sq)

    ll = 0.0
    solve_failed = np.zeros(1, dtype=np.bool_)

    # This loop runs in parallel across all available CPU cores
    for i in prange(len(big_dataset)):
        # If another thread has failed, stop work to exit quickly
        if solve_failed[0]:
            continue

        rec = big_dataset[i]

        # Pack data for the solver
        # user_data = np.empty(1, dtype=user_data_dtype)
        # user_data[0].p_p = np.int64(theta_lin.ctypes.data)
        # user_data[0].len_p = theta_lin.size
        # user_data[0].t_light_p = np.int64(rec.t.ctypes.data)
        # user_data[0].len_t_light = rec.t.size
        # user_data[0].g_light_p = np.int64(rec.I_G.ctypes.data)
        # user_data[0].len_g_light = rec.I_G.size
        # user_data[0].r_light_p = np.int64(rec.I_R.ctypes.data)
        # user_data[0].len_r_light = rec.I_R.size

        # NEW - Pack data for the solver
        user_data = np.empty(1, dtype=user_data_dtype)
        user_data[0].p_p = np.int64(theta_lin.ctypes.data)
        user_data[0].len_p = theta_lin.size
        user_data[0].g_light_p = np.int64(rec.I_G.ctypes.data)
        user_data[0].len_g_light = rec.I_G.size
        user_data[0].r_light_p = np.int64(rec.I_R.ctypes.data)
        user_data[0].len_r_light = rec.I_R.size
        user_data[0].t_start = rec.t_start
        user_data[0].time_step = rec.time_step

        # Solve the ODE
        sol, success = lsoda(
            funcptr, rec.u0, rec.t, data=user_data, atol=1e-8, rtol=1e-8
        )

        # --- Robustness Check ---
        # If the solver fails or returns non-finite values (NaN/inf), flag it.
        if not success or not np.all(np.isfinite(sol)):
            solve_failed[0] = True
            continue  # Stop processing this ROI

        # If successful, calculate and accumulate the log-likelihood
        y_model = sol.flatten()
        if y_model.shape == rec.y.shape:
            residual_sq = (y_model - rec.y) ** 2
            ll += -0.5 * np.sum(residual_sq / sigma_sq + log_sigma_term)
        else:
            solve_failed[0] = True  # Shape mismatch is also a failure
            continue

    # If any solve failed, reject the entire parameter set
    if solve_failed[0]:
        return -np.inf

    return logp + ll


def log_prob_parallel_wrapper(
    theta_log, lower_log, upper_log, big_dataset, prior_modes, mu_log, sig_log, funcptr
):
    """A simple Python wrapper to allow emcee to call the JIT-compiled function."""
    return parallel_log_prob(
        theta_log,
        lower_log,
        upper_log,
        prior_modes,
        mu_log,
        sig_log,
        big_dataset,
        funcptr,
    )


# --- MCMC Helper Functions (largely unchanged) ---
def make_initial_walkers(
    nwalkers, lower_log, upper_log, prior_modes, mu_log, sig_log, rng
):
    ndim = lower_log.size
    walkers = np.empty((nwalkers, ndim), dtype=np.float64)
    for j in range(ndim):
        lo, hi = lower_log[j], upper_log[j]
        if prior_modes[j] == PRIOR_UNIFORM:
            walkers[:, j] = rng.uniform(lo, hi, size=nwalkers)
        elif prior_modes[j] == PRIOR_GAUSS:
            scale = max(sig_log[j], 1e-10)
            a, b = (lo - mu_log[j]) / scale, (hi - mu_log[j]) / scale
            walkers[:, j] = truncnorm(a, b, loc=mu_log[j], scale=scale).rvs(
                size=nwalkers, random_state=rng
            )
    return walkers


# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Main function to run the entire data processing and MCMC pipeline."""

    base_path = "/Users/adaravena/Library/CloudStorage/GoogleDrive-adaravena94@gmail.com/My Drive/Optogenetics_JUN2025/fluopti_fits/"
    os.chdir(base_path)
    print(f"Current directory: {os.getcwd()}")

    data_analysis_dir = os.path.join(base_path, "data_analysis")
    PROCESSED_DATASET_DIR = data_analysis_dir

    # --- Data Loading (largely unchanged from original) ---
    print("Loading and processing datasets...")
    df = pd.read_csv(os.path.join(PROCESSED_DATASET_DIR, "AGGREGATED_ROI_DATA.csv"))

    # --- Create Numba-Friendly Dataset ---
    big_dataset = List()  # Use a Numba typed list
    for col in tqdm.tqdm(df.columns[1:], desc="Pre-processing ROI data"):
        exp_id, _ = col.split("_")

        t_vec = parse_array_string(df.at[1, col])
        y_vec = parse_array_string(df.at[2, col])

        # -----------------------------------------------------------
        # 0)  drop leading padded zeros
        # -----------------------------------------------------------
        # non_zero_mask = y_vec > 0  # boolean
        # if not np.any(non_zero_mask):
        #    continue  # skip empty trace
        # first_valid = np.argmax(non_zero_mask)  # first True index
        # to bypass the dropping of zeros
        first_valid = 0

        t_vec = t_vec[first_valid:]
        y_vec = y_vec[first_valid:]

        # Data cleaning: remove duplicates
        if len(t_vec) != len(np.unique(t_vec)):
            unique_indices = np.sort(np.unique(t_vec, return_index=True)[1])
            t_vec = t_vec[unique_indices]
            y_vec = y_vec[unique_indices]

        if t_vec.size < 2:
            continue

        # Ensure arrays are contiguous
        t_vec = np.ascontiguousarray(t_vec)
        y_vec = np.ascontiguousarray(y_vec)

        I_G_vec, I_R_vec = lookup_light_schedule(exp_id, t_vec)
        u0 = np.array([y_vec[0]], np.float64)

        # Create an instance of the jitclass and append to the typed list
        # roi_data = ROIData(t_vec, y_vec, u0, I_G_vec, I_R_vec)
        # big_dataset.append(roi_data)

        # NEW - Calculate the specific start time and time step for this ROI
        t_start = t_vec[0]
        time_step = t_vec[1] - t_vec[0]

        # Create an instance of the jitclass and append to the typed list
        roi_data = ROIData(t_vec, y_vec, u0, I_G_vec, I_R_vec, t_start, time_step)
        big_dataset.append(roi_data)

    print(
        f"Loaded and cached {len(big_dataset)} colony traces into Numba-ready format."
    )

    # --- Setup for ODE and MCMC ---

    # Use the globally defined funcptr_reduced for the ODE function pointer
    funcptr = np.int64(funcptr_reduced.address)

    # MCMC parameter set.up (unchanged)
    # lower_bound_reduced = np.array([1e-3, 1e-2, 1e-1, 10.0, 1e0, 1e-2, 1e-3, 1e-3])
    # upper_bound_reduced = np.array([1e1, 5e-1, 5e0, 20.0, 1e2, 1e1, 1e3, 1e3])
    lower_bound_reduced = np.array([1e-6, 1e-2, 1e-3, 10.0, 1e-3, 1e-3, 1e-3, 1e-3])
    upper_bound_reduced = np.array([2.5e1, 1e0, 1e3, 20.0, 1e3, 1e3, 1e3, 1e3])
    data_reduced_guess = np.array(
        [1.25, 0.06, 1.0, 13.0, 40.0, 0.588235294, 0.0882352941, 0.02012]
    )
    n_params = len(lower_bound_reduced)
    ndim = n_params + 1
    nwalkers = 32 * ndim
    nsteps = 50_000
    lower_log = np.append(np.log10(lower_bound_reduced), np.log10(0.1))
    upper_log = np.append(np.log10(upper_bound_reduced), np.log10(5.0))
    # prior_modes = np.array([PRIOR_GAUSS] * n_params + [PRIOR_UNIFORM])
    # Make a uniform prior for the all params
    prior_modes = np.array([PRIOR_UNIFORM] * n_params + [PRIOR_UNIFORM])
    # prior_modes = np.array(
    #    [
    #        PRIOR_GAUSS,  # a1
    #        PRIOR_GAUSS,  # d_p
    #        PRIOR_GAUSS,  # mu_max
    #        PRIOR_UNIFORM,  # t_half
    #        PRIOR_UNIFORM,  # a3
    #        PRIOR_UNIFORM,  # a4
    #        PRIOR_UNIFORM,  # a5
    #        PRIOR_UNIFORM,  # a6
    #        PRIOR_UNIFORM,  # log_sigma (last parameter)
    #    ]
    # )
    mu_log_reduced = np.append(np.log10(data_reduced_guess), np.log10(5.0))
    # sigmas_log = np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
    sigmas_log = np.array([1.96] * n_params + [1.96])
    rng = np.random.default_rng()
    p0_walkers = make_initial_walkers(
        nwalkers, lower_log, upper_log, prior_modes, mu_log_reduced, sigmas_log, rng
    )
    print("Initial walker positions generated.")

    # --- Numba Warm-up ---
    # Call the log probability function once to trigger JIT compilation
    # before starting the main MCMC run.
    print("Warming up the Numba JIT compiler...")
    for i in range(25):
        # This is a dummy call to ensure the function is compiled.
        # It will not be used in the actual MCMC run.
        log_prob_parallel_wrapper(
            p0_walkers[0],
            lower_log,
            upper_log,
            big_dataset,
            prior_modes,
            mu_log_reduced,
            sigmas_log,
            funcptr,
        )
    print("Compiler is ready.")

    # --- Run MCMC Sampler with Numba Parallelization ---
    # NOTE: The emcee pool is removed. Parallelism is now handled by Numba's @njit(parallel=True).
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_parallel_wrapper,
        moves=[
            (emcee.moves.DEMove(sigma=1e-2), 0.5),
            (emcee.moves.DESnookerMove(), 0.5),
        ],
        args=(
            lower_log,
            upper_log,
            big_dataset,
            prior_modes,
            mu_log_reduced,
            sigmas_log,
            funcptr,
        ),
    )

    print(f"Running MCMC with Numba backend. Executing with {nwalkers} walkers...")
    start_time = time.time()
    sampler.run_mcmc(p0_walkers, nsteps, progress=True)
    end_time = time.time()
    print(f"MCMC completed in {(end_time - start_time) / 60:.2f} minutes.")

    # **CORRECTION**: To make the sampler pickleable, we remove the reference
    # to the log probability function, which holds the un-pickleable Numba objects.
    # All important results (the chains and log probabilities) are stored separately
    # within the sampler and are not affected by this.
    sampler.log_prob_fn = None

    # Now, save the modified sampler object
    output_filename = "MCMC_07112025_UNIFORMPRIORS_WITH_ZEROVALS.pkl"
    output_path = os.path.join(data_analysis_dir, output_filename)
    with open(output_path, "wb") as f:
        import pickle

        pickle.dump(sampler, f)
    print(f"✓ Sampler results saved to {output_path}")


if __name__ == "__main__":
    main()
