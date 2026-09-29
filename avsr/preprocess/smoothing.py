"""Temporal smoothing of landmark trajectories.

Landmark detectors jitter by a pixel or two from frame to frame. Cropping with
the raw positions turns that jitter into apparent motion of the whole mouth
region, which the model cannot tell apart from articulation.

All functions take a trajectory of shape [T, D] and return the same shape.
"""

import numpy as np


def moving_average(x, window=12):
    """Centred moving average that shrinks its window near the clip edges.

    This is the smoothing used by Auto-AVSR (window of 12 frames).
    """
    x = np.asarray(x, dtype=np.float64)
    half = window // 2
    out = np.empty_like(x)
    for t in range(len(x)):
        margin = min(half, t, len(x) - 1 - t)
        out[t] = x[t - margin : t + margin + 1].mean(axis=0)
    return out.astype(np.float32)


def kalman_smooth(x, process_var=1.0, measurement_var=4.0):
    """Constant-velocity Kalman filter followed by a Rauch-Tung-Striebel pass.

    Each dimension is treated independently. ``process_var`` is the variance
    of the acceleration noise (px/frame^2)^2 and ``measurement_var`` the
    variance of the detector noise (px^2). Because preprocessing is offline,
    the backward pass removes the lag a causal filter would introduce.
    """
    x = np.asarray(x, dtype=np.float64)
    n, dims = x.shape
    if n < 3:
        return x.astype(np.float32)

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = process_var * np.array([[0.25, 0.5], [0.5, 1.0]])
    R = np.array([[measurement_var]])
    eye = np.eye(2)

    out = np.empty_like(x)
    for d in range(dims):
        z = x[:, d]
        mean_pred = np.zeros((n, 2))
        cov_pred = np.zeros((n, 2, 2))
        mean_filt = np.zeros((n, 2))
        cov_filt = np.zeros((n, 2, 2))

        mean = np.array([z[0], 0.0])
        cov = np.diag([measurement_var, 100.0])
        for t in range(n):
            if t > 0:
                mean = F @ mean
                cov = F @ cov @ F.T + Q
            mean_pred[t], cov_pred[t] = mean, cov
            gain = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + R)
            mean = mean + (gain @ (z[t] - H @ mean)).ravel()
            cov = (eye - gain @ H) @ cov
            mean_filt[t], cov_filt[t] = mean, cov

        smoothed = mean_filt.copy()
        for t in range(n - 2, -1, -1):
            gain = cov_filt[t] @ F.T @ np.linalg.inv(cov_pred[t + 1])
            smoothed[t] = mean_filt[t] + gain @ (smoothed[t + 1] - mean_pred[t + 1])
        out[:, d] = smoothed[:, 0]
    return out.astype(np.float32)


def one_euro(x, fps=25.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
    """One-Euro filter (Casiez et al., CHI 2012).

    A causal low-pass filter whose cutoff rises with speed: strong smoothing
    when the head is still, little lag when it moves. It is the only option
    here that also works frame by frame in a streaming setting.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x.astype(np.float32)

    def alpha(cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    out = np.empty_like(x)
    out[0] = x[0]
    dx_prev = np.zeros(x.shape[1])
    for t in range(1, len(x)):
        dx = (x[t] - out[t - 1]) * fps
        a_d = alpha(d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        a = alpha(min_cutoff + beta * np.abs(dx_hat))
        out[t] = a * x[t] + (1 - a) * out[t - 1]
        dx_prev = dx_hat
    return out.astype(np.float32)


def smooth_trajectory(x, method="kalman", **kwargs):
    """Dispatch to a smoothing method by name."""
    if method == "none":
        return np.asarray(x, dtype=np.float32)
    if method == "moving_average":
        return moving_average(x, **kwargs)
    if method == "kalman":
        return kalman_smooth(x, **kwargs)
    if method == "one_euro":
        return one_euro(x, **kwargs)
    raise ValueError(f"Unknown smoothing method: {method}")
