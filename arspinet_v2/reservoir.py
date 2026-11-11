"""Batched leaky integrate-and-fire reservoir.

The update rule is the v1 operator (chapter5Experiments/experiment_zero.py), unchanged:

    I_t   = W_in x_t + W_rec s_{t-1}
    m_t   = (1 - beta) m_{t-1} (1 - s_{t-1}) + I_t
    s_t   = [m_t >= theta]
    m_t   = max(m_t - theta s_t, 0)

What v2 adds is control over the three things the v1 code fixed silently:

* ``rho``     spectral radius of W_rec. v1 used 0.9. ``rho=0`` removes recurrence.
* sharing     v1's committed runner drew a different reservoir for every electrode
              (seed + 17 * channel) although the dissertation describes one shared
              reservoir. ``shared=True`` uses one reservoir for all electrodes.
* ``spiking`` ``False`` gives the matched non-spiking control: the same input weights
              and leak, no threshold, no reset, no recurrence, rectified output.
"""
from __future__ import annotations

import numpy as np

N_RES = 256
BETA = 0.05
THETA = 0.5
V1_RHO = 0.9


class LIFReservoir:
    def __init__(self, n_res=N_RES, beta=BETA, theta=THETA, rho=V1_RHO, seed=42,
                 spiking=True):
        rng = np.random.RandomState(seed)
        self.n_res, self.beta, self.theta = int(n_res), float(beta), float(theta)
        self.rho, self.seed, self.spiking = float(rho), int(seed), bool(spiking)
        lim_in = np.sqrt(6.0 / (1 + n_res))
        self.w_in = rng.uniform(-lim_in, lim_in, (n_res, 1))[:, 0]
        lim_rec = np.sqrt(6.0 / (2 * n_res))
        w = rng.uniform(-lim_rec, lim_rec, (n_res, n_res))
        radius = np.abs(np.linalg.eigvals(w)).max()
        # Same draw as v1 for every rho, so rho only rescales the recurrent matrix.
        self.w_rec = w * (self.rho / radius) if (radius > 0 and self.rho > 0) else None

    # ------------------------------------------------------------------ simulation
    def run(self, x, mem0=None, n_bins=None, return_spikes=False, return_rate=False,
            use_recurrence=True):
        """Drive the reservoir with a batch of scalar signals.

        x : (B, T) array. Returns a dict with any of
            ``counts`` (B, n_bins, N)  spike counts (or mean rectified state if not spiking)
            ``spikes`` (B, T, N) bool
            ``rate``   (B, T) population rate
            ``mem``    (B, N) final membrane state
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must have shape (batch, time)")
        B, T = x.shape
        N = self.n_res
        mem = np.zeros((B, N)) if mem0 is None else np.array(mem0, dtype=np.float64)
        spike = np.zeros((B, N))
        w_rec_t = self.w_rec.T if (self.w_rec is not None and use_recurrence
                                   and self.spiking) else None
        out = {}
        if n_bins:
            bin_size = T // n_bins
            if bin_size < 1:
                raise ValueError("fewer time steps than bins")
            counts = np.zeros((B, n_bins, N), dtype=np.float32)
        if return_spikes:
            spikes = np.zeros((B, T, N), dtype=bool)
        if return_rate:
            rate = np.zeros((B, T))
        for t in range(T):
            current = x[:, t, None] * self.w_in[None, :]
            if self.spiking:
                if w_rec_t is not None:
                    current = current + spike @ w_rec_t
                mem = (1.0 - self.beta) * mem * (1.0 - spike) + current
                spike = (mem >= self.theta).astype(np.float64)
                mem = np.maximum(mem - spike * self.theta, 0.0)
                emit = spike
            else:
                mem = (1.0 - self.beta) * mem + current
                emit = np.maximum(mem, 0.0)
            if n_bins:
                b = t // bin_size
                if b < n_bins:          # v1 drops the T % n_bins trailing samples
                    counts[:, b, :] += emit
            if return_spikes:
                spikes[:, t, :] = spike > 0
            if return_rate:
                rate[:, t] = spike.mean(axis=1)
        if n_bins:
            if not self.spiking:
                counts /= bin_size
            out["counts"] = counts
        if return_spikes:
            out["spikes"] = spikes
        if return_rate:
            out["rate"] = rate
        out["mem"] = mem
        return out


def encode(raw, rho=V1_RHO, seed=42, shared=True, n_bins=6, spiking=True, chunk=4096,
           beta=BETA, theta=THETA):
    """Electrode-resolved binned spike counts.

    raw : (observations, time, channels). Returns (observations, channels, n_bins * N)
    float32, laid out bin-major exactly like v1's ``bsc6_encode``.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n_obs, T, n_ch = raw.shape
    out = np.empty((n_obs, n_ch, n_bins * N_RES), dtype=np.float32)
    if shared:
        res = LIFReservoir(rho=rho, seed=seed, spiking=spiking, beta=beta, theta=theta)
        flat = raw.transpose(0, 2, 1).reshape(n_obs * n_ch, T)
        flat_out = out.reshape(n_obs * n_ch, n_bins * N_RES)
        for lo in range(0, len(flat), chunk):
            c = res.run(flat[lo:lo + chunk], n_bins=n_bins)["counts"]
            flat_out[lo:lo + chunk] = c.reshape(len(c), -1)
    else:
        for ch in range(n_ch):       # v1 committed runner: seed + 17 * channel
            res = LIFReservoir(rho=rho, seed=seed + ch * 17, spiking=spiking,
                               beta=beta, theta=theta)
            c = res.run(raw[:, :, ch], n_bins=n_bins)["counts"]
            out[:, ch, :] = c.reshape(n_obs, -1)
    return out
