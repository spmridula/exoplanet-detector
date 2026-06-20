import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


class LightCurvePreprocessor:
    """
    Cleans a raw light curve DataFrame (time, flux, flux_err) for ML use.

    Parameters
    ----------
    sigma_clip_threshold : float
        Reject flux points more than N sigma from the median. Default: 5.
    normalize_method : str
        "median" (recommended) or "mean".
    interpolate_gaps : bool
        Fill gaps with linear interpolation. Default: True.
    flatten_window : int
        Window length for Savitzky-Golay trend removal. Must be odd. Default: 401.
    flatten_polyorder : int
        Polynomial order for Savitzky-Golay filter. Default: 3.
    """

    def __init__(
        self,
        sigma_clip_threshold: float = 5.0,
        normalize_method: str = "median",
        interpolate_gaps: bool = True,
        flatten_window: int = 401,
        flatten_polyorder: int = 3,
    ):
        self.sigma_clip_threshold = sigma_clip_threshold
        self.normalize_method = normalize_method
        self.interpolate_gaps = interpolate_gaps
        self.flatten_window = flatten_window
        self.flatten_polyorder = flatten_polyorder

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: time, flux, flux_err

        Returns
        -------
        pd.DataFrame with cleaned flux values
        """
        df = df.copy()
        n_start = len(df)

        df = self._remove_nans(df)
        df = self._sigma_clip(df)
        df = self._normalize(df)

        if self.interpolate_gaps:
            df = self._interpolate_gaps(df)

        df = self._flatten_trend(df)

        logger.debug(f"Preprocessing: {n_start} → {len(df)} points ({n_start - len(df)} removed)")
        return df

    def _remove_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = np.isfinite(df["flux"]) & np.isfinite(df["time"])
        return df[mask].reset_index(drop=True)

    def _sigma_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        flux = df["flux"].values
        median = np.median(flux)
        std = np.std(flux)
        upper_mask = flux < (median + self.sigma_clip_threshold * std)
        lower_mask = flux > (median - self.sigma_clip_threshold * 2 * std)
        return df[upper_mask & lower_mask].reset_index(drop=True)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df["flux_raw"] = df["flux"].copy()
        flux = df["flux"].values
        baseline = np.median(flux) if self.normalize_method == "median" else np.mean(flux)
        if baseline == 0:
            return df
        df["flux"] = flux / baseline
        df["flux_err"] = df["flux_err"] / baseline
        return df

    def _interpolate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        time = df["time"].values
        flux = df["flux"].values
        dt_median = np.median(np.diff(time))
        t_regular = np.arange(time[0], time[-1], dt_median)
        flux_interp = np.interp(t_regular, time, flux)
        err_interp = np.interp(t_regular, time, df["flux_err"].values)
        return pd.DataFrame({"time": t_regular, "flux": flux_interp, "flux_err": err_interp})

    def _flatten_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        flux = df["flux"].values
        window = min(self.flatten_window, len(flux) - 1)
        if window % 2 == 0:
            window -= 1
        if window < self.flatten_polyorder + 2:
            return df
        trend = savgol_filter(flux, window_length=window, polyorder=self.flatten_polyorder)
        df["flux"] = flux / trend
        return df


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold a light curve onto a given orbital period.

    This stacks all transit observations on top of each other,
    dramatically improving signal-to-noise for faint dips.

    Parameters
    ----------
    time : np.ndarray        Time values (days)
    flux : np.ndarray        Normalized flux values
    period : float           Orbital period in days
    t0 : float               Reference transit time (mid-transit)
    n_bins : int             Number of phase bins. Default: 2000.

    Returns
    -------
    phase_bins : np.ndarray  shape (n_bins,)  — phase values [0, 1)
    binned_flux : np.ndarray shape (n_bins,)  — median flux per bin
    """
    phase = ((time - t0) % period) / period
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(phase_sorted, bin_edges) - 1, 0, n_bins - 1)

    binned_flux = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_indices == i
        binned_flux[i] = np.median(flux_sorted[mask]) if mask.sum() > 0 else 1.0

    phase_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return phase_centers, binned_flux