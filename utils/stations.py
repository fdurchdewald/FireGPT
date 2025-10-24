# utils/stations.py
"""
Helpers for managing fire stations.
Includes functions to parse LLM responses for station recommendations and to sort stations by distance.
"""
import pandas as pd
from utils import extract_metadata as em

BF_REGEX = r"Berufsfeuerwehr|^Feuerwehr$|^Hauptfeuerwache$"


def sorted_stations(lng: float, lat: float, n: int = 5) -> pd.DataFrame:
    """
    Sorts fire stations by distance from the given coordinates.
    Args:
        lng: Longitude of the location
        lat: Latitude of the location
        n: Number of stations to return (default: 5)
    Returns:
        DataFrame with sorted fire stations, including their distance from the location
    """
    diag = em.diagnostics(lng, lat, n_stations=n)
    stations = diag["stations"].copy()

    bf_mask = stations["name"].str.contains(BF_REGEX, case=False, regex=True, na=False)
    bf, other = stations[bf_mask], stations[~bf_mask]

    return (
        pd.concat([bf, other])
        .drop_duplicates(subset="name", keep="first")
        .sort_values("distance_m")
        .reset_index(drop=True)
    )
