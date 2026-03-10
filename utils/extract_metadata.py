"""
Metadata Extraction Utilities for FireGPT

This module provides functions to extract and process geographical and environmental
metadata related to fire emergencies. It includes functionality for:

- Querying CORINE Land Cover (CLC) data to determine terrain types
- Finding nearby fire stations and helicopter resources
- Retrieving current and historical weather data
- Calculating fire risk based on weather conditions
- Locating nearby water bodies for fire suppression
- Determining the administrative region (state) of a location

These utilities support the FireGPT application by providing contextual information
for AI-assisted fire emergency response planning.
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import rasterio
import numpy as np
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from datetime import datetime, timedelta, UTC


from datetime import datetime, timedelta
from typing import List, Dict, Optional

from rasterio.windows import Window

from rasterio.warp import transform


import urllib.request
import urllib.parse
import json


INDEX_TO_CLC_CODE = [
    111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142,
    211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244,
    311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335,
    411, 412, 421, 422, 423, 511, 512, 521, 522, 523
]

CLC_CLASSES = {
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    121: "Industrial or commercial units",
    122: "Road and rail networks and associated land",
    123: "Port areas",
    124: "Airports",
    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",
    141: "Green urban areas",
    142: "Sport and leisure facilities",
    211: "Non‑irrigated arable land",
    212: "Permanently irrigated land",
    213: "Rice fields",
    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    223: "Olive groves",
    231: "Pastures",
    241: "Annual crops with permanent crops",
    242: "Complex cultivation patterns",
    243: "Agriculture with natural vegetation",
    244: "Agro‑forestry areas",
    311: "Broad‑leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grasslands",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland‑shrub",
    331: "Beaches, dunes, sands",
    332: "Bare rocks",
    333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",
    411: "Inland marshes",
    412: "Peat bogs",
    421: "Salt marshes",
    422: "Salines",
    423: "Intertidal flats",
    511: "Water courses",
    512: "Water bodies",
    521: "Coastal lagoons",
    522: "Estuaries",
    523: "Sea and ocean",
}

WATER_CODES = {511, 512, 521, 522, 523}

def query_clc(tif_path: Path, lon: float, lat: float) -> int | None:
    """
    Query the CORINE Land Cover (CLC) code at the specified coordinates.
    
    Args:
        tif_path: Path to the CLC GeoTIFF file
        lon: Longitude of the query point
        lat: Latitude of the query point
        
    Returns:
        The CLC code at the specified location, or None if no data is available
    """
    with rasterio.open(tif_path) as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
        val = next(src.sample([(x[0], y[0])]))
        clc_code = int(val[0])
        if 1 <= clc_code <= 44:
            clc_code = INDEX_TO_CLC_CODE[clc_code - 1]
        if src.nodata is not None and clc_code == src.nodata:
            return None
        return clc_code

def get_clc_metadata(tif_path: Path, lon: float, lat: float) -> dict | None:
    """
    Get CORINE Land Cover code and class name for the specified coordinates.
    
    Args:
        tif_path: Path to the CLC GeoTIFF file
        lon: Longitude of the query point
        lat: Latitude of the query point
        
    Returns:
        A tuple containing (clc_code, class_name), or None if no data is available
    """
    clc_code = query_clc(tif_path, lon, lat)
    if clc_code is None:
        return None
    class_name = CLC_CLASSES.get(clc_code, "Unknown/Not in 44‑class list")
    return clc_code, class_name


def get_nearest_firestations(file_path: str, lon: float, lat: float, n: int = 5) -> gpd.GeoDataFrame:
    """
    Find the nearest fire stations to the specified coordinates.
    
    Prioritizes professional fire departments (Berufsfeuerwehr) and excludes
    certain types of fire stations like industrial or airport fire departments.
    
    Args:
        file_path: Path to the fire stations data file (CSV or GeoPackage)
        lon: Longitude of the query point
        lat: Latitude of the query point
        n: Maximum number of fire stations to return
        
    Returns:
        GeoDataFrame containing the nearest fire stations with distance information,
        sorted with professional fire departments first, then by distance
        
    Raises:
        FileNotFoundError: If the specified file path does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".gpkg":
        gdf = gpd.read_file(file_path)
    else:
        df = pd.read_csv(file_path)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    gdf = gdf.to_crs("EPSG:3857")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    qx, qy = transformer.transform(lon, lat)
    center = Point(qx, qy)
    gdf["distance_m"] = gdf.geometry.distance(center)

    # Exclude specific fire station types
    exclude_terms = ["Betriebsfeuerwehr", "Gerätehaus", "Werkfeuerwehr", "Flughafenfeuerwehr", "Katastrophenschutzzentrum"]
    pattern_exclude = '|'.join(exclude_terms)
    gdf = gdf[~gdf["name"].str.contains(pattern_exclude, case=False, na=False)].copy()

    # Identify Berufsfeuerwehr including specific terms
    bf_terms = ["Berufsfeuerwehr", "^Feuerwehr$", "^Feuerwehrhaus$", "^Hauptfeuerwache$"]
    pattern_bf = '|'.join(bf_terms)
    bf_gdf = gdf[gdf["name"].str.contains(pattern_bf, case=False, regex=True, na=False)].copy()

    nearest = gdf.nsmallest(n, "distance_m").copy()

    # Ensure nearest Berufsfeuerwehr is included
    bf_nearest = bf_gdf.nsmallest(1, "distance_m")
    if not bf_nearest.empty and not bf_nearest.iloc[0]["name"] in nearest["name"].values:
        nearest = pd.concat([nearest, bf_nearest], ignore_index=True)

    nearest = nearest.set_geometry("geometry").to_crs("EPSG:4326")

    # am Ende von get_nearest_firestations(), direkt vor dem return
    nearest["is_bf"] = nearest["name"].str.contains(pattern_bf, case=False, regex=True, na=False)
    nearest = (
        nearest.sort_values(["is_bf", "distance_m"], ascending=[False, True])
            .drop(columns="is_bf")
    )

    return nearest



# After get_nearest_firestations
def get_nearest_helicopter(file_path: str, lon: float, lat: float, n: int = 1) -> gpd.GeoDataFrame:
    """
    Find the nearest helicopter resources to the specified coordinates.
    
    Args:
        file_path: Path to the helicopter data file (CSV)
        lon: Longitude of the query point
        lat: Latitude of the query point
        n: Maximum number of helicopter resources to return
        
    Returns:
        GeoDataFrame containing the nearest helicopter resources with distance information
        
    Raises:
        FileNotFoundError: If the specified file path does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    qx, qy = transformer.transform(lon, lat)
    center = Point(qx, qy)
    gdf["distance_m"] = gdf.geometry.distance(center)
    nearest = gdf.nsmallest(n, "distance_m").copy()
    nearest = nearest.set_geometry("geometry").to_crs("EPSG:4326")
    return nearest


def get_dwd_current_weather(lat: float, lon: float) -> dict | None:
    """
    Retrieve current weather data from the DWD (German Weather Service).
    
    Uses the BrightSky API to fetch current weather conditions at the specified location.
    
    Args:
        lat: Latitude of the query point
        lon: Longitude of the query point
        
    Returns:
        Dictionary containing current weather information including temperature,
        humidity, and precipitation, or None if the request fails
    """
    url = f"https://api.brightsky.dev/current_weather?lat={lat:.5f}&lon={lon:.5f}"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.load(r)
    except Exception as exc:
        print(f"[WARN] DWD current weather request failed: {exc}")
        return None

    observations = data.get("weather") or data.get("current_weather")
    if not observations:
        return None

    if isinstance(observations, dict):
        obs = observations  
    elif isinstance(observations, list) and observations and isinstance(observations[0], dict):
        obs = observations[0]  
    else:
        return None
    return {
        "station_id": obs.get("station_id"),
        "temperature_C": obs.get("temperature"),
        "rel_humidity_%": obs.get("relative_humidity"),
        "precip_mm_10min": obs.get("precipitation_10"),
        "timestamp": obs.get("timestamp")
    }


def get_dwd_weather_history(lat: float, lon: float, days: int = 5) -> list[dict]:
    """
    Retrieve historical weather data from the DWD for the specified location.
    
    Uses the BrightSky API to fetch weather data for the past number of days.
    
    Args:
        lat: Latitude of the query point
        lon: Longitude of the query point
        days: Number of past days to retrieve data for
        
    Returns:
        List of dictionaries containing historical weather observations
    """
    history: list[dict] = []
    for offset in range(1, days + 1):
        date_str = (datetime.now(UTC) - timedelta(days=offset)).strftime("%Y-%m-%d")
        url = (
            f"https://api.brightsky.dev/weather?"
            f"lat={lat:.5f}&lon={lon:.5f}&date={date_str}"
        )
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                data = json.load(r)
            day_obs = data.get("weather", [])
            if isinstance(day_obs, dict):
                day_obs = [day_obs]
            history.extend(day_obs)
        except Exception as exc:
            print(f"[WARN] Historical weather request failed for {date_str}: {exc}")
            continue

    return history

def compute_fire_risk(lat: float, lon: float) -> str:
    """
    Calculate the fire risk level based on recent weather conditions.
    
    Uses historical weather data to assess fire risk based on precipitation,
    relative humidity, and wind speed over the past two days.
    
    Args:
        lat: Latitude of the query point
        lon: Longitude of the query point
        
    Returns:
        String indicating fire risk level: "VERY HIGH", "HIGH", "MODERATE", "LOW", or "UNKNOWN"
    """
    # Wir ziehen nur die letzten 2 Tage heran
    history: List[Dict] = get_dwd_weather_history(lat, lon, days=2)
    if not history:
        return "UNKNOWN"

    p_sum = sum(obs.get("precipitation_1h", 0.0) for obs in history)

    rh_vals: List[float] = []
    wind_vals: List[float] = []

    for obs in history:
        ts = obs.get("timestamp")
        if ts:
            hour = int(ts[11:13])
            rh = obs.get("relative_humidity")
            if 12 <= hour < 18 and rh is not None:
                rh_vals.append(rh)

        ws = obs.get("wind_speed")
        if ws is not None:
            wind_vals.append(ws)

    if not rh_vals or not wind_vals:
        return "UNKNOWN"

    avg_rh = sum(rh_vals) / len(rh_vals)
    avg_wind = sum(wind_vals) / len(wind_vals)

    # Sehr einfache Schwellen
    if p_sum < 0.5 and avg_rh < 25 and avg_wind > 6:
        return "VERY HIGH"
    elif p_sum < 1.0 and avg_rh < 30:
        return "HIGH"
    elif p_sum < 2.0 and avg_rh < 40:
        return "MODERATE"
    else:
        return "LOW"




def get_nearest_waterbody(
        tif_path: Path,
        lon: float,
        lat: float,
        initial_radius_m: float = 1.0,
        max_radius_m: float = 1000000.0,
        step_radius_m: float = 10.0
) -> dict | None:
    """
    Find the nearest water body to the specified coordinates.
    
    Searches for water bodies (lakes, rivers, etc.) in the CLC data,
    starting from a small radius and expanding outward until a water
    body is found or the maximum search radius is reached.
    
    Args:
        tif_path: Path to the CLC GeoTIFF file
        lon: Longitude of the query point
        lat: Latitude of the query point
        initial_radius_m: Initial search radius in meters
        max_radius_m: Maximum search radius in meters
        step_radius_m: Increment to increase the search radius by on each iteration
        
    Returns:
        Dictionary containing information about the nearest water body including
        its code, class name, coordinates, and distance, or None if no water body
        is found within the maximum search radius
    """
    from rasterio.vrt import WarpedVRT
    # Prepare mapping and water index set
    mapping = np.array([0] + INDEX_TO_CLC_CODE, dtype=np.int16)
    raw_water_idx = ({idx + 1 for idx, c in enumerate(INDEX_TO_CLC_CODE) if c in WATER_CODES} | {48, 49, 50})
    # Open source and create a VRT in metric CRS
    with rasterio.open(tif_path) as src, \
         WarpedVRT(src, crs="EPSG:3857", resampling=Resampling.nearest) as vrt:
        transform = vrt.transform
        px_m_x = abs(transform.a)
        px_m_y = abs(transform.e)
        # Transformers for reprojection
        transformer_to = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        transformer_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        # Query point in metric coords
        qx, qy = transformer_to.transform(lon, lat)
        # Pixel indices of query in VRT (use metric coords)
        row0, col0 = vrt.index(qx, qy)
        radius = initial_radius_m
        # Iterative search with circular mask
        while radius <= max_radius_m:
            pix_x = max(1, int(round(radius / px_m_x)))
            pix_y = max(1, int(round(radius / px_m_y)))
            row_start = max(0, row0 - pix_y)
            col_start = max(0, col0 - pix_x)
            height = min(vrt.height - row_start, 2 * pix_y + 1)
            width = min(vrt.width - col_start, 2 * pix_x + 1)
            if height <= 0 or width <= 0:
                radius += step_radius_m
                continue
            window = Window(col_start, row_start, width, height)
            data = vrt.read(1, window=window)
            # Map nodata to 0 and map CLC indices
            if vrt.nodata is not None:
                data[data == vrt.nodata] = 0
            mask_idx = (data >= 1) & (data <= 44)
            if mask_idx.any():
                data[mask_idx] = mapping[data[mask_idx]]
            # Build water mask
            water_mask = np.isin(data, list(WATER_CODES)) | np.isin(data, list(raw_water_idx))
            if water_mask.any():
                rows, cols = np.where(water_mask)
                global_rows = row_start + rows
                global_cols = col_start + cols
                xs, ys = rasterio.transform.xy(transform, global_rows, global_cols, offset='center')
                xs = np.asarray(xs)
                ys = np.asarray(ys)
                dists = np.hypot(xs - qx, ys - qy)
                # Filter points within the search radius
                circle_idx = np.where(dists <= radius)[0]
                if circle_idx.size == 0:
                    radius += step_radius_m
                    continue
                # Select the nearest point within the circle
                sel = circle_idx[np.argmin(dists[circle_idx])]
                code = int(data[rows[sel], cols[sel]])
                if code in raw_water_idx:
                    code = mapping[code] if code <= 44 else 512
                # Transform back to lon/lat
                lon_w, lat_w = transformer_back.transform(xs[sel], ys[sel])
                return {
                    "water_code": code,
                    "class_name": CLC_CLASSES.get(code, "Unknown class"),
                    "water_lat": float(lat_w),
                    "water_lon": float(lon_w),
                    "distance_m": float(dists[sel])
                }
            radius += step_radius_m
    return None

def find_state(lon, lat, geojson_path):
    """
    Determine which German state the specified coordinates fall within.
    
    Args:
        lon: Longitude of the query point
        lat: Latitude of the query point
        geojson_path: Path to the GeoJSON file containing state boundaries
        
    Returns:
        Name of the state containing the coordinates, or "NO STATE FOUND" if
        the coordinates don't fall within any state boundary
    """
    gdf = gpd.read_file(geojson_path)

    punkt = Point(lon, lat)

    for _, row in gdf.iterrows():
        if row["geometry"].contains(punkt):
            return row["name"]  
    return "NO STATE FOUND"

def diagnostics(lon: float, lat: float, n_stations: int = 5) -> dict:
    """
    Gather comprehensive diagnostic information for a location.
    
    Combines all metadata extraction functions to provide a complete
    analysis of a fire location including terrain, nearby resources,
    weather conditions, and fire risk.
    
    Args:
        lon: Longitude of the fire location
        lat: Latitude of the fire location
        n_stations: Number of nearby fire stations to include
        
    Returns:
        Dictionary containing all diagnostic information including state,
        land cover, nearby fire stations, weather conditions, fire risk,
        helicopter resources, and nearby water bodies
    """
    firestation_path = "Data/firestation.csv"
    geodata_path     = "Data/U2018_CLC2018_V2020_20u1.tif"
    states_path      = "Data/states.geo.json"
    heli_path        = "Data/helicopter.csv"

    state              = find_state(lon, lat, states_path)
    clc_code, clc_name = get_clc_metadata(geodata_path, lon, lat)
    stations           = get_nearest_firestations(firestation_path, lon, lat, n=n_stations)
    weather_now        = get_dwd_current_weather(lat, lon)
    fire_risk          = compute_fire_risk(lat, lon)
    heli_df            = get_nearest_helicopter(heli_path, lon, lat, n=1)
    water              = get_nearest_waterbody(geodata_path, lon, lat)

    return {
        "state": state,
        "clc": {"code": clc_code, "name": clc_name},
        "stations": stations,         # GeoDataFrame
        "weather_now": weather_now,   # dict | None
        "fire_risk": fire_risk,
        "heli": heli_df,              # GeoDataFrame
        "water": water,               # dict | None
    }

def print_diagnostics(lon: float, lat: float, n_stations: int = 5) -> None:
    """
    Print diagnostic information for a location to the console.
    
    A CLI-friendly version that outputs formatted diagnostic information.
    
    Args:
        lon: Longitude of the fire location
        lat: Latitude of the fire location
        n_stations: Number of nearby fire stations to include
    """
    d = diagnostics(lon, lat, n_stations)

    print("=============== STATES ==============================")
    print("State:", d["state"])

    print("=============== GEODATA =============================")
    clc = d["clc"]
    print(f"Class Code: {clc['code']}, Class Name: {clc['name']}")

    print("=============== FIRESTATIONS ========================")
    for _, row in d["stations"].iterrows():
        lon_s, lat_s = row.geometry.x, row.geometry.y
        print(f"{row['name']}: {row['distance_m']:.0f} m  ({lat_s:.6f}, {lon_s:.6f})")

    print("=============== CURRENT WEATHER (DWD) ===============")
    wx = d["weather_now"]
    if wx:
        print(f"Temp: {wx['temperature_C']} °C")
        print(f"RH:   {wx['rel_humidity_%']} %")
        print(f"Prec: {wx['precip_mm_10min']} mm/10 min")
    else:
        print("no DATA available.")

    print("=============== FIRE RISK ESTIMATE ==================")
    print("Estimated Fire Risk:", d["fire_risk"])

    print("=============== NEAREST HELICOPTER ==================")
    for _, row in d["heli"].iterrows():
        lon_h, lat_h = row.geometry.x, row.geometry.y
        print(f"{row['Typ']} ({row['Betreiber']}) at {row['Standort']}: "
              f"{row['distance_m']:.0f} m  ({lat_h:.6f}, {lon_h:.6f})")

    print("=============== NEAREST WATER BODY ==================")
    if d["water"]:
        w = d["water"]
        print(f"Water Code: {w['water_code']}  {w['class_name']}")
        print(f"Coordinates: {w['water_lat']:.6f}, {w['water_lon']:.6f}")
        print(f"Distance:    {w['distance_m']:.0f} m")
    else:
        print("No water body found within the search radius.")


def main() -> None:
    """
    Entry point for running the script directly.
    
    Demonstrates the functionality of the module by analyzing a
    sample location in Germany and printing the results.
    """
    firestation_path = "Data/firestation.csv"
    geodata_path = "Data/U2018_CLC2018_V2020_20u1.tif"
    states_path = "Data/states.geo.json"

    longitude =   8.752790
    latitude = 50.957139
    print("=============== STATES ==============================")
    state = find_state(longitude, latitude, states_path)
    print(f"State: {state}")
    print("=============== GEODATA =============================")
    clc_code, class_name = get_clc_metadata(geodata_path, longitude, latitude)
    print(f"Class Code: {clc_code}, Class Name: {class_name}")
    print("=============== FIRESTATIONS ========================")
    nearest5 = get_nearest_firestations(firestation_path, longitude, latitude, n=5)
    for idx, row in nearest5.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        print(f"{row['name']}: {row['distance_m']:.0f} m, Koordinaten: {lat:.6f}, {lon:.6f}")

    print("=============== CURRENT WEATHER (DWD) ============")
    wx = get_dwd_current_weather(latitude, longitude)
    if wx:
        print(f"Temp: {wx['temperature_C']} °C")
        print(f"RH:   {wx['rel_humidity_%']} %")
        print(f"Prec.:{wx['precip_mm_10min']} mm (last 10 min)")
    else:
        print("no DATA available.")

    print("=============== FIRE RISK ESTIMATE =======================")
    fire_risk = compute_fire_risk(latitude, longitude)
    print(f"Estimated Fire Risk: {fire_risk}")

    print("=============== NEAREST HELICOPTER =======================")
    heli_df = get_nearest_helicopter("DATA/helicopter.csv", longitude, latitude, n=1)
    for _, row in heli_df.iterrows():
        lon_h, lat_h = row.geometry.x, row.geometry.y
        print(
            f"{row['Typ']} ({row['Betreiber']}) at {row['Standort']}: "
            f"{row['distance_m']:.0f} m, Koordinaten: {lat_h:.6f}, {lon_h:.6f}"
        )
    print("=============== NEAREST WATER BODY =======================")
    water = get_nearest_waterbody(geodata_path, longitude, latitude)
    if water:
        print(f"Water Code: {water['water_code']}, Class Name: {water['class_name']}")
        print(f"Coordinates: {water['water_lat']:.6f}, {water['water_lon']:.6f}")
        print(f"Distance: {water['distance_m']:.0f} m")
    else:
        print("No water body found within the search radius.")

if __name__ == "__main__":
    main()
