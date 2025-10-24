# utils/routing.py
"""
Helpers for routing with OSRM.
Uses the OSRM API to calculate routes and distances.
"""

import os
import requests
import polyline

OSRM_URL = os.getenv("OSRM_URL", "http://localhost:5001")  

def is_osrm_available():
    try:
        health_url = f"{OSRM_URL}/health"
        resp = requests.get(health_url, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

COLORS = ["red", "blue", "green", "orange", "purple", "yellow"]

def osrm_route(lat1: float, lng1: float, lat2: float, lng2: float):
    """
    Liefert:
        pts        – Liste [(lat, lon), …] → für dl.Polyline
        duration_s – Sekunden
        distance_m – Meter
    """
    coords = f"{lng1},{lat1};{lng2},{lat2}"
    url = f"{OSRM_URL}/route/v1/driving/{coords}?overview=full&geometries=polyline"

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()                       

    data = resp.json()["routes"][0]
    pts = polyline.decode(data["geometry"])    

    return [(lat, lon) for lat, lon in pts], data["duration"], data["distance"]
