"""
Helper functions for the FireGPT application.

This module provides utility functions for handling Dash components,
processing data structures, and parsing LLM responses for the 
FireGPT application.
"""
import re
from typing import List, Dict, Tuple
import numpy as np


def remove_old_circle(children: List) -> List:
    """
    Remove elements with id=='fire-radius-circle' from a children list.
    
    Args:
        children: List of Dash components or component dictionaries
        
    Returns:
        List of components with fire radius circle elements removed
    """
    cleaned = []
    for comp in children or []:
        # Dash components have .props; dicts come from pattern matching
        cid = getattr(comp, "props", {}).get("id") if hasattr(comp, "props") else comp.get("id")
        if cid != "fire-radius-circle":
            cleaned.append(comp)
    return cleaned


def gdf_to_records(gdf):
    """
    Convert a GeoDataFrame to a list of dictionaries with lat/lon coordinates.
    
    Extracts geometry objects and replaces them with explicit lat/lon fields.
    
    Args:
        gdf: GeoDataFrame with geometry column containing Point objects
        
    Returns:
        List of dictionaries with geometry replaced by lat/lon fields
    """
    records = []
    for row in gdf.itertuples(index=False):
        rec = row._asdict()
        geom = rec.pop("geometry")
        rec["lon"] = geom.x
        rec["lat"] = geom.y
        records.append(rec)
    return records
    

def get_id(comp):
    """
    Extract the ID from a Dash component or serialized representation.
    
    Works with Dash components, serialized dict representations from frontend,
    or older Leaflet objects.
    
    Args:
        comp: Component to extract ID from
        
    Returns:
        Component ID or None if ID cannot be extracted
    """
    if comp is None:
        return None

    # 1. Serialized dict form (frontend to backend)
    if isinstance(comp, dict):
        # Direct top-level ID
        if "id" in comp:
            return comp["id"]
        # ID in props (more common)
        if isinstance(comp.get("props"), dict):
            return comp["props"].get("id")
        return None

    # 2. Classic Dash component (backend to frontend)
    try:
        return comp.id
    except AttributeError:
        pass

    # 3. Fallback for older dash-leaflet versions
    return getattr(getattr(comp, "props", {}), "get", lambda _ : None)("id")


FIRE_IDS = {"fire-marker", "fire-radius-circle"}


def strip_fire_graphics(children: List) -> List:
    """
    Remove all fire markers and radius circles from a children list.
    
    Args:
        children: List of Dash components
        
    Returns:
        Filtered list with fire-related components removed
    """
    return [c for c in (children or []) if get_id(c) not in FIRE_IDS]


def serialize_history(history: List) -> List[Dict]:
    """
    Ensure chat history is JSON serializable.
    
    If history is already a list of dictionaries, returns it unchanged.
    Otherwise converts message objects to dictionaries.
    
    Args:
        history: List of message objects or dictionaries
        
    Returns:
        JSON-serializable list of message dictionaries
    """
    if not history:
        return []
    if isinstance(history[0], dict):
        return history
    return [{"role": m.role, "content": m.content} for m in history]


def deserialize_history(payload: List[Dict]) -> List[Dict]:
    """
    Process serialized history for the RAG system.
    
    Currently just passes through the payload as RAG expects List[dict].
    
    Args:
        payload: Serialized history data
        
    Returns:
        List of message dictionaries
    """
    return payload or []


# Regular expressions for parsing LLM responses
_RX_HEADER_DEPLOYED = re.compile(r"^.*Deployed Stations.*$", flags=re.I | re.M)
_RX_HEADER_NEXT     = re.compile(r"^(\s*$|[*_A-Z].*?:)", flags=re.M)   # leere Zeile oder neue Überschrift
_RX_BULLET          = re.compile(
    r"""^\s*(?:[-–*•]|\d+\.)\s*        # Aufzählungszeichen
        (?:the\s+)?                    # evtl. 'The'
        (?P<name>(?:[*_]{2}|__)?       # evtl. Markdown-Start
            [A-ZÄÖÜ].+?)               # eigentlicher Name
        (?:[*_]{2}|__)?                # evtl. Markdown-Ende
        (?:\s*\(|\s+is\s+|\s+are\s+|\s*$)  # Ende des Namens
    """,
    flags=re.X | re.I
)



_RX_HEADER_DEPLOYED = re.compile(r"^.*Deployed Stations.*$", flags=re.I | re.M)
_RX_HEADER_NEXT     = re.compile(r"^(\s*$|[*_A-Z].*?:)", flags=re.M)   # leere Zeile oder neue Überschrift
_RX_BULLET          = re.compile(
    r"""^\s*(?:[-–*•]|\d+\.)\s*        # Aufzählungszeichen
        (?:the\s+)?                    # evtl. 'The'
        (?P<name>(?:[*_]{2}|__)?       # evtl. Markdown-Start
            [A-ZÄÖÜ].+?)               # eigentlicher Name
        (?:[*_]{2}|__)?                # evtl. Markdown-Ende
        (?:\s*\(|\s+is\s+|\s+are\s+|\s*$)  # Ende des Namens
    """,
    flags=re.X | re.I
)

_RX_HEADER_DEPLOYED = re.compile(r"^.*Deployed Stations.*$", flags=re.I | re.M)
_RX_HEADER_NEXT     = re.compile(r"^(\s*$|[*_A-Z].*?:)", flags=re.M)   # leere Zeile oder neue Überschrift
_RX_BULLET          = re.compile(
    r"""^\s*(?:[-–*•]|\d+\.)\s*        # Aufzählungszeichen
        (?:the\s+)?                    # evtl. 'The'
        (?P<name>(?:[*_]{2}|__)?       # evtl. Markdown-Start
            [A-ZÄÖÜ].+?)               # eigentlicher Name
        (?:[*_]{2}|__)?                # evtl. Markdown-Ende
        (?:\s*\(|\s+is\s+|\s+are\s+|\s*$)  # Ende des Namens
    """,
    flags=re.X | re.I
)

_RX_HELI = re.compile(r"\bHelicopter\b.*?\bYes\b", flags=re.I | re.S)


def _norm(txt: str) -> str:
    """
    Normalize text by converting to lowercase and removing non-alphanumeric characters.
    
    Args:
        txt: Text to normalize
        
    Returns:
        Normalized text string
    """
    return re.sub(r"[^a-z0-9äöüß]", "", txt.lower())


def parse_rag_recommendations(
        llm_answer: str,
        station_candidates: List[Dict]
) -> Tuple[List[Dict], bool]:
    """
    Parse LLM response to identify recommended fire stations and helicopter usage.
    
    Args:
        llm_answer: Complete text returned by the LLM
        station_candidates: List of dictionaries representing available fire stations,
                           each containing at least a "name" field
    
    Returns:
        Tuple containing:
        - List of dictionaries representing selected fire stations
        - Boolean indicating whether helicopter assistance is recommended
    """
    answer_norm = _norm(llm_answer)

    # Select based on normalized names
    selected = [
        s for s in station_candidates
        if _norm(s["name"]) in answer_norm
    ]

    # Remove generic "Feuerwehr" station if too many stations are selected
    if len(selected) > 2 and any(_norm(s["name"]) == _norm("Feuerwehr") for s in selected):
        selected = [s for s in selected if _norm(s["name"]) != _norm("Feuerwehr")]

    heli_required = bool(_RX_HELI.search(llm_answer))
    return selected, heli_required


def simplify_path(pts: List, step: int = 4) -> List:
    """
    Simplify a path by taking every nth point.
    
    Reduces path complexity for faster animation.
    
    Args:
        pts: List of coordinate points (lat, lon)
        step: Step size for point selection
        
    Returns:
        Simplified list of points
    """
    return pts[::step] if step > 1 else pts


def densify_path(pts: List, seg_points: int = 25) -> List:
    """
    Create a smoother path with evenly spaced intermediate points.
    
    Interpolates between each pair of points to create a denser, smoother path.
    
    Args:
        pts: List of coordinate points (lat, lon)
        seg_points: Number of points to create between each original point pair
        
    Returns:
        Densified list of points with consistent spacing
    """
    densified = []
    for (lat1, lon1), (lat2, lon2) in zip(pts, pts[1:]):
        lats = np.linspace(lat1, lat2, seg_points, endpoint=False)
        lons = np.linspace(lon1, lon2, seg_points, endpoint=False)
        densified.extend(zip(lats, lons))
    densified.append(pts[-1])
    return densified