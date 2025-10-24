# utils/icons.py
def get_icon(kind: str):
    """
    Delivers a dictionary with icon properties for the specified kind.
    """
    base = {"iconSize": [32, 32], "iconAnchor": [16, 16]}
    icon_map = {
        "fire":         {**base, "iconUrl": "/assets/fire.png"},
        "firestation":  {**base, "iconUrl": "/assets/haus.png"},
        "water":         {**base, "iconUrl": "/assets/water.png"},
        "heli": {
            "iconUrl": "/assets/heli.png",
            "iconSize":   [45, 45],  
            "iconAnchor": [22, 20],   
        },        
        "firetruck": {
            "iconUrl": "/assets/firetruck.png",
            "iconSize":   [22, 22],   
            "iconAnchor": [11, 22],  
        },
        "hangar": {
            **base, "iconUrl": "/assets/hangar.png",  
        },
    }
    return icon_map[kind]
