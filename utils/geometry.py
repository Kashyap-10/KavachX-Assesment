"""
Geometry and zone containment utility functions.
"""

import json
from typing import Dict, List, Tuple
from shapely.geometry import Point, Polygon


def load_zones(zones_path: str) -> Dict[str, Polygon]:
    """Loads polygon zones from a JSON file.
    
    Expected format:
    {
        "zone_name": [[x1, y1], [x2, y2], [x3, y3], ...]
    }
    """
    with open(zones_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    zones = {}
    for zone_name, coords in data.items():
        if len(coords) >= 3:
            zones[zone_name] = Polygon(coords)
        else:
            raise ValueError(f"Zone '{zone_name}' requires at least 3 vertices.")
    return zones


def is_point_in_zone(point: Tuple[float, float], zone_polygon: Polygon) -> bool:
    """Checks whether (x, y) point lies inside or on the boundary of the zone polygon."""
    p = Point(point[0], point[1])
    return zone_polygon.contains(p) or zone_polygon.touches(p)


def get_foot_position(bbox: List[float]) -> Tuple[float, float]:
    """Computes the bottom-center point (foot position) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    bottom_y = float(y2)
    return (center_x, bottom_y)
