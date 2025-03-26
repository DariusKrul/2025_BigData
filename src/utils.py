# src/utils.py

import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth using the Haversine formula.

    Args:
        lat1, lon1: Latitude and Longitude of the first point (in decimal degrees).
        lat2, lon2: Latitude and Longitude of the second point (in decimal degrees).

    Returns:
        Distance in nautical miles.
    """
    R = 3440.065  # Earth radius in nautical miles

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def knots_to_kmph(knots):
    return knots * 1.852


def kmph_to_knots(kmph):
    return kmph / 1.852
