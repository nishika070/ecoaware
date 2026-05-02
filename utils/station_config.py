STATION_COORDINATES = {
    "Anand Vihar": {"lat": 28.6469, "lng": 77.3161},
    "Ashok Vihar": {"lat": 28.6954, "lng": 77.1817},
    "Bawana": {"lat": 28.7996, "lng": 77.0317},
    "CRRI Mathura Road": {"lat": 28.5512, "lng": 77.2736},
    "DTU": {"lat": 28.7498, "lng": 77.1177},
    "IGI Airport T3": {"lat": 28.5562, "lng": 77.0999},
    "ITO": {"lat": 28.6289, "lng": 77.2411},
    "Jahangirpuri": {"lat": 28.7328, "lng": 77.1706},
    "Jawaharlal Nehru Stadium": {"lat": 28.5829, "lng": 77.2337},
    "Lodhi Road": {"lat": 28.5918, "lng": 77.2279},
    "Narela": {"lat": 28.8527, "lng": 77.0924},
    "North Campus DU": {"lat": 28.6869, "lng": 77.2095},
    "NSIT Dwarka": {"lat": 28.6091, "lng": 77.0381},
    "Okhla Phase-2": {"lat": 28.5313, "lng": 77.2722},
    "Patparganj": {"lat": 28.6237, "lng": 77.2872},
    "Punjabi Bagh": {"lat": 28.6683, "lng": 77.1333},
    "Rohini": {"lat": 28.7495, "lng": 77.0565},
    "Sirifort": {"lat": 28.5504, "lng": 77.2156},
    "Vivek Vihar": {"lat": 28.6721, "lng": 77.3177},
    "Wazirpur": {"lat": 28.6998, "lng": 77.1654},
}

DEFAULT_STATION = "Anand Vihar"

# validation
if DEFAULT_STATION not in STATION_COORDINATES:
    raise ValueError(f"Invalid DEFAULT_STATION: {DEFAULT_STATION}")


def get_station_coords(station: str):
    coords = STATION_COORDINATES.get(station)
    if not coords:
        return STATION_COORDINATES[DEFAULT_STATION]
    return coords