from strategy.zones import ZoneType


def calculate_equilibrium(swing_high: float, swing_low: float) -> float:
    """
    Calculates the 50% equilibrium midpoint of an active dealing range.
    Formula: (Swing High + Swing Low) / 2
    """
    return (swing_high + swing_low) / 2


def filter_zone_by_premium_discount(zone, swing_high: float, swing_low: float) -> bool:
    """
    Evaluates whether a zone strictly respects the Premium/Discount filtering rule:
    - DEMAND zones must reside entirely in the DISCOUNT half (zone.price_top <= equilibrium).
    - SUPPLY zones must reside entirely in the PREMIUM half (zone.price_bottom >= equilibrium).
    
    Returns True if valid (accepted), False if invalid (rejected).
    """
    eq = calculate_equilibrium(swing_high, swing_low)

    if zone.zone_type == ZoneType.DEMAND:
        # Strict rule: The entire demand zone must be at or below equilibrium
        return zone.price_top <= eq
    
    elif zone.zone_type == ZoneType.SUPPLY:
        # Strict rule: The entire supply zone must be at or above equilibrium
        return zone.price_bottom >= eq

    return False


def filter_zones(zones, swing_high: float, swing_low: float):
    """
    Filters a list of zones, returning only those that strictly pass the Premium/Discount rule.
    """
    return [zone for zone in zones if filter_zone_by_premium_discount(zone, swing_high, swing_low)]