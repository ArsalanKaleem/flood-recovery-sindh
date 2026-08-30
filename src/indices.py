"""
Spectral indices used across the project.

All functions accept an ee.Image and return an ee.Image with the index
appended as a new band. Band names follow Sentinel-2 SR conventions
(B2, B3, B4, B8, B11, B12); for Landsat 8/9 SR use the mapper below.
"""

from __future__ import annotations

import ee

# --------------------------------------------------------------------------
# Sentinel-2 SR
# --------------------------------------------------------------------------
def add_ndvi_s2(img: ee.Image) -> ee.Image:
    """NDVI = (NIR - RED) / (NIR + RED). Sentinel-2: B8, B4."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return img.addBands(ndvi)


def add_evi_s2(img: ee.Image) -> ee.Image:
    """
    EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1).
    Sentinel-2 SR bands are scaled 0-10000; divide by 10000 for reflectance.
    """
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        {
            "NIR": img.select("B8").divide(10000),
            "RED": img.select("B4").divide(10000),
            "BLUE": img.select("B2").divide(10000),
        },
    ).rename("EVI")
    return img.addBands(evi)


def add_ndwi_s2(img: ee.Image) -> ee.Image:
    """NDWI (McFeeters, water) = (GREEN - NIR) / (GREEN + NIR). S2: B3, B8."""
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return img.addBands(ndwi)


def add_mndwi_s2(img: ee.Image) -> ee.Image:
    """MNDWI (Xu, modified NDWI, better water/urban separation). S2: B3, B11."""
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    return img.addBands(mndwi)


# --------------------------------------------------------------------------
# Landsat 8/9 SR (Collection 2)
# --------------------------------------------------------------------------
def add_ndvi_landsat(img: ee.Image) -> ee.Image:
    """Landsat 8/9 SR bands: SR_B5 (NIR), SR_B4 (RED)."""
    ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    return img.addBands(ndvi)


def add_evi_landsat(img: ee.Image) -> ee.Image:
    """Landsat 8/9 SR EVI. Apply the Collection 2 scale factor first."""
    scaled = img.select(["SR_B2", "SR_B4", "SR_B5"]).multiply(0.0000275).add(-0.2)
    evi = scaled.expression(
        "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        {"NIR": scaled.select("SR_B5"), "RED": scaled.select("SR_B4"), "BLUE": scaled.select("SR_B2")},
    ).rename("EVI")
    return img.addBands(evi)


# --------------------------------------------------------------------------
# Sentinel-1 SAR — flood/water mapping
# --------------------------------------------------------------------------
def s1_water_mask(img: ee.Image, vv_threshold: float = -17.0) -> ee.Image:
    """
    Simple VV backscatter threshold for open water. -17 dB is a defensible
    starting point for Sindh flat cropland; validate against JRC GSW on Day 3
    and tune per-scene if needed.
    """
    water = img.select("VV").lt(vv_threshold).rename("water")
    return img.addBands(water)
