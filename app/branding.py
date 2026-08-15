"""Company reference data and logo loading for the index roll."""
import base64
from functools import lru_cache
from pathlib import Path

LOGO_DIR = Path(__file__).parent / "assets" / "logos"

# Order mirrors a rough sector grouping (financials, materials/chemicals,
# autos, industrials, consumer, healthcare) rather than alphabetical —
# reads like an index composition sheet, not a dropdown.
COMPANIES = [
    {"key": "Allianz",       "name": "Allianz",        "ticker": "ALV",  "logo": "allianz.svg"},
    {"key": "MunichRe",      "name": "Munich Re",      "ticker": "MUV2", "logo": "munichre.svg"},
    {"key": "BASF",          "name": "BASF",           "ticker": "BAS",  "logo": "basf.svg"},
    {"key": "Bayer",         "name": "Bayer",          "ticker": "BAYN", "logo": "bayer.svg"},
    {"key": "Merck",         "name": "Merck KGaA",     "ticker": "MRK",  "logo": "merck.svg"},
    {"key": "BMW",           "name": "BMW",            "ticker": "BMW",  "logo": "bmw.svg"},
    {"key": "Mercedes",      "name": "Mercedes-Benz",  "ticker": "MBG",  "logo": "mercedes.svg"},
    {"key": "VW",            "name": "Volkswagen",     "ticker": "VOW3", "logo": "volkswagen.svg"},
    {"key": "Siemens",       "name": "Siemens",        "ticker": "SIE",  "logo": "siemens.svg"},
    {"key": "SiemensEnergy", "name": "Siemens Energy", "ticker": "ENR",  "logo": "siemensenergy.svg"},
    {"key": "Infineon",      "name": "Infineon",       "ticker": "IFX",  "logo": "infineon.svg"},
    {"key": "DHL",           "name": "DHL Group",      "ticker": "DHL",  "logo": "dhl.svg"},
    {"key": "SAP",           "name": "SAP",            "ticker": "SAP",  "logo": "sap.svg"},
    {"key": "Henkel",        "name": "Henkel",         "ticker": "HEN3", "logo": "henkel.svg"},
    {"key": "Beiersdorf",    "name": "Beiersdorf",     "ticker": "BEI",  "logo": "beiersdorf.svg"},
]

BY_KEY = {c["key"]: c for c in COMPANIES}

ASSET_DIR = Path(__file__).parent / "assets"
DAX_LOGO = LOGO_DIR / "dax-index.svg"
AUTHOR_PHOTO_CANDIDATES = [ASSET_DIR / "moritz.png", ASSET_DIR / "moritz.jpg"]
STATIC_DIR = Path(__file__).parent / "static"
REPORTS_DIR = STATIC_DIR / "reports"


@lru_cache(maxsize=None)
def logo_data_uri(filename: str) -> str:
    path = LOGO_DIR / filename
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


@lru_cache(maxsize=None)
def dax_logo_uri() -> str:
    return logo_data_uri("dax-index.svg")


@lru_cache(maxsize=None)
def author_photo_uri() -> str | None:
    for path in AUTHOR_PHOTO_CANDIDATES:
        if path.exists():
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return None


def company_meta(key: str) -> dict:
    return BY_KEY.get(key, {"key": key, "name": key, "ticker": key[:4].upper(), "logo": None})


@lru_cache(maxsize=None)
def _pdf_filename(key: str, year: str) -> str | None:
    """Report filenames follow `{Key}_Report_{year}.pdf` (see pipeline/ingest.py's
    naming convention) — checked against disk so a missing year (e.g. a
    constituent's FY2024 report not yet sourced) degrades to no link rather
    than a broken one."""
    candidate = REPORTS_DIR / year / f"{key}_Report_{year}.pdf"
    return candidate.name if candidate.exists() else None


def report_pdf_url(key: str, year: str = "2025", page: int | str | None = None) -> str | None:
    filename = _pdf_filename(key, year)
    if filename is None:
        return None
    url = f"app/static/reports/{year}/{filename}"
    if page:
        url += f"#page={int(page)}"
    return url
