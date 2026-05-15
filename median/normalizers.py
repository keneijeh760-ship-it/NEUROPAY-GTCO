from median.product_catalog import PRODUCT_CATALOG


def normalize_product(product: str) -> str:
    if not product:
        return ""

    p = product.strip().lower()

    for canonical, info in PRODUCT_CATALOG.items():
        aliases = [a.lower() for a in info.get("aliases", [])]

        if p == canonical.lower() or p in aliases:
            return canonical

        for alias in aliases:
            if alias in p or p in alias:
                return canonical

    return p


def get_default_unit(product: str) -> str:
    product = normalize_product(product)
    info = PRODUCT_CATALOG.get(product)

    if not info:
        return "unit"

    return info.get("default_unit", "unit")


def normalize_unit(unit: str, product: str = "") -> str:
    if not unit:
        return get_default_unit(product)

    u = unit.strip().lower()

    unit_aliases = {
        "kilo": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
        "kg": "kg",

        "liter": "litre",
        "liters": "litre",
        "litre": "litre",
        "litres": "litre",

        "paint": "paint bucket",
        "paint rubber": "paint bucket",
        "rubber": "paint bucket",
        "bucket": "paint bucket",

        "derica": "cup",
        "cup": "cup",

        "sack": "bag",
        "bag": "bag",

        "bunch": "bunch",
        "bundle": "bundle",

        "crate": "crate",
        "piece": "piece",
        "pieces": "piece",
        "tuber": "tuber",
        "tubers": "tuber",

        "mudu": "mudu",
        "congo": "mudu",
    }

    return unit_aliases.get(u, u)