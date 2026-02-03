"""
Scheme Signature Templates
Pre-defined behavioral signatures for known ML schemes from FIU operational alerts.
Each signature defines expected Tier 1 pillar elevations for a scheme type.
"""

from src.utils.config import cfg


def get_scheme_signatures() -> dict:
    """
    Return scheme signature vectors as {scheme_name: {pillar: expected_score}}.

    Signatures are derived from FINTRAC and FinCEN operational alerts and represent
    the expected pillar profile when a client is involved in a given scheme.
    Values are on [0, 1] scale representing expected relative elevation.
    """
    templates = cfg.p6.scheme_templates
    signatures = {}
    for name, template in templates.items():
        sig = template["signature"]
        signatures[name] = {
            "S_P1": sig.get("P1", 0.5),
            "S_P2": sig.get("P2", 0.5),
            "S_P3": sig.get("P3", 0.5),
            "S_P4": sig.get("P4", 0.5),
            "S_P5": sig.get("P5", 0.5),
        }
    return signatures


def get_scheme_descriptions() -> dict:
    """Return human-readable descriptions for each scheme template."""
    templates = cfg.p6.scheme_templates
    return {name: template["description"] for name, template in templates.items()}


def get_scheme_weights() -> dict:
    """Return pillar importance weights per scheme (for weighted similarity)."""
    templates = cfg.p6.scheme_templates
    return {name: template["weight"] for name, template in templates.items()}
