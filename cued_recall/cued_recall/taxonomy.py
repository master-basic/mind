"""Fixed, versioned tag vocabulary for block tagging.

Tags must come from this list, not free text: if one CRMM instance tags a
block "networking" and another tags the same kind of thing "network-stuff",
tag-based filtering silently breaks the moment blocks move between instances.
Bump TAXONOMY_VERSION when the list changes; importers on an older version
should drop tags they don't recognize rather than choke on them.

Grouped by domain purely for readability (the tagger's prompt and the admin
dropdown both use TAXONOMY_GROUPS); validation always checks against the flat
TAXONOMY list, group membership has no functional effect.
"""

TAXONOMY_VERSION = "2"

TAXONOMY_GROUPS = {
    "infrastructure": [
        "networking", "security", "firewall", "dns", "vpn", "routing",
        "windows", "linux", "docker", "virtualization", "cloud",
        "devops", "sysadmin",
    ],
    "storage": ["database", "storage", "filesystem", "backup"],
    "languages": [
        "python", "javascript", "typescript", "rust", "go", "c", "cpp",
        "csharp", "java", "php", "sql", "bash", "powershell", "html-css",
    ],
    "software-engineering": [
        "llm-tooling", "llm-config", "prompt-engineering", "embeddings",
        "api-design", "web-dev", "mobile-dev", "debugging", "performance",
        "testing", "algorithms", "automation", "scripting",
    ],
    "science": [
        "biology", "chemistry", "physics", "mathematics", "genetics",
        "neuroscience", "medicine", "astronomy", "geology", "ecology",
    ],
    "research": [
        "research-methods", "statistics", "data-science",
        "machine-learning", "ai-research",
    ],
    "general-knowledge": [
        "web-research", "history", "geography", "current-events",
        "economics", "finance", "law", "politics",
    ],
    "professional": [
        "career", "education", "writing", "marketing", "hr",
        "project-management", "business",
    ],
}

TAXONOMY = [tag for tags in TAXONOMY_GROUPS.values() for tag in tags]

_TAXONOMY_SET = set(TAXONOMY)
assert len(TAXONOMY) == len(_TAXONOMY_SET), "duplicate tag across groups"


def validate_tags(tags, max_tags: int = 3) -> list:
    """Filter arbitrary model output down to known tags, deduped, capped."""
    out = []
    seen = set()
    for t in tags or []:
        t = str(t).strip().lower()
        if t in _TAXONOMY_SET and t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= max_tags:
            break
    return out


def validate_gist(gist, max_chars: int = 40) -> str:
    g = (gist or "").strip()
    if len(g) > max_chars:
        g = g[: max_chars - 1].rstrip() + "…"
    return g
