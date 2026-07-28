from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

from llama_index.core.agent.workflow import (
    ReActAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.llms.openai_like import OpenAILike


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------


st.set_page_config(
    page_title="GeoPlan Agents",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_setting(
    name: str,
    default: str | None = None,
) -> str | None:
    """
    Read a project secret, then fall back to an environment variable.
    """
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        value = None

    if value is None:
        value = os.getenv(name, default)

    return str(value) if value is not None else None


OPENROUTER_API_KEY = get_setting(
    "OPENROUTER_API_KEY"
)
OPENROUTER_MODEL = (
    get_setting(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )
    or "openrouter/free"
).strip()

OPENROUTER_FALLBACK_MODELS = [
    model.strip()
    for model in (
        get_setting(
            "OPENROUTER_FALLBACK_MODELS",
            "",
        )
        or ""
    ).split(",")
    if model.strip()
]

OPENROUTER_API_BASE = (
    "https://openrouter.ai/api/v1"
)

SCENARIO_LABELS = {
    "balanced": "Balanced",
    "coverage_first": "Coverage First",
    "demand_first": "Demand First",
    "equity_balanced": "Equity Balanced",
}


def normalize_scenario_name(value: Any) -> str:
    """
    Normalize exported scenario names for matching in the interface.
    """
    normalized = str(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 18px;
            margin-bottom: 1rem;
            background:
                linear-gradient(
                    135deg,
                    rgba(49, 130, 206, 0.11),
                    rgba(16, 185, 129, 0.08)
                );
        }

        .hero h1 {
            margin: 0;
            padding: 0;
        }

        .hero p {
            margin: 0.45rem 0 0 0;
            opacity: 0.82;
        }

        .status-box {
            padding: 0.8rem 1rem;
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 14px;
            margin-bottom: 0.8rem;
        }

        .small-note {
            font-size: 0.88rem;
            opacity: 0.76;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 14px;
            padding: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Project and tool loading
# ---------------------------------------------------------------------

def locate_tool_module(start: Path) -> Path:
    candidates = [
        start,
        start / "src",
        start / "src" / "geoplan_agents",
        start.parent,
    ]

    for candidate in candidates:
        if (candidate / "geoplan_agent_tools.py").exists():
            return candidate

    raise FileNotFoundError(
        "geoplan_agent_tools.py was not found. Place this app beside "
        "the module or inside the GeoPlan project root."
    )


CURRENT_DIR = Path.cwd().resolve()
MODULE_DIR = locate_tool_module(CURRENT_DIR)

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from geoplan_agent_tools import GeoPlanToolbox  # noqa: E402


@st.cache_resource(show_spinner=False)
def load_toolbox() -> tuple[GeoPlanToolbox, dict[str, Any], Path]:
    project_root = GeoPlanToolbox.locate_project_root(
        CURRENT_DIR
    )

    toolbox = GeoPlanToolbox(
        project_root,
        strict=True,
    )

    tools = toolbox.create_llamaindex_tools()

    return toolbox, tools, project_root


toolbox, tools, project_root = load_toolbox()
output_dir = project_root / "outputs"


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number(
    value: Any,
    decimals: int = 3,
) -> str:
    numeric = safe_float(value)

    if numeric is None:
        return "—"

    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"

    return f"{numeric:.{decimals}f}"


def payload_records(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    for key in (
        "selection",
        "candidates",
        "comparison",
        "records",
        "results",
        "data",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    return []


def candidate_ids_from_toolbox() -> list[str]:
    candidates = toolbox.candidates

    if isinstance(candidates, pd.DataFrame):
        if "candidate_id" not in candidates.columns:
            raise KeyError(
                "candidate_id is missing from toolbox.candidates."
            )

        return (
            candidates["candidate_id"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

    if isinstance(candidates, list):
        ids = [
            str(item["candidate_id"])
            for item in candidates
            if isinstance(item, dict)
            and item.get("candidate_id") is not None
        ]
        return sorted(set(ids))

    raise TypeError(
        "Unsupported toolbox.candidates format."
    )


def candidate_label(candidate_id: str) -> str:
    result = toolbox.get_candidate(candidate_id)

    if result.get("status") != "ok":
        return candidate_id

    candidate = result.get("candidate", {})
    address = candidate.get("address")

    if address:
        return f"{candidate_id} — {address}"

    return candidate_id


def locate_geojson() -> Path | None:
    preferred_names = [
        "scored_ev_charging_candidates.geojson",
        "selected_ev_charging_sites.geojson",
        "selected_sites.geojson",
    ]

    for name in preferred_names:
        matches = list(project_root.rglob(name))
        if matches:
            return matches[0]

    broad_matches = list(
        project_root.rglob("*selected*site*.geojson")
    )

    if broad_matches:
        return broad_matches[0]

    return None


def flatten_coordinate_pairs(
    coordinates: Any,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []

    if (
        isinstance(coordinates, (list, tuple))
        and len(coordinates) >= 2
        and isinstance(coordinates[0], (int, float))
        and isinstance(coordinates[1], (int, float))
    ):
        pairs.append(
            (
                float(coordinates[0]),
                float(coordinates[1]),
            )
        )
        return pairs

    if isinstance(coordinates, (list, tuple)):
        for item in coordinates:
            pairs.extend(
                flatten_coordinate_pairs(item)
            )

    return pairs


def geometry_center(
    geometry: dict[str, Any] | None,
) -> tuple[float, float] | None:
    if not geometry:
        return None

    coordinate_pairs = flatten_coordinate_pairs(
        geometry.get("coordinates")
    )

    if not coordinate_pairs:
        return None

    longitude = sum(
        pair[0]
        for pair in coordinate_pairs
    ) / len(coordinate_pairs)

    latitude = sum(
        pair[1]
        for pair in coordinate_pairs
    ) / len(coordinate_pairs)

    return longitude, latitude


@st.cache_data(show_spinner=False)
def load_map_points(
    geojson_path_text: str,
) -> pd.DataFrame:
    geojson_path = Path(geojson_path_text)

    with geojson_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    rows = []

    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        center = geometry_center(
            feature.get("geometry")
        )

        if center is None:
            continue

        longitude, latitude = center

        candidate_id = (
            properties.get("candidate_id")
            or properties.get("id")
            or properties.get("site_id")
        )

        rows.append(
            {
                "candidate_id": (
                    str(candidate_id)
                    if candidate_id is not None
                    else None
                ),
                "address": properties.get("address"),
                "longitude": longitude,
                "latitude": latitude,
                "overall_score": properties.get(
                    "overall_score"
                ),
            }
        )

    return pd.DataFrame(rows)


def show_candidate_map(
    candidate_ids: list[str],
) -> None:
    geojson_path = locate_geojson()

    if geojson_path is None:
        st.info(
            "No candidate GeoJSON was found, so the map is hidden."
        )
        return

    points = load_map_points(str(geojson_path))

    if points.empty:
        st.info(
            "The GeoJSON did not contain displayable coordinates."
        )
        return

    if (
        "candidate_id" in points.columns
        and candidate_ids
    ):
        filtered = points[
            points["candidate_id"].isin(candidate_ids)
        ].copy()

        if not filtered.empty:
            points = filtered

    mean_latitude = float(points["latitude"].mean())
    mean_longitude = float(points["longitude"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[longitude, latitude]",
        get_radius=220,
        get_fill_color=[30, 136, 229, 185],
        get_line_color=[255, 255, 255, 220],
        line_width_min_pixels=1,
        stroked=True,
        pickable=True,
    )

    labels = pdk.Layer(
        "TextLayer",
        data=points,
        get_position="[longitude, latitude]",
        get_text="candidate_id",
        get_size=12,
        get_color=[20, 20, 20, 220],
        get_pixel_offset=[0, -18],
        pickable=False,
    )

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=mean_latitude,
            longitude=mean_longitude,
            zoom=10,
            pitch=0,
        ),
        layers=[layer, labels],
        tooltip={
            "html": (
                "<b>{candidate_id}</b><br/>"
                "{address}<br/>"
                "Overall score: {overall_score}"
            )
        },
    )

    st.pydeck_chart(
        deck,
        use_container_width=True,
    )


def openrouter_status() -> tuple[bool, str]:
    """
    Report whether the OpenRouter API key is configured.
    """
    if not OPENROUTER_API_KEY:
        return (
            False,
            "OPENROUTER_API_KEY is not configured.",
        )

    configured_models = [
        OPENROUTER_MODEL,
        *OPENROUTER_FALLBACK_MODELS,
    ]

    return (
        True,
        "Configured models: "
        + ", ".join(configured_models),
    )


# ---------------------------------------------------------------------
# Deterministic request routing
# ---------------------------------------------------------------------

CATEGORY_METRICS = [
    "demand_score",
    "coverage_score",
    "accessibility_score",
    "feasibility_score",
    "equity_score",
]


def extract_candidate_ids(
    prompt: str,
) -> list[str]:
    """
    Extract candidate IDs while preserving their order.
    """
    matches = re.findall(
        r"\b(?:green_p|ttc)_[A-Za-z0-9_-]+\b",
        prompt,
        flags=re.IGNORECASE,
    )

    return list(
        dict.fromkeys(
            candidate_id.lower()
            for candidate_id in matches
        )
    )


def requested_result_count(
    prompt: str,
    default: int = 5,
) -> int:
    """
    Read phrases such as 'top 10' or 'first 10'.
    """
    match = re.search(
        r"\b(?:top|first|show|return|list)\s+(\d{1,2})\b",
        prompt,
        flags=re.IGNORECASE,
    )

    if match is None:
        return default

    return max(
        1,
        min(int(match.group(1)), 20),
    )


def scenario_keys_from_prompt(
    prompt: str,
) -> list[str]:
    """
    Find supported planning scenarios without treating the word
    'balanced' inside 'equity balanced' as a second scenario.
    """
    normalized = normalize_scenario_name(prompt)
    found: list[str] = []

    specific_keys = [
        "coverage_first",
        "demand_first",
        "equity_balanced",
    ]

    remaining = normalized

    for key in specific_keys:
        if key in remaining:
            found.append(key)
            remaining = remaining.replace(key, " ")

    if re.search(
        r"(?<![a-z0-9])balanced(?![a-z0-9])",
        remaining,
    ):
        found.append("balanced")

    return found


def scenario_export_lookup() -> dict[str, str]:
    """
    Map normalized scenario names to exact exported names.
    """
    listing = toolbox.get_scenario_results()

    if listing.get("status") != "ok":
        return {}

    return {
        normalize_scenario_name(name): str(name)
        for name in listing.get(
            "available_scenarios",
            [],
        )
    }


def get_scenario_payload(
    scenario_key: str,
    n: int,
) -> tuple[dict[str, Any], str]:
    """
    Retrieve a scenario selection. Balanced can fall back to the
    primary deterministic selected sequence.
    """
    lookup = scenario_export_lookup()
    exact_name = lookup.get(scenario_key)

    if exact_name is not None:
        return (
            toolbox.get_scenario_results(
                scenario_name=exact_name,
                n=n,
            ),
            exact_name,
        )

    if scenario_key == "balanced":
        return (
            toolbox.get_selected_sequence(n=n),
            "balanced",
        )

    return (
        {
            "status": "unavailable",
            "message": (
                f"{SCENARIO_LABELS[scenario_key]} is unavailable. "
                "Generate outputs/scenario_results.csv from the "
                "planning-scenarios section of notebook 01."
            ),
            "selection": [],
        },
        scenario_key,
    )


def deterministic_candidate_comparison(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Compare two named candidates directly from deterministic data.
    """
    if "compar" not in prompt.lower():
        return None

    candidate_ids = extract_candidate_ids(prompt)

    if len(candidate_ids) < 2:
        return None

    candidate_ids = candidate_ids[:2]

    requested_metrics = [
        metric
        for metric in CATEGORY_METRICS
        if (
            metric in prompt.lower()
            or metric.replace("_score", "")
            in prompt.lower()
        )
    ]

    metrics = (
        requested_metrics
        if requested_metrics
        else CATEGORY_METRICS
    )

    payload = toolbox.compare_candidates(
        candidate_ids=candidate_ids,
        metrics=metrics,
    )
    records = payload_records(payload)

    if len(records) < 2:
        return (
            payload.get(
                "message",
                "The comparison data could not be retrieved.",
            ),
            [
                {
                    "event": "deterministic_router",
                    "tool": "compare_candidates",
                    "details": {
                        "candidate_ids": candidate_ids,
                        "status": payload.get("status"),
                    },
                }
            ],
        )

    lookup = {
        str(record.get("candidate_id")): record
        for record in records
    }

    first = lookup.get(candidate_ids[0], records[0])
    second = lookup.get(candidate_ids[1], records[1])

    first_id = str(
        first.get("candidate_id", candidate_ids[0])
    )
    second_id = str(
        second.get("candidate_id", candidate_ids[1])
    )

    rows = [
        (
            f"| Metric | {first_id} | "
            f"{second_id} | Stronger |"
        ),
        "|---|---:|---:|---|",
    ]

    wins = {
        first_id: 0,
        second_id: 0,
    }

    for metric in metrics:
        first_value = safe_float(first.get(metric))
        second_value = safe_float(second.get(metric))

        if (
            first_value is None
            or second_value is None
        ):
            stronger = "Unavailable"
        elif first_value > second_value:
            stronger = first_id
            wins[first_id] += 1
        elif second_value > first_value:
            stronger = second_id
            wins[second_id] += 1
        else:
            stronger = "Tie"

        rows.append(
            f"| {metric} | "
            f"{format_number(first_value, 4)} | "
            f"{format_number(second_value, 4)} | "
            f"{stronger} |"
        )

    if wins[first_id] > wins[second_id]:
        conclusion = (
            f"**Conclusion:** {first_id} is higher on "
            f"{wins[first_id]} of the compared metrics."
        )
    elif wins[second_id] > wins[first_id]:
        conclusion = (
            f"**Conclusion:** {second_id} is higher on "
            f"{wins[second_id]} of the compared metrics."
        )
    else:
        conclusion = (
            "**Conclusion:** The candidates split the compared "
            "metrics evenly."
        )

    addresses = []

    for record in (first, second):
        candidate_id = record.get("candidate_id")
        address = record.get("address")

        if candidate_id and address:
            addresses.append(
                f"- **{candidate_id}:** {address}"
            )

    response = "\n".join(
        [
            conclusion,
            "",
            *addresses,
            "",
            *rows,
            "",
            (
                "This comparison uses deterministic GeoPlan scores. "
                "It is planning screening, not confirmed engineering "
                "or electrical-capacity feasibility."
            ),
        ]
    )

    return (
        response,
        [
            {
                "event": "deterministic_router",
                "tool": "compare_candidates",
                "details": {
                    "candidate_ids": candidate_ids,
                    "metrics": metrics,
                },
            }
        ],
    )


def deterministic_candidate_explanation(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Explain one candidate using its stored metrics and selection status.
    """
    prompt_lower = prompt.lower()

    explanation_intent = any(
        word in prompt_lower
        for word in (
            "explain",
            "why",
            "strong",
            "weak",
            "selected",
            "strength",
            "concern",
        )
    )

    candidate_ids = extract_candidate_ids(prompt)

    if not explanation_intent or len(candidate_ids) != 1:
        return None

    candidate_id = candidate_ids[0]
    payload = toolbox.get_candidate(candidate_id)

    if payload.get("status") != "ok":
        return (
            payload.get(
                "message",
                f"{candidate_id} was not found.",
            ),
            [
                {
                    "event": "deterministic_router",
                    "tool": "get_candidate",
                    "details": {
                        "candidate_id": candidate_id,
                        "status": payload.get("status"),
                    },
                }
            ],
        )

    candidate = payload.get("candidate", {})

    available_scores = {
        metric: safe_float(candidate.get(metric))
        for metric in [
            "overall_score",
            *CATEGORY_METRICS,
        ]
        if safe_float(candidate.get(metric))
        is not None
    }

    category_values = {
        metric: value
        for metric, value in available_scores.items()
        if metric in CATEGORY_METRICS
    }

    strongest = (
        max(
            category_values,
            key=category_values.get,
        )
        if category_values
        else None
    )
    weakest = (
        min(
            category_values,
            key=category_values.get,
        )
        if category_values
        else None
    )

    selected_payload = toolbox.get_selected_sequence(
        n=20
    )
    selected_records = payload_records(
        selected_payload
    )

    selected_record = next(
        (
            record
            for record in selected_records
            if str(record.get("candidate_id"))
            == candidate_id
        ),
        None,
    )

    if selected_record is not None:
        selection_round = selected_record.get(
            "selection_round"
        )
        selection_statement = (
            f"It appears in the deterministic selected sequence"
            + (
                f" at round {selection_round}."
                if selection_round is not None
                else "."
            )
        )
    else:
        selection_statement = (
            "It does not appear in the first 20 sites of the "
            "current deterministic selected sequence."
        )

    rows = [
        "| Metric | Value |",
        "|---|---:|",
    ]

    for metric in [
        "overall_score",
        *CATEGORY_METRICS,
    ]:
        if metric in available_scores:
            rows.append(
                f"| {metric} | "
                f"{format_number(available_scores[metric], 4)} |"
            )

    interpretation = []

    if strongest is not None:
        interpretation.append(
            f"- Highest category score: **{strongest}** "
            f"({format_number(category_values[strongest], 4)})."
        )

    if weakest is not None:
        interpretation.append(
            f"- Lowest category score: **{weakest}** "
            f"({format_number(category_values[weakest], 4)})."
        )

    address = candidate.get("address")

    response = "\n".join(
        [
            f"**{candidate_id}**"
            + (
                f" — {address}"
                if address
                else ""
            ),
            "",
            selection_statement,
            "",
            *rows,
            "",
            *interpretation,
            "",
            (
                "The scores support relative planning screening. "
                "They do not confirm construction cost, electrical "
                "capacity, ownership, permitting, or engineering "
                "feasibility."
            ),
        ]
    )

    return (
        response,
        [
            {
                "event": "deterministic_router",
                "tool": "get_candidate",
                "details": {
                    "candidate_id": candidate_id,
                },
            },
            {
                "event": "deterministic_router",
                "tool": "get_selected_sequence",
                "details": {
                    "n": 20,
                },
            },
        ],
    )


def deterministic_scenario_response(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    List or compare planning-scenario selections directly from exports.
    """
    prompt_lower = prompt.lower()
    scenario_keys = scenario_keys_from_prompt(
        prompt
    )

    if not scenario_keys:
        return None

    scenario_intent = any(
        word in prompt_lower
        for word in (
            "scenario",
            "compare",
            "show",
            "list",
            "return",
            "sites",
        )
    )

    if not scenario_intent:
        return None

    n = requested_result_count(
        prompt,
        default=5,
    )

    if (
        "compar" in prompt_lower
        and len(scenario_keys) >= 2
    ):
        first_key = scenario_keys[0]
        second_key = scenario_keys[1]

        first_payload, first_exact = (
            get_scenario_payload(
                first_key,
                n=20,
            )
        )
        second_payload, second_exact = (
            get_scenario_payload(
                second_key,
                n=20,
            )
        )

        failures = []

        for payload in (
            first_payload,
            second_payload,
        ):
            if payload.get("status") != "ok":
                failures.append(
                    payload.get(
                        "message",
                        "Scenario unavailable.",
                    )
                )

        if failures:
            return (
                "\n".join(
                    [
                        "**Scenario comparison unavailable.**",
                        "",
                        *[
                            f"- {message}"
                            for message in failures
                        ],
                    ]
                ),
                [
                    {
                        "event": "deterministic_router",
                        "tool": "get_scenario_results",
                        "details": {
                            "scenarios": scenario_keys[:2],
                            "status": "unavailable",
                        },
                    }
                ],
            )

        first_records = payload_records(
            first_payload
        )
        second_records = payload_records(
            second_payload
        )

        first_ids = [
            str(record.get("candidate_id"))
            for record in first_records
            if record.get("candidate_id") is not None
        ]
        second_ids = [
            str(record.get("candidate_id"))
            for record in second_records
            if record.get("candidate_id") is not None
        ]

        first_top = first_ids[:n]
        second_top = second_ids[:n]

        shared = [
            candidate_id
            for candidate_id in first_top
            if candidate_id in second_top
        ]
        first_only = [
            candidate_id
            for candidate_id in first_top
            if candidate_id not in second_top
        ]
        second_only = [
            candidate_id
            for candidate_id in second_top
            if candidate_id not in first_top
        ]

        table = [
            (
                f"| Rank | {SCENARIO_LABELS[first_key]} | "
                f"{SCENARIO_LABELS[second_key]} |"
            ),
            "|---:|---|---|",
        ]

        for index in range(n):
            first_id = (
                first_top[index]
                if index < len(first_top)
                else "—"
            )
            second_id = (
                second_top[index]
                if index < len(second_top)
                else "—"
            )

            table.append(
                f"| {index + 1} | "
                f"{first_id} | {second_id} |"
            )

        response = "\n".join(
            [
                (
                    f"**Deterministic comparison: "
                    f"{SCENARIO_LABELS[first_key]} vs. "
                    f"{SCENARIO_LABELS[second_key]}**"
                ),
                "",
                *table,
                "",
                (
                    f"**Shared among the top {n}:** "
                    + (
                        ", ".join(shared)
                        if shared
                        else "None"
                    )
                ),
                (
                    f"**Only in {SCENARIO_LABELS[first_key]} "
                    f"top {n}:** "
                    + (
                        ", ".join(first_only)
                        if first_only
                        else "None"
                    )
                ),
                (
                    f"**Only in {SCENARIO_LABELS[second_key]} "
                    f"top {n}:** "
                    + (
                        ", ".join(second_only)
                        if second_only
                        else "None"
                    )
                ),
                "",
                (
                    "This uses the exported deterministic scenario "
                    "sequences and does not require an LLM request."
                ),
            ]
        )

        return (
            response,
            [
                {
                    "event": "deterministic_router",
                    "tool": "get_scenario_results",
                    "details": {
                        "scenarios": [
                            first_exact,
                            second_exact,
                        ],
                        "displayed_top_n": n,
                    },
                }
            ],
        )

    scenario_key = scenario_keys[0]
    payload, exact_name = get_scenario_payload(
        scenario_key,
        n=n,
    )

    if payload.get("status") != "ok":
        return (
            payload.get(
                "message",
                "The scenario is unavailable.",
            ),
            [
                {
                    "event": "deterministic_router",
                    "tool": "get_scenario_results",
                    "details": {
                        "scenario": scenario_key,
                        "status": "unavailable",
                    },
                }
            ],
        )

    records = payload_records(payload)

    table = [
        "| Rank | Candidate | Address | Score |",
        "|---:|---|---|---:|",
    ]

    for index, record in enumerate(
        records[:n],
        start=1,
    ):
        score = next(
            (
                record.get(key)
                for key in (
                    "scenario_score",
                    "selection_score",
                    "overall_score",
                )
                if record.get(key) is not None
            ),
            None,
        )

        table.append(
            f"| {record.get('selection_round', index)} | "
            f"{record.get('candidate_id', '—')} | "
            f"{record.get('address', '—')} | "
            f"{format_number(score, 4)} |"
        )

    return (
        "\n".join(
            [
                (
                    f"**{SCENARIO_LABELS[scenario_key]} "
                    f"scenario — first {len(records[:n])} sites**"
                ),
                "",
                *table,
                "",
                (
                    "These are deterministic planning-screening "
                    "recommendations."
                ),
            ]
        ),
        [
            {
                "event": "deterministic_router",
                "tool": (
                    "get_scenario_results"
                    if exact_name != "balanced"
                    else "get_selected_sequence"
                ),
                "details": {
                    "scenario": exact_name,
                    "returned": len(records[:n]),
                },
            }
        ],
    )


def deterministic_selected_sites_response(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Return the primary deterministic selected sequence.
    """
    prompt_lower = prompt.lower()

    selection_intent = (
        (
            "selected" in prompt_lower
            or "recommended" in prompt_lower
        )
        and any(
            word in prompt_lower
            for word in (
                "show",
                "return",
                "list",
                "first",
                "top",
            )
        )
        and not scenario_keys_from_prompt(prompt)
    )

    if not selection_intent:
        return None

    n = requested_result_count(
        prompt,
        default=5,
    )
    payload = toolbox.get_selected_sequence(n=n)
    records = payload_records(payload)

    table = [
        "| Rank | Candidate | Address | Score |",
        "|---:|---|---|---:|",
    ]

    for index, record in enumerate(
        records,
        start=1,
    ):
        score = next(
            (
                record.get(key)
                for key in (
                    "selection_score",
                    "overall_score",
                )
                if record.get(key) is not None
            ),
            None,
        )

        table.append(
            f"| {record.get('selection_round', index)} | "
            f"{record.get('candidate_id', '—')} | "
            f"{record.get('address', '—')} | "
            f"{format_number(score, 4)} |"
        )

    return (
        "\n".join(
            [
                (
                    f"**First {len(records)} deterministic "
                    "selected sites**"
                ),
                "",
                *table,
            ]
        ),
        [
            {
                "event": "deterministic_router",
                "tool": "get_selected_sequence",
                "details": {
                    "n": n,
                },
            }
        ],
    )


def deterministic_audit_response(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Execute candidate or recommendation-set audits directly.
    """
    if "audit" not in prompt.lower():
        return None

    candidate_ids = extract_candidate_ids(prompt)

    if not candidate_ids:
        return None

    if len(candidate_ids) == 1:
        tool_name = "audit_candidate"
        payload = toolbox.audit_candidate(
            candidate_ids[0]
        )
    else:
        tool_name = "audit_recommendation_set"
        payload = toolbox.audit_recommendation_set(
            candidate_ids
        )

    return (
        "\n".join(
            [
                f"**Deterministic audit: {tool_name}**",
                "",
                "```json",
                json.dumps(
                    payload,
                    indent=2,
                    default=str,
                ),
                "```",
            ]
        ),
        [
            {
                "event": "deterministic_router",
                "tool": tool_name,
                "details": {
                    "candidate_ids": candidate_ids,
                },
            }
        ],
    )


def deterministic_configuration_response(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Return saved planning configuration for configuration questions.
    """
    prompt_lower = prompt.lower()

    configuration_terms = (
        "planning configuration",
        "weights",
        "minimum separation",
        "exclusion distance",
        "service radius",
        "how many sites",
    )

    if not any(
        term in prompt_lower
        for term in configuration_terms
    ):
        return None

    payload = toolbox.get_planning_configuration()

    return (
        "\n".join(
            [
                "**Deterministic planning configuration**",
                "",
                "```json",
                json.dumps(
                    payload,
                    indent=2,
                    default=str,
                ),
                "```",
            ]
        ),
        [
            {
                "event": "deterministic_router",
                "tool": "get_planning_configuration",
                "details": {},
            }
        ],
    )


def route_deterministic_request(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Route common factual questions without relying on LLM tool calls.
    """
    routers = [
        deterministic_scenario_response,
        deterministic_candidate_comparison,
        deterministic_audit_response,
        deterministic_selected_sites_response,
        deterministic_configuration_response,
        deterministic_candidate_explanation,
    ]

    for router in routers:
        response = router(prompt)

        if response is not None:
            return response

    return None


# ---------------------------------------------------------------------
# Focused OpenRouter chatbot agent
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_chat_agent(
    model_name: str,
) -> ReActAgent:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    llm = OpenAILike(
        model=model_name,
        api_base=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=8192,
        max_tokens=700,
        temperature=0.0,
        timeout=120.0,
        max_retries=1,
    )

    allowed_tool_names = [
        "get_planning_configuration",
        "list_available_metrics",
        "list_candidate_ids",
        "get_candidate",
        "get_top_candidates",
        "compare_candidates",
        "get_selected_sequence",
        "get_scenario_results",
        "get_sensitivity_results",
        "audit_candidate",
        "audit_recommendation_set",
    ]

    available_tools = [
        tools[name]
        for name in allowed_tool_names
        if name in tools
    ]

    return ReActAgent(
        name="GeoPlanAssistant",
        description=(
            "Answers planning questions using deterministic GeoPlan tools."
        ),
        system_prompt=(
            "You are the GeoPlan Assistant for Toronto public "
            "EV-charging planning. Use only the exact registered "
            "tool names. Never print XML-style tool-call tags. "
            "The deterministic tools are the source of truth. "
            "Preserve candidate IDs and numeric values exactly. "
            "Do not invent missing fields, locations, scenario "
            "results, distances, or feasibility claims. State when "
            "requested evidence is unavailable. This is planning "
            "screening, not confirmed engineering feasibility."
        ),
        tools=available_tools,
        llm=llm,
        streaming=True,
    )


async def run_agent_async(
    prompt: str,
    model_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    agent = load_chat_agent(model_name)

    handler = agent.run(
        user_msg=prompt,
        max_iterations=6,
        early_stopping_method="generate",
    )

    trace: list[dict[str, Any]] = []

    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            trace.append(
                {
                    "event": "tool_call",
                    "tool": event.tool_name,
                    "details": event.tool_kwargs,
                }
            )

        elif isinstance(event, ToolCallResult):
            trace.append(
                {
                    "event": "tool_result",
                    "tool": event.tool_name,
                    "details": str(
                        event.tool_output
                    )[:1200],
                }
            )

    response = await handler
    response_text = str(response).strip()

    pseudo_tool_call = any(
        marker in response_text
        for marker in (
            "<tool_call>",
            "</tool_call>",
            "<arg_key>",
            "<arg_value>",
        )
    )

    malformed_text = (
        response_text.count("<unk>") >= 2
        or response_text.count("webkit") >= 8
        or response_text.count("urp") >= 12
    )

    if pseudo_tool_call:
        raise RuntimeError(
            "The model printed a tool-call placeholder instead of "
            "executing a structured call."
        )

    if malformed_text:
        raise RuntimeError(
            "The model returned malformed text."
        )

    if not response_text:
        raise RuntimeError(
            "The model returned an empty response."
        )

    trace.append(
        {
            "event": "model_response",
            "tool": None,
            "details": {
                "model": model_name,
            },
        }
    )

    return response_text, trace


def run_agent(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]]:
    deterministic_response = (
        route_deterministic_request(prompt)
    )

    if deterministic_response is not None:
        return deterministic_response

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. Common factual "
            "GeoPlan questions remain available through the "
            "deterministic router."
        )

    model_chain = list(
        dict.fromkeys(
            [
                OPENROUTER_MODEL,
                *OPENROUTER_FALLBACK_MODELS,
            ]
        )
    )

    attempted_models: list[str] = []

    for model_name in model_chain:
        attempted_models.append(model_name)

        try:
            response_text, trace = asyncio.run(
                run_agent_async(
                    prompt,
                    model_name,
                )
            )

            if len(attempted_models) > 1:
                trace.insert(
                    0,
                    {
                        "event": "model_fallback",
                        "tool": None,
                        "details": {
                            "attempted_models": attempted_models,
                            "successful_model": model_name,
                        },
                    },
                )

            return response_text, trace

        except Exception as error:
            message = str(error).lower()

            retryable = any(
                marker in message
                for marker in (
                    "429",
                    "rate limit",
                    "rate-limit",
                    "temporarily",
                    "503",
                    "unavailable",
                    "tool-call placeholder",
                    "malformed text",
                    "empty response",
                )
            )

            if not retryable:
                raise

    raise RuntimeError(
        "All configured OpenRouter models were unavailable or "
        "returned unusable output. Attempted: "
        + ", ".join(attempted_models)
        + ". Retry later or configure another model in "
        "OPENROUTER_FALLBACK_MODELS."
    )


# ---------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>⚡ GeoPlan Agents</h1>
        <p>
            A deterministic geospatial EV-charging site-selection engine
            with an evidence-grounded AI planning assistant.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("System status")

    openrouter_ready, openrouter_details = (
        openrouter_status()
    )

    if openrouter_ready:
        st.success("OpenRouter is configured")
        st.caption(openrouter_details)
    else:
        st.warning("OpenRouter is unavailable")
        st.caption(
            "Deterministic recommendations, comparisons, "
            "scenario questions, audits, and configuration queries "
            "still work. Configure OPENROUTER_API_KEY for other "
            "open-ended explanations."
        )

    st.metric(
        "Candidate sites",
        len(candidate_ids_from_toolbox()),
    )

    st.metric(
        "Selected sequence",
        len(toolbox.selected_sequence),
    )

    st.divider()

    st.markdown("**System boundary**")
    st.caption(
        "Python and GIS calculate scores and select sites. "
        "The LLM retrieves, compares, audits, and explains results."
    )


# ---------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------

recommend_tab, compare_tab, chat_tab, evidence_tab = st.tabs(
    [
        "📍 Recommendations",
        "⚖️ Compare sites",
        "💬 Ask GeoPlan",
        "✅ System evidence",
    ]
)


# ---------------------------------------------------------------------
# Recommendations tab
# ---------------------------------------------------------------------

with recommend_tab:
    st.subheader("Deterministic recommendations")

    maximum_sites = max(
        1,
        len(toolbox.selected_sequence),
    )

    scenario_listing = (
        toolbox.get_scenario_results()
    )

    exported_scenario_names = (
        scenario_listing.get(
            "available_scenarios",
            [],
        )
        if scenario_listing.get("status") == "ok"
        else []
    )

    exported_scenario_lookup = {
        normalize_scenario_name(name): str(name)
        for name in exported_scenario_names
    }

    scenario_keys = list(
        SCENARIO_LABELS.keys()
    )

    controls_col, summary_col = st.columns(
        [1, 2],
        gap="large",
    )

    with controls_col:
        requested_count = st.slider(
            "Number of recommended sites",
            min_value=1,
            max_value=maximum_sites,
            value=min(5, maximum_sites),
        )

        scenario_key = st.selectbox(
            "Planning scenario",
            options=scenario_keys,
            format_func=lambda key: (
                SCENARIO_LABELS[key]
                if (
                    key == "balanced"
                    or key in exported_scenario_lookup
                )
                else (
                    f"{SCENARIO_LABELS[key]} "
                    "— export required"
                )
            ),
            help=(
                "Balanced uses the deterministic selected sequence. "
                "The other scenarios require "
                "outputs/scenario_results.csv."
            ),
        )

        scenario_available = (
            scenario_key == "balanced"
            or scenario_key
            in exported_scenario_lookup
        )

        if not scenario_available:
            st.caption(
                "Run the planning-scenarios section of "
                "01_site_selection.ipynb and copy "
                "outputs/scenario_results.csv into this repository."
            )

        generate_clicked = st.button(
            "Generate recommendations",
            type="primary",
            use_container_width=True,
            disabled=not scenario_available,
        )

    request_signature = (
        scenario_key,
        requested_count,
    )

    should_generate = (
        generate_clicked
        or "recommendation_payload"
        not in st.session_state
        or st.session_state.get(
            "recommendation_signature"
        )
        != request_signature
    )

    if should_generate:
        exact_scenario_name = (
            exported_scenario_lookup.get(
                scenario_key
            )
        )

        if exact_scenario_name is not None:
            recommendation_payload = (
                toolbox.get_scenario_results(
                    scenario_name=(
                        exact_scenario_name
                    ),
                    n=requested_count,
                )
            )
        elif scenario_key == "balanced":
            recommendation_payload = (
                toolbox.get_selected_sequence(
                    n=requested_count
                )
            )
        else:
            recommendation_payload = {
                "status": "unavailable",
                "message": (
                    "This scenario has not been exported. "
                    "Generate outputs/scenario_results.csv "
                    "from the deterministic site-selection "
                    "notebook."
                ),
                "selection": [],
            }

        st.session_state.recommendation_payload = (
            recommendation_payload
        )
        st.session_state.recommendation_signature = (
            request_signature
        )

    recommendation_payload = (
        st.session_state.recommendation_payload
    )
    recommendation_records = payload_records(
        recommendation_payload
    )
    recommendation_df = pd.DataFrame(
        recommendation_records
    )

    with summary_col:
        if recommendation_payload.get("status") == "ok":
            metric_columns = st.columns(3)

            metric_columns[0].metric(
                "Returned sites",
                len(recommendation_records),
            )
            metric_columns[1].metric(
                "Scenario",
                SCENARIO_LABELS[
                    scenario_key
                ],
            )
            metric_columns[2].metric(
                "Selection method",
                "Sequential",
            )
        else:
            st.error(
                recommendation_payload.get(
                    "message",
                    (
                        "Recommendation retrieval failed. "
                        "Check the scenario export."
                    ),
                )
            )

    if not recommendation_df.empty:
        preferred_columns = [
            "scenario",
            "selection_round",
            "candidate_id",
            "address",
            "scenario_score",
            "selection_score",
            "marginal_population",
            "capacity_filled",
            "overall_score",
        ]

        visible_columns = [
            column
            for column in preferred_columns
            if column in recommendation_df.columns
        ]

        st.dataframe(
            recommendation_df[visible_columns],
            use_container_width=True,
            hide_index=True,
        )

        candidate_ids = (
            recommendation_df["candidate_id"]
            .astype(str)
            .tolist()
        )

        show_candidate_map(candidate_ids)

        st.download_button(
            "Download recommendations as CSV",
            data=recommendation_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                f"geoplan_{scenario_key}_"
                f"{requested_count}_sites.csv"
            ),
            mime="text/csv",
        )

    st.info(
        "These are planning-screening recommendations. "
        "Electrical capacity, construction cost, property ownership, "
        "permitting, and detailed engineering feasibility require "
        "additional assessment."
    )


# ---------------------------------------------------------------------
# Comparison tab
# ---------------------------------------------------------------------

with compare_tab:
    st.subheader("Compare two candidate locations")

    all_candidate_ids = candidate_ids_from_toolbox()

    first_default = (
        all_candidate_ids.index("green_p_710")
        if "green_p_710" in all_candidate_ids
        else 0
    )

    second_default = (
        all_candidate_ids.index("green_p_821")
        if "green_p_821" in all_candidate_ids
        else min(1, len(all_candidate_ids) - 1)
    )

    selection_col_a, selection_col_b = st.columns(2)

    with selection_col_a:
        candidate_a = st.selectbox(
            "Location A",
            options=all_candidate_ids,
            index=first_default,
            format_func=candidate_label,
            key="candidate_a",
        )

    with selection_col_b:
        candidate_b = st.selectbox(
            "Location B",
            options=all_candidate_ids,
            index=second_default,
            format_func=candidate_label,
            key="candidate_b",
        )

    available_metrics = [
        "overall_score",
        "demand_score",
        "coverage_score",
        "accessibility_score",
        "feasibility_score",
        "equity_score",
        "capacity_filled",
        "population_1000m",
        "traffic_volume_weighted_1000m",
        "distance_to_nearest_charger_m",
        "existing_chargers_2000m",
    ]

    selected_metrics = st.multiselect(
        "Metrics",
        options=available_metrics,
        default=[
            "overall_score",
            "demand_score",
            "coverage_score",
            "accessibility_score",
            "feasibility_score",
            "equity_score",
        ],
    )

    if candidate_a == candidate_b:
        st.warning(
            "Choose two different candidates."
        )
    else:
        compare_payload = toolbox.compare_candidates(
            candidate_ids=[
                candidate_a,
                candidate_b,
            ],
            metrics=selected_metrics,
        )

        comparison_records = payload_records(
            compare_payload
        )

        if not comparison_records:
            comparison_records = []

            for candidate_id in (
                candidate_a,
                candidate_b,
            ):
                candidate_payload = (
                    toolbox.get_candidate(
                        candidate_id
                    )
                )

                if (
                    candidate_payload.get("status")
                    == "ok"
                ):
                    comparison_records.append(
                        candidate_payload[
                            "candidate"
                        ]
                    )

        comparison_df = pd.DataFrame(
            comparison_records
        )

        if not comparison_df.empty:
            visible_columns = [
                "candidate_id",
                "address",
            ] + [
                metric
                for metric in selected_metrics
                if metric in comparison_df.columns
            ]

            visible_columns = [
                column
                for column in visible_columns
                if column in comparison_df.columns
            ]

            st.dataframe(
                comparison_df[visible_columns],
                use_container_width=True,
                hide_index=True,
            )

            score_metrics = [
                metric
                for metric in selected_metrics
                if metric.endswith("_score")
                and metric in comparison_df.columns
            ]

            if score_metrics:
                chart_df = (
                    comparison_df[
                        ["candidate_id"]
                        + score_metrics
                    ]
                    .set_index("candidate_id")
                    .T
                )

                st.bar_chart(chart_df)

            show_candidate_map(
                [candidate_a, candidate_b]
            )

        explanation_button = st.button(
            "Explain this comparison",
            type="primary",
        )

        if explanation_button:
            prompt = f"""
            Compare GeoPlan candidates {candidate_a} and {candidate_b}.

            You must call compare_candidates with these metrics:
            {selected_metrics}

            Provide:
            1. a concise conclusion;
            2. metric-by-metric evidence;
            3. the main trade-offs;
            4. which candidate is stronger overall;
            5. the planning-screening limitation.

            Preserve all tool-returned numbers exactly.
            """

            with st.spinner(
                "The agent is retrieving deterministic evidence..."
            ):
                try:
                    explanation, trace = run_agent(
                        prompt
                    )

                    st.markdown(explanation)

                    with st.expander(
                        "Show agent tool trace"
                    ):
                        st.json(trace)

                except Exception as error:
                    st.error(
                        f"Agent execution failed: {error}"
                    )


# ---------------------------------------------------------------------
# Chatbot tab
# ---------------------------------------------------------------------

with chat_tab:
    st.subheader("Ask the GeoPlan Assistant")

    st.caption(
        "Good questions include: “Why was green_p_710 selected?”, "
        "“Compare green_p_710 and green_p_821”, or "
        "“Audit these five candidate IDs.”"
    )

    example_columns = st.columns(3)

    example_prompts = [
        (
            "Explain a site",
            "Explain why green_p_710 is a strong or weak candidate.",
        ),
        (
            "Compare two sites",
            "Compare green_p_710 and green_p_821 using the main category scores.",
        ),
        (
            "Compare scenarios",
            "Compare the balanced and coverage-first planning scenarios.",
        ),
    ]

    for column, (label, prompt_text) in zip(
        example_columns,
        example_prompts,
    ):
        if column.button(
            label,
            use_container_width=True,
        ):
            st.session_state.pending_prompt = (
                prompt_text
            )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me about candidate evidence, comparisons, "
                    "planning scenarios, or recommendation audits."
                ),
                "trace": [],
            }
        ]

    for message in st.session_state.chat_messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(message["content"])

            if message.get("trace"):
                with st.expander(
                    "Tool trace"
                ):
                    st.json(message["trace"])

    typed_prompt = st.chat_input(
        "Ask a grounded planning question...",
    )

    pending_prompt = st.session_state.pop(
        "pending_prompt",
        None,
    )

    active_prompt = typed_prompt or pending_prompt

    if active_prompt:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": active_prompt,
                "trace": [],
            }
        )

        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner(
                "Checking GeoPlan evidence..."
            ):
                try:
                    response_text, trace = run_agent(
                        active_prompt
                    )

                    st.markdown(response_text)

                    if trace:
                        with st.expander(
                            "Tool trace"
                        ):
                            st.json(trace)

                except Exception as error:
                    response_text = (
                        "The agent could not complete the request. "
                        f"Error: {error}"
                    )
                    trace = []
                    st.error(response_text)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "trace": trace,
            }
        )


# ---------------------------------------------------------------------
# Evidence and evaluation tab
# ---------------------------------------------------------------------

with evidence_tab:
    st.subheader("Validation and evaluation evidence")

    grounding_path = (
        output_dir
        / "final_agent_report_grounding.json"
    )
    validation_path = (
        output_dir
        / "final_agent_report_validation.json"
    )
    evaluation_path = (
        output_dir
        / "agent_evaluation_summary.csv"
    )
    trace_path = (
        output_dir
        / "multiagent_trace.csv"
    )

    evidence_columns = st.columns(3)

    if grounding_path.exists():
        grounding = json.loads(
            grounding_path.read_text(
                encoding="utf-8"
            )
        )
        grounded = (
            grounding.get("grounded") is True
        )
        evidence_columns[0].metric(
            "Grounding validation",
            "Passed" if grounded else "Failed",
        )
    else:
        evidence_columns[0].metric(
            "Grounding validation",
            "Not found",
        )

    if validation_path.exists():
        validation = json.loads(
            validation_path.read_text(
                encoding="utf-8"
            )
        )
        saved_checks = validation.get(
            "checks",
            {},
        )
        passed_count = sum(
            value is True
            for value in saved_checks.values()
        )
        evidence_columns[1].metric(
            "Saved validation checks",
            f"{passed_count}/{len(saved_checks)}",
        )
    else:
        evidence_columns[1].metric(
            "Saved validation checks",
            "Not found",
        )

    if evaluation_path.exists():
        evaluation_df = pd.read_csv(
            evaluation_path
        )

        total_tests = int(
            evaluation_df["tests"].sum()
        )
        passed_tests = int(
            evaluation_df["passed"].sum()
        )

        evidence_columns[2].metric(
            "Evaluation tests",
            f"{passed_tests}/{total_tests}",
        )

        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        evidence_columns[2].metric(
            "Evaluation tests",
            "Not found",
        )
        st.caption(
            "Run evaluate_multiagent_system.ipynb "
            "to generate the evaluation summary."
        )

    if trace_path.exists():
        st.markdown("#### Latest multi-agent trace")

        trace_df = pd.read_csv(trace_path)

        trace_columns = [
            column
            for column in [
                "timestamp_utc",
                "agent",
                "event_type",
                "tool_name",
                "arguments",
                "output_preview",
            ]
            if column in trace_df.columns
        ]

        st.dataframe(
            trace_df[trace_columns],
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    with st.expander(
        "Architecture and responsibility boundary"
    ):
        st.markdown(
            """
            ```text
            User
              ↓
            Streamlit interface
              ↓
            GeoPlan deterministic tools
              ├── retrieve candidates
              ├── compare metrics
              ├── return selected sequence
              └── audit recommendations
              ↓
            LlamaIndex assistant
              ├── chooses tools
              ├── explains evidence
              └── reports limitations
            ```

            The assistant does not independently calculate suitability
            scores or replace engineering review.
            """
        )
