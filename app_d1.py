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
OPENROUTER_MODEL = get_setting(
    "OPENROUTER_MODEL",
    "openrouter/free",
)
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

    return (
        True,
        f"Configured model: {OPENROUTER_MODEL}",
    )


# ---------------------------------------------------------------------
# Focused chatbot agent
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_chat_agent() -> ReActAgent:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    llm = OpenAILike(
        model=OPENROUTER_MODEL,
        api_base=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=8192,
        max_tokens=700,
        temperature=0.0,
        timeout=120.0,
        max_retries=2,
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
            "EV-charging planning. You MUST use at least one tool "
            "before answering factual questions. The deterministic "
            "tools are the source of truth. Preserve candidate IDs "
            "and numeric values exactly. Do not invent missing fields, "
            "locations, scenario results, distances, or feasibility "
            "claims. When asked about a planning scenario, call "
            "get_scenario_results and use the exact exported scenario "
            "name. When asked about robustness, call "
            "get_sensitivity_results. Distinguish population_1000m "
            "from marginal_population. State when a requested analysis "
            "or export is unavailable. This is planning screening, not "
            "confirmed engineering feasibility. Answer with a concise "
            "conclusion, supporting evidence, and limitations."
        ),
        tools=available_tools,
        llm=llm,
        streaming=True,
    )


async def run_agent_async(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]]:
    agent = load_chat_agent()

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

    return str(response), trace


def run_agent(
    prompt: str,
) -> tuple[str, list[dict[str, Any]]]:
    return asyncio.run(
        run_agent_async(prompt)
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
            "The deterministic recommendation and comparison "
            "features still work. Configure OPENROUTER_API_KEY "
            "to enable AI explanations."
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
            "Explain this comparison with the agent",
            disabled=not openrouter_ready,
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
        disabled=not openrouter_ready,
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
