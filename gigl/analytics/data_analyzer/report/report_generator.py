"""Generates a single self-contained HTML report from analysis results.

Loads the AI-owned template (report.ai.html), styles (styles.ai.css),
and chart logic (charts.ai.js), then injects serialized analysis data.

The template, styles, and chart logic are defined by SPEC.md in this
directory. AI-owned files (*.ai.html, *.ai.js, *.ai.css) can be
regenerated from the SPEC.
"""
import json
from importlib import resources
from typing import Optional

from gigl.analytics.data_analyzer.types import FeatureProfileResult, GraphAnalysisResult
from gigl.common.logger import Logger

logger = Logger()


def generate_report(
    analysis_result: Optional[GraphAnalysisResult] = None,
    profile_result: Optional[FeatureProfileResult] = None,
) -> str:
    """Generate a self-contained HTML report from analysis results.

    Args:
        analysis_result: Graph structure analysis results.
        profile_result: TFDV feature profiling results (optional).
        config: Analyzer config for metadata display (optional).

    Returns:
        Complete HTML string that opens standalone in any browser.

    Example:
        >>> html = generate_report(
        ...     analysis_result=result,
        ...     profile_result=None,
        ...     config=None,
        ... )
        >>> # Write to GCS or local file
    """
    template_dir = resources.files("gigl.analytics.data_analyzer.report")
    html_template = template_dir.joinpath("report.ai.html").read_text()
    css = template_dir.joinpath("styles.ai.css").read_text()
    js = template_dir.joinpath("charts.ai.js").read_text()

    analysis_json = json.dumps(
        analysis_result.model_dump(mode="json") if analysis_result else {}
    )
    profile_json = json.dumps(
        profile_result.model_dump(mode="json") if profile_result else {}
    )

    html = html_template
    html = html.replace("/* INJECT_STYLES */", css)
    html = html.replace("/* INJECT_SCRIPTS */", js)
    html = html.replace("/* INJECT_ANALYSIS_DATA */", analysis_json)
    html = html.replace("/* INJECT_PROFILE_DATA */", profile_json)

    logger.info(f"Generated HTML report ({len(html)} bytes)")
    return html
