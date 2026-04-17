"""
BQ Data Analyzer for pre-training graph data analysis.

Produces a single HTML report covering data quality, feature distributions,
and graph structure metrics from BigQuery node/edge tables.
"""

from gigl.analytics.data_analyzer.data_analyzer import DataAnalyzer

__all__ = ["DataAnalyzer"]
