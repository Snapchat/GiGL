"""TFDV feature profiling via Beam/Dataflow.

Builds standalone Beam pipelines that read from BQ tables, run
tfdv.GenerateStatistics(), and produce FACETS HTML visualizations.

Will reuse existing PTransforms:
- GenerateAndVisualizeStats (gigl/src/data_preprocessor/lib/transform/utils.py:120)
- IngestRawFeatures (gigl/src/data_preprocessor/lib/transform/utils.py:85)
- init_beam_pipeline_options (gigl/src/common/utils/dataflow.py)

NOTE: Currently a stub. Full implementation is deferred to a future PR
once the wrapping Dataflow infrastructure is ready. The stub logs a
warning and returns an empty FeatureProfileResult so callers can wire
up their code without blocking on Dataflow.
"""
from typing import Optional

from gigl.analytics.data_analyzer.config import DataAnalyzerConfig
from gigl.analytics.data_analyzer.types import FeatureProfileResult
from gigl.common import Uri
from gigl.common.logger import Logger

logger = Logger()


class FeatureProfiler:
    """Runs TFDV feature profiling on BQ tables via Dataflow.

    Currently a stub. See module docstring.

    Example:
        >>> profiler = FeatureProfiler()
        >>> result = profiler.profile(config)
        >>> # result.facets_html_paths will be empty until full impl lands
    """

    def profile(
        self,
        config: DataAnalyzerConfig,
        resource_config_uri: Optional[Uri] = None,
    ) -> FeatureProfileResult:
        """Run TFDV profiling on all tables in config.

        Args:
            config: Analyzer configuration with table specs.
            resource_config_uri: Optional resource config for Dataflow sizing.

        Returns:
            FeatureProfileResult with GCS paths to TFDV artifacts.
        """
        logger.warning(
            "FeatureProfiler not yet implemented. "
            "Returning empty results. "
            "Full implementation will wire up Beam/Dataflow pipelines "
            "using GenerateAndVisualizeStats and IngestRawFeatures."
        )
        return FeatureProfileResult()
