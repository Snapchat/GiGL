from apache_beam.options.pipeline_options import PipelineOptions


class CommonOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            "--resource_config_uri",
            help="Runtime argument for resource and env specifications of each component",
        )
