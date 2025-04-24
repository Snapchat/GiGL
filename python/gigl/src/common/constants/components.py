from enum import Enum


class GiGLComponents(Enum):
    ConfigValidator = "config_validator"
    ConfigPopulator = "config_populator"
    DataPreprocessor = "data_preprocessor"
    SubgraphSampler = "subgraph_sampler"
    SplitGenerator = "split_generator"
    Trainer = "trainer"
    Inferencer = "inferencer"
    PostProcessor = "post_processor"

    @property
    def kebab_case_value(self):
        return self.value.replace("_", "-")


GLT_BACKEND_UNSUPPORTED_COMPONENTS = [
    GiGLComponents.SubgraphSampler.value,
    GiGLComponents.SplitGenerator.value,
]
