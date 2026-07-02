from gigl.common.utils.feature_quantization.tensorflow_ops import quantize_tft_tensor
from gigl.common.utils.feature_quantization.torch_ops import dequantize_torch_tensor

__all__ = ["dequantize_torch_tensor", "quantize_tft_tensor"]
