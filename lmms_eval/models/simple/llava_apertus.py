from typing import Optional, Union

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava import Llava, best_fit_attn_implementation

try:
    from llava.mm_utils import get_model_name_from_path
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)
    get_model_name_from_path = None


def _infer_conv_template(model_name: str) -> str:
    name = model_name.lower()
    if "llama3" in name:
        return "llava_llama_3"
    if "qwen25" in name:
        return "qwen_2_5"
    if "apertus" in name:
        return "apertus_ori"
    if "llama-2" in name:
        return "llava_llama_2"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


@register_model("llava_apertus")
class LlavaApertus(Llava):
    def __init__(
        self,
        pretrained: str = "checkpoints/llava-next-apertus-8b-finetune-full-ori-fixed",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation=best_fit_attn_implementation,
        device_map: str = "cuda:0",
        conv_template: Optional[str] = "auto",
        use_cache: bool = True,
        use_stop_str: bool = True,
        tie_weights: bool = True,
        truncate_context: bool = False,
        customized_config: Optional[str] = None,
        **kwargs,
    ) -> None:
        resolved_model_name = model_name
        if resolved_model_name is None:
            if get_model_name_from_path is None:
                resolved_model_name = pretrained
            else:
                resolved_model_name = get_model_name_from_path(pretrained)

        if conv_template is None or conv_template == "auto":
            conv_template = _infer_conv_template(resolved_model_name)

        super().__init__(
            pretrained=pretrained,
            truncation=truncation,
            device=device,
            batch_size=batch_size,
            model_name=resolved_model_name,
            attn_implementation=attn_implementation,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            use_stop_str=use_stop_str,
            tie_weights=tie_weights,
            truncate_context=truncate_context,
            customized_config=customized_config,
            **kwargs,
        )
