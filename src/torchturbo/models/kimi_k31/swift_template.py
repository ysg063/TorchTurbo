"""Kimi XTML prompts and Swift collation for text and image SFT."""

from io import BytesIO
from typing import Any, Literal

from PIL import Image
from torch import nn
from swift.model.model_meta import MODEL_MAPPING
from swift.template.base import Template
from swift.template.register import TemplateMeta, register_template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.templates.moonshot import KimiK25Template
from swift.template.vision_utils import load_file


class TurboKimiK31Template(KimiK25Template):
    """Reuse Kimi image collation with K3.1 chat and image prompts."""

    placeholder_tokens = ["<|media_pad|>"]
    load_images = True

    @staticmethod
    def _load_image(image: Any, load_images: bool) -> Image.Image:
        # Keep transparency for the checkpoint processor's background fill.
        if isinstance(image, dict):
            image = image.get("bytes") or image["path"]
        loaded = load_file(image)
        if isinstance(loaded, BytesIO):
            with Image.open(loaded) as decoded:
                return decoded.copy()
        return loaded

    def _get_system(self, inputs: StdTemplateInputs) -> str | None:
        return Template._get_system(self, inputs)

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        for message in inputs.messages:
            content = message["content"]
            if message["role"] != "assistant" or not isinstance(content, str):
                continue
            if content.startswith("<|open|>think<|sep|>"):
                continue
            thought = ""
            response = content
            if content.startswith("<think>") and "</think>" in content:
                thought, _, response = content.removeprefix("<think>").partition("</think>")
            message["content"] = (
                f"<|open|>think<|sep|>{thought}<|close|>think<|sep|>"
                f"<|open|>response<|sep|>{response}<|close|>response<|sep|>"
            )

    def replace_tag(
        self, media_type: Literal["image", "video", "audio"], index: int, inputs: StdTemplateInputs,
    ) -> list[str]:
        if media_type != "image":
            raise ValueError(f"Kimi K3.1 training expects images, got {media_type}")
        width, height = inputs.images[index].size
        return [self.processor.image_processor.make_image_prompt(width, height)]

    def _post_encode(self, model: nn.Module, inputs: dict[str, Any]) -> dict[str, Any]:
        # TorchTurbo fuses images before the language model's SP boundary.
        return inputs


def register() -> None:
    """Register the template using the checkpoint's tokenizer and processor."""
    template_name = "turbo_kimi_k31"
    end_message = "<|close|>message<|sep|><|end_of_msg|>"
    register_template(
        TemplateMeta(
            template_name,
            prefix=[],
            prompt=[
                '<|open|>message role="user"<|sep|>{{QUERY}}' + end_message
                + '<|open|>message role="assistant"<|sep|>'
            ],
            chat_sep=[end_message],
            suffix=[end_message],
            system_prefix=['<|open|>message role="system"<|sep|>{{SYSTEM}}' + end_message],
            template_cls=TurboKimiK31Template,
        ),
        exist_ok=True,
    )
    model = MODEL_MAPPING[template_name]
    model.template = template_name
    model.candidate_templates = [template_name]
