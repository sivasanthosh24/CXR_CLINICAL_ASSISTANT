from pathlib import Path
import torch
from .model_loader import load_module2
from .preprocessing import prepare_image_batch
from .generator import Module2Generator
class Module2Pipeline:
    def __init__(
        self,
        device=None
    ):
        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )
        self.components = load_module2(
            device=device
        )
        self.visual_conditioner = (
            self.components[
                "visual_conditioner"
            ]
        )
        self.decoder = (
            self.components[
                "decoder"
            ]
        )
        self.tokenizer = (
            self.components[
                "tokenizer"
            ]
        )
        self.preprocess = (
            self.components[
                "preprocess"
            ]
        )
        self.device = (
            device
            if device is not None
            else self.components[
                "device"
            ]
        )
        self.checkpoint_path = (
            self.components[
                "checkpoint_path"
            ]
        )
        self.generator = (
            Module2Generator(
                visual_conditioner=(
                    self.visual_conditioner
                ),
                decoder=self.decoder,
                tokenizer=self.tokenizer,
                device=self.device
            )
        )
        self._ready = True
    @property
    def is_ready(self):
        return self._ready
    @torch.inference_mode()
    def generate_report(
        self,
        image
    ):
        pixel_values = (
            prepare_image_batch(
                image=image,
                preprocess=self.preprocess,
                device=self.device
            )
        )
        report, _ = (
            self.generator.generate_single(
                pixel_values
            )
        )
        return report
    @torch.inference_mode()
    def generate_report_with_ids(
        self,
        image
    ):
        pixel_values = (
            prepare_image_batch(
                image=image,
                preprocess=self.preprocess,
                device=self.device
            )
        )
        report, generated_ids = (
            self.generator.generate_single(
                pixel_values
            )
        )
        return {
            "report": report,
            "generated_token_ids": (
                generated_ids
                .detach()
                .cpu()
                .tolist()
            )
        }
    @torch.inference_mode()
    def generate_single(
        self,
        image
    ):
        return self.generate_report(
            image
        )
    @torch.inference_mode()
    def generate_reports(
        self,
        images
    ):
        pixel_values = torch.cat(
            [
                prepare_image_batch(
                    image=image,
                    preprocess=self.preprocess,
                    device=self.device
                )
                for image in images
            ],
            dim=0
        )

        reports, generated_ids = (
            self.generator.generate(
                pixel_values
            )
        )

        return reports

    def get_model_info(
        self
    ):

        visual_total = sum(
            parameter.numel()
            for parameter
            in self.visual_conditioner.parameters()
        )

        visual_trainable = sum(
            parameter.numel()
            for parameter
            in self.visual_conditioner.parameters()
            if parameter.requires_grad
        )

        decoder_total = sum(
            parameter.numel()
            for parameter
            in self.decoder.parameters()
        )

        decoder_trainable = sum(
            parameter.numel()
            for parameter
            in self.decoder.parameters()
            if parameter.requires_grad
        )

        checkpoint = str(
            self.checkpoint_path
        )

        return {
            "module": "Module 2 V2",

            "checkpoint": checkpoint,

            "device": str(
                self.device
            ),

            "visual_total_parameters": (
                visual_total
            ),

            "visual_trainable_parameters": (
                visual_trainable
            ),

            "decoder_total_parameters": (
                decoder_total
            ),

            "decoder_trainable_parameters": (
                decoder_trainable
            ),

            "total_trainable_parameters": (
                visual_trainable
                + decoder_trainable
            ),

            "generation": {
                "num_beams": 4,
                "max_new_tokens": 160,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.0,
                "early_stopping": True,
                "do_sample": False
            }
        }


def load_pipeline(
    device=None
):

    return Module2Pipeline(
        device=device
    )


__all__ = [
    "Module2Pipeline",
    "load_pipeline"
]