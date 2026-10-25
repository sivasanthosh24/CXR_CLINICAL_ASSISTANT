from pathlib import Path
import warnings
import torch
import torch.nn as nn
import open_clip
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE2_ROOT = (
    PROJECT_ROOT
    / "models"
    / "module2"
)
BIOMEDCLIP_MODEL_NAME = (
    "hf-hub:microsoft/"
    "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
GPT2_MODEL_NAME = "gpt2"
VISUAL_HIDDEN_SIZE = 768
TRAINABLE_LAST_N = 3
PROJECTION_DROPOUT = 0.10
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10
ATTENTION_LORA_PATTERN = (
    r".*\.(attn|crossattention)\."
    r"(c_attn|c_proj|q_attn)$"
)
EXPECTED_VISUAL_TOTAL = 87_373_056
EXPECTED_VISUAL_TRAINABLE = 22_444_800
EXPECTED_VISUAL_FROZEN = 64_928_256
EXPECTED_DECODER_TOTAL = 153_765_120
EXPECTED_DECODER_TRAINABLE = 29_306_880
EXPECTED_DECODER_FROZEN = 124_458_240
EXPECTED_TOTAL_TRAINABLE = 51_751_680
EXPECTED_VISUAL_TOKENS = 197
DEFAULT_CHECKPOINT_NAME = (
    "module2_v2_best.pt"
)
class VisualFeatureProjection(
    nn.Module
):
    def __init__(
        self,
        hidden_size=768,
        dropout=0.10
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_size,
                hidden_size
            )
        )
    def forward(
        self,
        x
    ):
        return self.projection(x)
class BiomedCLIPVisualConditioner(
    nn.Module
):
    def __init__(
        self,
        visual_encoder,
        projection,
        trainable_last_n=3
    ):
        super().__init__()
        self.visual_encoder = (
            visual_encoder
        )
        self.projection = (
            projection
        )
        self.trainable_last_n = (
            trainable_last_n
        )
        self.num_layers = len(
            self.visual_encoder
            .trunk
            .blocks
        )
        self.freeze_until = (
            self.num_layers
            - self.trainable_last_n
        )
    def forward(
        self,
        pixel_values
    ):
        trunk = (
            self.visual_encoder
            .trunk
        )
        x = trunk.patch_embed(
            pixel_values
        )
        if x.ndim == 4:
            x = x.flatten(
                2
            ).transpose(
                1,
                2
            )

        if hasattr(
            trunk,
            "cls_token"
        ):

            cls_token = (
                trunk.cls_token.expand(
                    x.shape[0],
                    -1,
                    -1
                )
            )

            x = torch.cat(
                [
                    cls_token,
                    x
                ],
                dim=1
            )

        if hasattr(
            trunk,
            "pos_embed"
        ):

            x = (
                x
                + trunk.pos_embed
            )

        if hasattr(
            trunk,
            "pos_drop"
        ):

            x = trunk.pos_drop(
                x
            )

        for block in trunk.blocks:

            x = block(x)

        x = trunk.norm(x)

        x = self.projection(
            x
        )

        return x


def _count_parameters(
    module
):

    total = 0

    trainable = 0

    for parameter in (
        module.parameters()
    ):

        total += parameter.numel()

        if parameter.requires_grad:

            trainable += parameter.numel()

    frozen = (
        total
        - trainable
    )

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen
    }


def _set_visual_trainability(
    visual_encoder,
    trainable_last_n=3
):

    for parameter in (
        visual_encoder.parameters()
    ):

        parameter.requires_grad = False

    trunk = (
        visual_encoder.trunk
    )

    num_layers = len(
        trunk.blocks
    )

    freeze_until = (
        num_layers
        - trainable_last_n
    )

    for layer_idx in range(
        freeze_until,
        num_layers
    ):

        for parameter in (
            trunk
            .blocks[layer_idx]
            .parameters()
        ):

            parameter.requires_grad = True

    return freeze_until


def _build_visual_conditioner():

    print(
        "Loading BiomedCLIP architecture..."
    )

    model, preprocess_train, preprocess_val = (
        open_clip.create_model_and_transforms(
            BIOMEDCLIP_MODEL_NAME
        )
    )

    visual = model.visual

    if not hasattr(
        visual,
        "trunk"
    ):

        raise RuntimeError(
            "BiomedCLIP visual encoder does not "
            "contain the expected 'trunk' attribute."
        )

    trunk = visual.trunk

    if not hasattr(
        trunk,
        "blocks"
    ):

        raise RuntimeError(
            "BiomedCLIP ViT trunk does not "
            "contain transformer blocks."
        )

    num_blocks = len(
        trunk.blocks
    )

    if num_blocks != 12:

        raise RuntimeError(
            "Unexpected BiomedCLIP transformer "
            f"depth: {num_blocks}. Expected 12."
        )

    _set_visual_trainability(
        visual,
        TRAINABLE_LAST_N
    )

    projection = (
        VisualFeatureProjection(
            hidden_size=VISUAL_HIDDEN_SIZE,
            dropout=PROJECTION_DROPOUT
        )
    )

    visual_conditioner = (
        BiomedCLIPVisualConditioner(
            visual_encoder=visual,
            projection=projection,
            trainable_last_n=TRAINABLE_LAST_N
        )
    )

    return (
        visual_conditioner,
        preprocess_val
    )


def _build_decoder():

    print(
        "Building GPT-2 decoder with "
        "cross-attention..."
    )

    config = (
        GPT2Config.from_pretrained(
            GPT2_MODEL_NAME
        )
    )

    config.add_cross_attention = True

    if (
        config.n_embd
        != VISUAL_HIDDEN_SIZE
    ):

        raise RuntimeError(
            "GPT-2 hidden size "
            f"{config.n_embd} does not match "
            f"visual hidden size "
            f"{VISUAL_HIDDEN_SIZE}."
        )

    decoder = (
        GPT2LMHeadModel.from_pretrained(
            GPT2_MODEL_NAME,
            config=config
        )
    )

    cross_attention_count = 0

    for block in (
        decoder.transformer.h
    ):

        if hasattr(
            block,
            "crossattention"
        ):

            cross_attention_count += 1

    if (
        cross_attention_count
        != config.n_layer
    ):

        raise RuntimeError(
            "Cross-attention was not created "
            "in every GPT-2 block. "
            f"Found {cross_attention_count}; "
            f"expected {config.n_layer}."
        )

    print(
        "Applying exact attention-only LoRA..."
    )

    print(
        "LoRA target pattern:"
    )

    print(
        ATTENTION_LORA_PATTERN
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            ATTENTION_LORA_PATTERN
        )
    )

    decoder = get_peft_model(
        decoder,
        lora_config
    )

    _verify_lora_structure(
        decoder
    )

    _enable_cross_attention_base(
        decoder
    )

    return decoder


def _verify_lora_structure(
    decoder
):

    lora_modules = []

    for name, module in (
        decoder.named_modules()
    ):

        if (
            hasattr(
                module,
                "lora_A"
            )
            or
            hasattr(
                module,
                "lora_B"
            )
        ):

            lora_modules.append(
                name
            )

    mlp_lora_modules = [
        name
        for name in lora_modules
        if ".mlp." in name
    ]

    if mlp_lora_modules:

        raise RuntimeError(
            "Incorrect LoRA construction. "
            "MLP layers received LoRA:\n"
            + "\n".join(
                mlp_lora_modules
            )
        )

    self_lora_modules = [
        name
        for name in lora_modules
        if (
            ".attn." in name
            and ".crossattention."
            not in name
        )
    ]

    cross_lora_modules = [
        name
        for name in lora_modules
        if ".crossattention." in name
    ]

    cross_q_modules = [
        name
        for name in cross_lora_modules
        if ".q_attn" in name
    ]

    cross_c_attn_modules = [
        name
        for name in cross_lora_modules
        if ".c_attn" in name
    ]

    cross_c_proj_modules = [
        name
        for name in cross_lora_modules
        if ".c_proj" in name
    ]

    if len(
        self_lora_modules
    ) != 24:

        raise RuntimeError(
            "Unexpected self-attention "
            "LoRA module count: "
            f"{len(self_lora_modules)}. "
            "Expected 24."
        )

    if len(
        cross_q_modules
    ) != 12:

        raise RuntimeError(
            "Unexpected cross-attention "
            "q_attn LoRA count: "
            f"{len(cross_q_modules)}. "
            "Expected 12."
        )

    if len(
        cross_c_attn_modules
    ) != 12:

        raise RuntimeError(
            "Unexpected cross-attention "
            "c_attn LoRA count: "
            f"{len(cross_c_attn_modules)}. "
            "Expected 12."
        )

    if len(
        cross_c_proj_modules
    ) != 12:

        raise RuntimeError(
            "Unexpected cross-attention "
            "c_proj LoRA count: "
            f"{len(cross_c_proj_modules)}. "
            "Expected 12."
        )

    if len(
        cross_lora_modules
    ) != 36:

        raise RuntimeError(
            "Unexpected cross-attention "
            f"LoRA module count: "
            f"{len(cross_lora_modules)}. "
            "Expected 36."
        )

    total_lora_modules = (
        len(self_lora_modules)
        + len(cross_lora_modules)
    )

    if total_lora_modules != 60:

        raise RuntimeError(
            "Unexpected total LoRA module "
            f"count: {total_lora_modules}. "
            "Expected 60."
        )

    print(
        "LoRA structure verification: PASS"
    )

    print(
        "  Self-attention modules : 24"
    )

    print(
        "  Cross-attention modules: 36"
    )

    print(
        "  Total LoRA modules     : 60"
    )

    print(
        "  MLP LoRA modules       : 0"
    )


def _enable_cross_attention_base(
    decoder
):

    base_decoder = (
        decoder.base_model.model
    )

    for block in (
        base_decoder.transformer.h
    ):

        cross_attention = (
            block.crossattention
        )

        for parameter in (
            cross_attention.parameters()
        ):

            parameter.requires_grad = True


def _verify_cross_attention_trainability(
    decoder
):

    base_decoder = (
        decoder.base_model.model
    )

    base_count = 0

    base_tensors = 0

    lora_count = 0

    lora_tensors = 0

    for name, parameter in (
        decoder.named_parameters()
    ):

        name_lower = (
            name.lower()
        )

        if (
            ".crossattention."
            in name_lower
            and "lora_" not in name_lower
        ):

            if parameter.requires_grad:

                base_count += (
                    parameter.numel()
                )

                base_tensors += 1

        if (
            "lora_" in name_lower
            and parameter.requires_grad
        ):

            lora_count += (
                parameter.numel()
            )

            lora_tensors += 1

    print(
        "\nCross-attention trainability:"
    )

    print(
        "  Base parameters : "
        f"{base_count:,}"
    )

    print(
        "  Base tensors    : "
        f"{base_tensors}"
    )

    print(
        "  LoRA parameters : "
        f"{lora_count:,}"
    )

    print(
        "  LoRA tensors    : "
        f"{lora_tensors}"
    )

    if base_count != 28_348_416:

        raise RuntimeError(
            "Unexpected trainable cross-attention "
            "base parameter count: "
            f"{base_count:,}. "
            "Expected 28,348,416."
        )

    if lora_count != 958_464:

        raise RuntimeError(
            "Unexpected trainable LoRA parameter "
            f"count: {lora_count:,}. "
            "Expected 958,464."
        )

    if (
        base_count + lora_count
        != EXPECTED_DECODER_TRAINABLE
    ):

        raise RuntimeError(
            "Cross-attention + LoRA parameter "
            "count does not match the "
            "verified training architecture."
        )

    print(
        "Cross-attention verification: PASS"
    )


def _load_tokenizer():

    tokenizer_dir = (
        MODULE2_ROOT
        / "tokenizer"
    )

    if not tokenizer_dir.exists():

        raise FileNotFoundError(
            "Module 2 tokenizer directory "
            "was not found:\n"
            f"{tokenizer_dir}"
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True
        )
    )

    if (
        tokenizer.pad_token_id
        is None
    ):

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    if (
        tokenizer.vocab_size
        != 50257
    ):

        raise RuntimeError(
            "Unexpected tokenizer vocabulary "
            f"size: {tokenizer.vocab_size}. "
            "Expected 50257."
        )

    if (
        tokenizer.eos_token_id
        != 50256
    ):

        raise RuntimeError(
            "Unexpected EOS token ID: "
            f"{tokenizer.eos_token_id}. "
            "Expected 50256."
        )

    if (
        tokenizer.pad_token_id
        != 50256
    ):

        raise RuntimeError(
            "Unexpected PAD token ID: "
            f"{tokenizer.pad_token_id}. "
            "Expected 50256."
        )

    return tokenizer


def _resolve_checkpoint_path():

    direct_path = (
        MODULE2_ROOT
        / DEFAULT_CHECKPOINT_NAME
    )

    if direct_path.exists():

        return direct_path

    checkpoint_dir = (
        MODULE2_ROOT
        / "checkpoints"
    )

    if checkpoint_dir.exists():

        candidates = list(
            checkpoint_dir.rglob(
                DEFAULT_CHECKPOINT_NAME
            )
        )

        if candidates:

            return candidates[0]

    candidates = list(
        MODULE2_ROOT.rglob(
            DEFAULT_CHECKPOINT_NAME
        )
    )

    if candidates:

        return candidates[0]

    raise FileNotFoundError(
        "Module 2 V2 checkpoint not found.\n"
        f"Searched under:\n{MODULE2_ROOT}"
    )


def _load_checkpoint(
    visual_conditioner,
    decoder,
    checkpoint_path
):

    print(
        "Loading Module 2 V2 checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    required_keys = [
        "visual_conditioner_state_dict",
        "decoder_state_dict"
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise RuntimeError(
                "Checkpoint is missing required "
                f"key: {key}\n"
                f"Available keys: "
                f"{list(checkpoint.keys())}"
            )

    visual_state = checkpoint[
        "visual_conditioner_state_dict"
    ]

    decoder_state = checkpoint[
        "decoder_state_dict"
    ]

    visual_model_keys = set(
        visual_conditioner
        .state_dict()
        .keys()
    )

    visual_checkpoint_keys = set(
        visual_state.keys()
    )

    visual_missing = sorted(
        visual_model_keys
        - visual_checkpoint_keys
    )

    visual_unexpected = sorted(
        visual_checkpoint_keys
        - visual_model_keys
    )

    if (
        visual_missing
        or visual_unexpected
    ):

        print(
            "\nVISUAL CHECKPOINT KEY "
            "DIAGNOSTICS"
        )

        if visual_missing:

            print(
                "\nMissing visual keys:"
            )

            for key in (
                visual_missing[:30]
            ):

                print(
                    " ",
                    key
                )

        if visual_unexpected:

            print(
                "\nUnexpected visual keys:"
            )

            for key in (
                visual_unexpected[:30]
            ):

                print(
                    " ",
                    key
                )

        raise RuntimeError(
            "Visual conditioner checkpoint "
            "does not match the training "
            "architecture."
        )

    visual_result = (
        visual_conditioner.load_state_dict(
            visual_state,
            strict=True
        )
    )

    if visual_result.missing_keys:

        raise RuntimeError(
            "Unexpected missing visual "
            "checkpoint keys: "
            f"{visual_result.missing_keys}"
        )

    if visual_result.unexpected_keys:

        raise RuntimeError(
            "Unexpected visual checkpoint "
            "keys: "
            f"{visual_result.unexpected_keys}"
        )

    decoder_model_keys = set(
        decoder
        .state_dict()
        .keys()
    )

    decoder_checkpoint_keys = set(
        decoder_state.keys()
    )

    decoder_missing = sorted(
        decoder_model_keys
        - decoder_checkpoint_keys
    )

    decoder_unexpected = sorted(
        decoder_checkpoint_keys
        - decoder_model_keys
    )

    if (
        decoder_missing
        or decoder_unexpected
    ):

        print(
            "\nDECODER CHECKPOINT KEY "
            "DIAGNOSTICS"
        )

        if decoder_missing:

            print(
                "\nMissing decoder keys:"
            )

            for key in (
                decoder_missing[:40]
            ):

                print(
                    " ",
                    key
                )

        if decoder_unexpected:

            print(
                "\nUnexpected decoder keys:"
            )

            for key in (
                decoder_unexpected[:40]
            ):

                print(
                    " ",
                    key
                )

        raise RuntimeError(
            "Decoder checkpoint does not "
            "match the exact Module 2 V2 "
            "training architecture."
        )

    decoder_result = (
        decoder.load_state_dict(
            decoder_state,
            strict=True
        )
    )

    if decoder_result.missing_keys:

        raise RuntimeError(
            "Unexpected missing decoder "
            "checkpoint keys: "
            f"{decoder_result.missing_keys}"
        )

    if decoder_result.unexpected_keys:

        raise RuntimeError(
            "Unexpected decoder "
            "checkpoint keys: "
            f"{decoder_result.unexpected_keys}"
        )

    print(
        "Checkpoint state dictionaries: PASS"
    )

    return checkpoint


def _verify_loaded_models(
    visual_conditioner,
    decoder
):

    visual_info = (
        _count_parameters(
            visual_conditioner
        )
    )

    decoder_info = (
        _count_parameters(
            decoder
        )
    )

    total_trainable = (
        visual_info["trainable"]
        + decoder_info["trainable"]
    )

    print(
        "\nMODULE 2 PARAMETER VERIFICATION"
    )

    print(
        "\nVisual conditioner:"
    )

    print(
        f"  Total     : "
        f"{visual_info['total']:,}"
    )

    print(
        f"  Trainable : "
        f"{visual_info['trainable']:,}"
    )

    print(
        f"  Frozen    : "
        f"{visual_info['frozen']:,}"
    )

    print(
        "\nDecoder:"
    )

    print(
        f"  Total     : "
        f"{decoder_info['total']:,}"
    )

    print(
        f"  Trainable : "
        f"{decoder_info['trainable']:,}"
    )

    print(
        f"  Frozen    : "
        f"{decoder_info['frozen']:,}"
    )

    print(
        "\nTotal Module 2 trainable:"
    )

    print(
        f"  {total_trainable:,}"
    )

    if (
        visual_info["total"]
        != EXPECTED_VISUAL_TOTAL
    ):

        raise RuntimeError(
            "Visual total parameter count "
            "does not match V2."
        )

    if (
        visual_info["trainable"]
        != EXPECTED_VISUAL_TRAINABLE
    ):

        raise RuntimeError(
            "Visual trainable parameter "
            "count does not match V2."
        )

    if (
        visual_info["frozen"]
        != EXPECTED_VISUAL_FROZEN
    ):

        raise RuntimeError(
            "Visual frozen parameter count "
            "does not match V2."
        )

    if (
        decoder_info["total"]
        != EXPECTED_DECODER_TOTAL
    ):

        raise RuntimeError(
            "Decoder total parameter count "
            "does not match V2."
        )

    if (
        decoder_info["trainable"]
        != EXPECTED_DECODER_TRAINABLE
    ):

        raise RuntimeError(
            "Decoder trainable parameter "
            "count does not match V2."
        )

    if (
        decoder_info["frozen"]
        != EXPECTED_DECODER_FROZEN
    ):

        raise RuntimeError(
            "Decoder frozen parameter count "
            "does not match V2."
        )

    if (
        total_trainable
        != EXPECTED_TOTAL_TRAINABLE
    ):

        raise RuntimeError(
            "Total trainable parameter count "
            "does not match V2."
        )

    print(
        "\nParameter verification: PASS"
    )


def load_module2(
    device=None
):

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if isinstance(
        device,
        str
    ):

        device = torch.device(
            device
        )

    print(
        "=" * 80
    )

    print(
        "MODULE 2 V2 — LOADING"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {device}"
    )

    checkpoint_path = (
        _resolve_checkpoint_path()
    )

    print(
        "\nCheckpoint:"
    )

    print(
        checkpoint_path
    )

    print(
        "Checkpoint size: "
        f"{checkpoint_path.stat().st_size / (1024 ** 3):.3f} GB"
    )

    visual_conditioner, preprocess = (
        _build_visual_conditioner()
    )

    decoder = _build_decoder()

    tokenizer = _load_tokenizer()

    _verify_cross_attention_trainability(
        decoder
    )

    checkpoint = _load_checkpoint(
        visual_conditioner=(
            visual_conditioner
        ),
        decoder=decoder,
        checkpoint_path=checkpoint_path
    )

    visual_conditioner = (
        visual_conditioner.to(
            device
        )
    )

    decoder = (
        decoder.to(
            device
        )
    )

    visual_conditioner.eval()

    decoder.eval()

    _verify_loaded_models(
        visual_conditioner,
        decoder
    )

    parameter_info = {
        "visual_conditioner":
            _count_parameters(
                visual_conditioner
            ),
        "decoder":
            _count_parameters(
                decoder
            )
    }
    print(
        "\nTokenizer:"
    )
    print(
        f"  Vocabulary size : "
        f"{tokenizer.vocab_size}"
    )
    print(
        f"  EOS token ID    : "
        f"{tokenizer.eos_token_id}"
    )
    print(
        f"  PAD token ID    : "
        f"{tokenizer.pad_token_id}"
    )
    print(
        f"  Padding side    : "
        f"{tokenizer.padding_side}"
    )
    print(
        "\nExpected visual token shape:"
    )
    print(
        f"  [B, {EXPECTED_VISUAL_TOKENS}, "
        f"{VISUAL_HIDDEN_SIZE}]"
    )
    print(
        "\nMODULE 2 V2 LOADING: PASS"
    )
    print(
        "=" * 80
    )
    return {
        "visual_conditioner":
            visual_conditioner,
        "decoder":
            decoder,
        "tokenizer":
            tokenizer,
        "preprocess":
            preprocess,
        "device":
            device,
        "checkpoint":
            checkpoint,
        "checkpoint_path":
            checkpoint_path,
        "parameter_info":
            parameter_info
    }
__all__ = [
    "VisualFeatureProjection",
    "BiomedCLIPVisualConditioner",
    "load_module2"
]