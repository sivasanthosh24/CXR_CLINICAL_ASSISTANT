import torch
NUM_BEAMS = 4
MAX_NEW_TOKENS = 160
NO_REPEAT_NGRAM_SIZE = 3
LENGTH_PENALTY = 1.0
EARLY_STOPPING = True
DO_SAMPLE = False
class Module2Generator:
    def __init__(
        self,
        visual_conditioner,
        decoder,
        tokenizer,
        device
    ):
        self.visual_conditioner = (
            visual_conditioner
        )
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )
        self.tokenizer.pad_token_id = (
            self.tokenizer.eos_token_id
        )
        if (
            self.tokenizer.padding_side
            != "left"
        ):
            raise RuntimeError(
                "GPT-2 tokenizer must use "
                "left padding for generation."
            )
        if (
            self.tokenizer.eos_token_id
            is None
        ):
            raise RuntimeError(
                "Tokenizer EOS token is not configured."
            )
        self.visual_conditioner.eval()
        self.decoder.eval()
    @torch.inference_mode()
    def generate(
        self,
        pixel_values
    ):
        pixel_values = pixel_values.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True
        )
        visual_tokens = (
            self.visual_conditioner(
                pixel_values
            )
        )
        if visual_tokens.ndim != 3:
            raise RuntimeError(
                "Unexpected visual token shape: "
                f"{tuple(visual_tokens.shape)}"
            )
        batch_size = (
            visual_tokens.size(0)
        )
        visual_token_count = (
            visual_tokens.size(1)
        )
        visual_hidden_size = (
            visual_tokens.size(2)
        )
        if visual_token_count != 197:
            raise RuntimeError(
                "Unexpected visual token count: "
                f"{visual_token_count}. "
                "Expected 197."
            )
        if visual_hidden_size != 768:
            raise RuntimeError(
                "Unexpected visual hidden size: "
                f"{visual_hidden_size}. "
                "Expected 768."
            )
        encoder_attention_mask = torch.ones(
            batch_size,
            visual_token_count,
            dtype=torch.long,
            device=self.device
        )
        decoder_input_ids = torch.full(
            (
                batch_size,
                1
            ),
            self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.device
        )
        generated_ids = (
            self.decoder.generate(
                input_ids=decoder_input_ids,
                encoder_hidden_states=visual_tokens,
                encoder_attention_mask=(
                    encoder_attention_mask
                ),
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=DO_SAMPLE,
                no_repeat_ngram_size=(
                    NO_REPEAT_NGRAM_SIZE
                ),
                length_penalty=LENGTH_PENALTY,
                early_stopping=EARLY_STOPPING,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),
                use_cache=True
            )
        )
        generated_texts = (
            self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        )
        generated_texts = [
            text.strip()
            for text in generated_texts
        ]
        return generated_texts, generated_ids
    def generate_single(
        self,
        pixel_values
    ):
        if pixel_values.ndim == 3:

            pixel_values = (
                pixel_values.unsqueeze(0)
            )
        generated_texts, generated_ids = (
            self.generate(
                pixel_values
            )
        )
        return (
            generated_texts[0],
            generated_ids[0]
        )
__all__ = [
    "Module2Generator",
    "NUM_BEAMS",
    "MAX_NEW_TOKENS",
    "NO_REPEAT_NGRAM_SIZE",
    "LENGTH_PENALTY",
    "EARLY_STOPPING",
    "DO_SAMPLE"
]