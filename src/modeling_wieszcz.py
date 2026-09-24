"""A `transformers` entry point for the released checkpoints.

`model.py` is a plain `nn.Module` taking a config dict, which is what trained the
weights. `WieszczForCausalLM` holds it unmodified and adds the three things
`AutoModelForCausalLM.from_pretrained` requires: a `PretrainedConfig`, a forward
returning `CausalLMOutputWithPast`, and the embedding accessors weight tying goes
through.

Wrapping rather than porting keeps one definition of the architecture. A rewritten
`PreTrainedModel` version of the network would be a second implementation obliged to stay
bit-identical to the one that trained the weights, and when it drifted the model would
still load and quietly compute something else.

The checkpoint keeps the bare `GPT` parameter names rather than gaining a `model.` prefix,
so one `model.safetensors` serves both entry points: `from_pretrained` prepends
`base_model_prefix` when it meets a checkpoint saved from the base network, while the
`GPT` path in the model card loads the same file directly.

The wrapper costs speed. `generate` here recomputes the whole prefix at every step,
because the KV cache in `model.py` predates the library's `Cache` protocol and does not
implement it. Anything long or repeated should call `GPT.stream`, which is cached; the
model card shows both paths.
"""

from __future__ import annotations

from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .model import GPT

GPT_FIELDS = ("vocab_size", "block_size", "n_layer", "n_head", "n_kv_head",
              "n_embd", "multiple_of", "rope_theta", "dropout")


class WieszczConfig(PretrainedConfig):
    """The training config dict, restated as the library expects to receive it.

    `attribute_map` exposes the standard names generic tooling reaches for without
    renaming the fields the training code and the paper both use.
    """

    model_type = "wieszcz"
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_key_value_heads": "n_kv_head",
        "num_hidden_layers": "n_layer",
        "max_position_embeddings": "block_size",
    }

    def __init__(self, vocab_size: int = 8000, block_size: int = 1024, n_layer: int = 12,
                 n_head: int = 12, n_kv_head: int = 4, n_embd: int = 768,
                 multiple_of: int = 64, rope_theta: float = 10000, dropout: float = 0.0,
                 tie_word_embeddings: bool = True, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.multiple_of = multiple_of
        self.rope_theta = rope_theta
        self.dropout = dropout
        # The tokenizer terminates every document with <|endoftext|>, which it numbers 0,
        # so that id is the only sentinel the model ever saw.
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 0)
        kwargs.setdefault("pad_token_id", 0)
        # Caching is the wrapper's one unimplemented promise, so it is declined here
        # rather than attempted and silently ignored downstream.
        kwargs.setdefault("use_cache", False)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def to_gpt_cfg(self) -> dict:
        return {field: getattr(self, field) for field in GPT_FIELDS}


class WieszczForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = WieszczConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _tied_weights_keys = ["model.head.weight"]
    _supports_sdpa = True

    def __init__(self, config: WieszczConfig):
        super().__init__(config)
        self.model = GPT(config.to_gpt_cfg())
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

    def get_output_embeddings(self):
        return self.model.head

    def set_output_embeddings(self, value):
        self.model.head = value

    def tie_weights(self):
        """Re-point the output head at the embedding table.

        `GPT.__init__` ties the two by assigning one tensor to both, and loading a
        checkpoint replaces the embedding tensor, which silently breaks that identity: the
        head keeps the tensor it was built with. Restating the assignment here is what
        makes the tie survive `from_pretrained`, and without it the head is left on the
        meta device and the first forward pass raises.
        """
        self.model.head.weight = self.model.tok_emb.weight

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                return_dict=None, **kwargs):
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError(
                "this wrapper has no padding mask, so a padded batch would attend to and "
                "score pad positions as text. Generate or score one sequence at a time.")
        length = input_ids.size(1)
        if length > self.config.block_size:
            raise ValueError(
                f"the model was trained with a {self.config.block_size}-token context and "
                f"its rotary tables are that long; got {length} tokens.")

        logits, _ = self.model(input_ids)

        loss = None
        if labels is not None:
            # `GPT.forward` scores targets already aligned to their inputs; the library's
            # convention hands labels aligned to `input_ids`, so the shift happens here.
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Uncached decoding, so every step resubmits the prefix. Past the trained context
        # the window slides instead of raising, which is the useful behaviour for a base
        # model asked to keep writing.
        return {"input_ids": input_ids[:, -self.config.block_size:]}

    def stream(self, *args, **kwargs):
        """The cached sampler from `model.py`, reached through the wrapper.

        Gradients are already disabled inside `GPT.stream`, which is where they have to be
        handled: this method returns a generator, so a context manager opened here would
        close before the first token was ever sampled.
        """
        return self.model.stream(*args, **kwargs)
