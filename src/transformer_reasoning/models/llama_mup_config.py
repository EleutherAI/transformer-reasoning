from transformers import LlamaConfig

class LlamaMuPConfig(LlamaConfig):
    def __init__(
        self,
        attn_mult=None,  # muP attention multiplier
        **kwargs
    ):
        super().__init__(**kwargs)
        if attn_mult is None:
        # defaults back to 1/sqrt(d) attn
            self.attn_mult = (self.hidden_size / self.num_attention_heads)**0.5
        else:
            self.attn_mult = attn_mult
