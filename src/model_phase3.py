import torch
from torch import nn


class Block(nn.Module):
    """One transformer block: attention then MLP, both with a residual connection."""

    def __init__(self, d_model=128, n_heads=4, d_mlp=512, preln=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)
        self.preln = bool(preln)
        if self.preln:
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)


class ContextualTransformer(nn.Module):
    """One-block causal transformer with an explicit context token."""

    def __init__(self, p=59, n_contexts=4, d_model=128, n_heads=4, d_mlp=512,
                 n_layers=1, preln=0):
        super().__init__()
        self.p = p
        self.n_contexts = n_contexts
        self.context_offset = p
        self.equals_token = p + n_contexts
        self.vocab = p + n_contexts + 1
        self.token_embed = nn.Embedding(self.vocab, d_model)
        self.pos_embed = nn.Parameter(torch.empty(4, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)
        self.n_layers = int(n_layers)
        self.preln = bool(preln)
        if self.preln:
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.ln_final = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [Block(d_model, n_heads, d_mlp, self.preln)
             for _ in range(self.n_layers - 1)])
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1),
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, tokens, return_cache=False, ablate_directions=None,
                ablate_mean=None):
        x0 = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
        x = x0
        hidden = None
        for index in range(self.n_layers):
            if index == 0:
                attn_module = self.attn
                mlp_in = self.mlp_in
                mlp_out = self.mlp_out
                ln_attn = self.ln_attn if self.preln else None
                ln_mlp = self.ln_mlp if self.preln else None
            else:
                block = self.layers[index - 1]
                attn_module = block.attn
                mlp_in = block.mlp_in
                mlp_out = block.mlp_out
                ln_attn = block.ln_attn if block.preln else None
                ln_mlp = block.ln_mlp if block.preln else None
            attn_input = ln_attn(x) if ln_attn is not None else x
            attn_out, attn_weights = attn_module(
                attn_input, attn_input, attn_input, attn_mask=self.causal_mask,
                need_weights=return_cache, average_attn_weights=False,
            )
            x1 = x + attn_out
            mlp_input = ln_mlp(x1) if ln_mlp is not None else x1
            hidden = torch.relu(mlp_in(mlp_input))

            if ablate_directions is not None:
                directions = ablate_directions.to(hidden.device, hidden.dtype)
                if directions.ndim == 1:
                    directions = directions.unsqueeze(0)
                mean = torch.zeros(hidden.shape[-1], device=hidden.device,
                                   dtype=hidden.dtype)
                if ablate_mean is not None:
                    mean = ablate_mean.to(hidden.device, hidden.dtype)
                last = hidden[:, -1, :]
                centered = last - mean
                projection = (centered @ directions.T) @ directions
                hidden = hidden.clone()
                hidden[:, -1, :] = last - projection

            x = x1 + mlp_out(hidden)

        final = self.ln_final(x) if self.preln else x
        logits = self.unembed(final[:, -1, :])
        if return_cache:
            return logits, {
                "embed": x0.detach(),
                "attn": attn_out.detach(),
                "post_attn": x1.detach(),
                "mlp_hidden": hidden.detach(),
                "resid": x.detach(),
                "attn_weights": attn_weights.detach(),
            }
        return logits
