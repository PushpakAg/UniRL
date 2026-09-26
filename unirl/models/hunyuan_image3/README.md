# HunyuanImage3 (Tencent HunyuanImage-3.0) — mixed AR + diffusion package

> **Where it fits:** one model package under [`unirl/models/`](../README.md), covering both the
> `gen_text` AR path (`ar.py`) and the `gen_image` diffusion path (`diffusion.py`) of the same
> checkpoint. Recipes live in `examples/unified_model/hi3_*.yaml`.

## What it is

The transformer is loaded as upstream `trust_remote_code` code (`modeling_hunyuan_image_3.py`
from `tencent/HunyuanImage-3.0-Instruct`), not a vendored copy. UniRL drives its HF-style
generation helpers (`prepare_inputs_for_generation`,
`_update_model_kwargs_for_generation`) step by step instead of calling `generate`, so the
train-side AR loop owns the KV cache, the attention mask, and the `input_ids` buffer itself.
`compat.py` holds the transformers-5.x runtime shims.

## Gotchas

- **A `dynamic=True` KV cache cannot express a batch.** `HunyuanStaticCache.update` trims the
  returned KV to one scalar end position (`cache_position[0, -1] + 1`) and guards it with
  `assert len(cache_position) == 1` — where `len()` on a 2-D cache position is the *batch*
  dim, so every `B > 1` prefill raises (#452). Per-row right-padded prompts have per-row end
  positions, which no single scalar can represent. `HunyuanImage3ARStep.init_state` therefore
  only passes `dynamic=True` at `B == 1`; `B > 1` uses the full static cache plus an explicit
  mask (below). The diffusion path already builds `dynamic=False` because its sequence never
  grows.
- **Upstream deletes the attention mask during `gen_text` decode** ("Remove attention mask to
  use full attention of 1 x seqlen in decode steps"), which is only safe for a cache trimmed to
  the valid length. Over a static cache the decode query would also attend the right-pad KV
  written at prefill and the never-written tail slots, so `HunyuanImage3ARStep.step` supplies a
  bool mask admitting row `i`'s keys `[0, real_pos_i + step_idx)`. At prefill it right-pads the
  mask's key axis instead: a static cache returns its full `max_cache_len` key axis while the
  prefill mask is only `L` wide, and SDPA requires the two to agree.
- **A sampled token must be scattered, not appended.** Upstream re-reads the next input token
  with `input_ids.gather(1, position_ids)` where `position_ids` is `real_pos` — the first
  `<pad>` index, advanced by one per step. So the token sampled at step `s` belongs at column
  `real_pos_i + s` of the right-padded buffer. Appending at the tail is equivalent only when
  `real_pos == L` (the longest row); any right-padded row, including a padded `B == 1` prompt,
  would otherwise be fed its own pad token as the next input.
- **`real_pos` is required at every batch size.** After prefill, upstream
  `_update_model_kwargs_for_generation` sets the decode `position_ids` from
  `tokenizer_output.real_pos`; without `tokenizer_output` it takes its decode branch at prefill
  and every later step silently re-feeds the shifted prompt. `init_state` therefore requires
  `tokenizer_output` and reads the same values from `fused.prompt_lengths`, which `embed_for_ar`
  normalizes to `[B]`.
- **`position_ids` are the KV-cache write indices, so padding them with a constant corrupts
  slot 0.** Upstream passes `position_ids` straight through as `cache_position`
  (`HunyuanImage3SDPAAttention.forward`), and `update` applies it with `index_copy_`. When
  `FusedMultimodalCondition.concat` padded the ragged `L` axis with `0`, every pad position of
  every short row wrote its KV onto cache slot 0 — last write wins, so each padded row's first
  real token was silently replaced (measured: slot 0 off by ~4.0, every other slot
  bit-identical). `_pad_positions` pads with each row's own continuing indices instead, which
  keeps the pad writes on distinct slots that the decode mask then excludes.
- All of these are properties of the upstream remote code, so re-check them when the
  checkpoint revision moves. If upstream ever trims dynamic KV per row, `B > 1` can drop back
  to `dynamic=True` and the decode mask becomes redundant.
