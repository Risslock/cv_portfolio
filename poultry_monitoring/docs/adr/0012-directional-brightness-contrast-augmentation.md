# ADR 0012: Directional (asymmetric) brightness/contrast training-time augmentation — new best

## Status

Accepted

## Context

Two earlier attempts at using this project's own "birds are white, background is dark
litter" domain observation both failed:

- **Test-time preprocessing** (ADR 0004, reconfirmed with a `brightness_contrast`
  candidate and a narrow-cutoff `autocontrast` re-test): every deterministic,
  inference-only transform tested — including a domain-motivated one that looked clean
  in a visual sweep — hurt `box_map50_95`, `brightness_contrast` most of all
  (`-0.027` to `-0.045`, worse than the original CLAHE/hist-eq failures). Same mechanism
  each time: a transform the model never saw during training creates a real
  train/inference distribution mismatch.
- **The ADR 0008 conservative hyperparameter search** (ADR 0011): its winning
  configuration underperformed the pre-session baseline at real scale, in both
  cold-start and warm-start regimes.

ADR 0004 already named the actual fix, just hadn't been applied to this specific
observation yet: "baking a *matching* enhancement into training augmentation, not
bolting a mismatched one onto inference, is what actually helped" (referring to
`_RandomCutoffAutoContrast`). The open question was whether Ultralytics had a native
knob for this. Checked directly rather than assumed:

- `hsv_v` (brightness) is genuinely native — but it's a uniform exposure jitter, not a
  contrast-widening operation, so it doesn't target the same thing the sweep found
  valuable (background/bird separation, not overall brightness).
- Contrast and autocontrast have **no** native Ultralytics equivalent for detection —
  only this project's own `RandomBrightnessContrast`/`_RandomCutoffAutoContrast` in
  `augmentation/shared.py`, both already implemented and already searchable via
  `CUSTOM_AUGMENTATION_PARAM_RANGES`.

The one real gap: `RandomBrightnessContrast`'s `brightness_limit`/`contrast_limit` are
symmetric by default (`±limit`, roughly half the time making the image *less* contrasty
or brighter — the opposite direction from what the sweep found helpful), and
`_build_custom_augmentations` passes hyperparameter values straight through to
Albumentations with no type coercion. Verified directly: Albumentations'
`RandomBrightnessContrast` already accepts `brightness_limit`/`contrast_limit` as an
asymmetric `(low, high)` tuple, and a JSON array (`[-0.35, -0.05]`, a `list` not a
`tuple`) works identically — so biasing the augmentation toward the domain-motivated
direction needed **zero code changes**, just a different hyperparameters file.

## Decision

Ran `train()` cold-start with `hyperparameters = {"brightness_limit": [-0.35, -0.05],
"contrast_limit": [0.3, 0.7]}` (everything else at the pre-session baseline's config:
`epochs=300`, `patience=15`, `fraction=1.0`, `freeze=10`, `lr0=0.01`), then carried the
same fixed hyperparameters through all three `progressive_unfreeze_train` stages
(`DEFAULT_UNFREEZE_STAGES`, unmodified schedule).

Results (`box_map50_95`, this project's val metric):

| stage | pre-session baseline | directional-contrast | Δ |
|---|---|---|---|
| cold-start | 0.8768 | 0.8769 | +0.0001 (noise) |
| unfreeze stage 0 (`freeze=10`) | 0.8802 | 0.8745 | −0.0057 |
| unfreeze stage 1 (`freeze=5`) | 0.8830 | 0.8813 | −0.0017 |
| **unfreeze stage 2 (`freeze=0`, final)** | **0.8919** | **0.8929** | **+0.0010** |

Full final-stage comparison: `box_map50=0.9878` (+0.0005), `box_map50_95=0.8929`
(+0.0010), `box_precision=0.9706` (+0.0058), `box_recall=0.9488` (−0.0040).

Notably, this configuration is *behind* the baseline through the first two unfreeze
stages and only overtakes once the whole backbone is trainable — the opposite of the
ADR 0011 config, which stayed behind at every stage and never caught up. Plausible
read: the directional bias needs real capacity to exploit (more of the network
trainable) before it pays off; with most of the backbone frozen there's less room for
it to teach anything the frozen features don't already encode.

**Adopted as the new production configuration.** `yolo26n-unfreeze-directional-contrast-stage2/weights/best.pt`
is the new best-known `yolo26n` checkpoint, replacing the pre-session one.

## Consequences

- The improvement is real but modest (`+0.001` `box_map50_95`, `+0.006` precision,
  `-0.004` recall) — worth adopting, not worth overselling as a breakthrough.
- `augmentation/shared.py`'s `build_domain_transforms` *function defaults* are
  deliberately **not** changed — `brightness_limit=0.3, contrast_limit=0.3` (symmetric)
  stays the fallback for any caller that doesn't explicitly override, including
  `tune_augmentation_parameters`'s own search. This directional pair is a specific,
  deliberately-chosen `train()`/`unfreeze` hyperparameters override, not a new default —
  baking it into the function signature would silently change behavior for every other
  caller, a separate decision not made here.
- `yolo26s` hasn't been retrained with this config — still on the old near-default
  augmentation. Natural follow-up, not launched automatically.
- The exact hyperparameters file used
  (`data/ChickenDet/YOLO/hyperparameters_directional_contrast.json`) is gitignored
  (under `data/`) — reproduce from the literal values in this ADR, not by assuming the
  file exists in a fresh clone.
- This closes the loop ADR 0004 opened: three attempts at this same domain observation
  (test-time fixed, search-derived, training-time directional), only the last one — the
  one ADR 0004 predicted would work — actually did.
