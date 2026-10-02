"""Bundle a snowGAN training run into a HuggingFace Hub model release.

Uploads the weights + sidecar configs + an auto-generated MANIFEST.md to a
HuggingFace model repo at a tagged revision. Downstream consumers (AvAI,
etc.) then pin exactly which backbone they're using via
``snowgan.weights.fetch("RMDig/snowGAN-core", "v0.1.0")``.

Usage:
    python scripts/release_weights.py \\
        --save-dir keras/snowgan/magnified_profiles \\
        --repo RMDig/snowGAN-magnified-profile \\
        --tag v0.1.0 \\
        --notes "First public release; trained to fade_step 428k."

Requires (only when actually uploading):
    pip install huggingface_hub
    huggingface-cli login   # or set HF_TOKEN env var

Use ``--dry-run`` to inspect the manifest and the upload list without
contacting HF Hub. Use ``--create-repo`` to bootstrap a new model repo on
first release.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


REQUIRED_ARTIFACTS = (
    "discriminator.weights.h5",
    "discriminator_config.json",
    "generator.weights.h5",
    "generator_config.json",
)
OPTIONAL_ARTIFACTS = (
    "generator_ema.weights.h5",
    "generator_fade_endpoints.weights.h5",
    "discriminator_lowres.weights.h5",
)
MANIFEST_FILENAME = "MANIFEST.md"
MODEL_CARD_FILENAME = "README.md"


def build_manifest(
    save_dir: Path,
    tag: str,
    notes: str,
    *,
    snowgan_version: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> str:
    """Generate the MANIFEST.md body for a release.

    Pulled out as a standalone function so tests can exercise the
    formatting without touching the filesystem or HF Hub. ``snowgan_version``
    and ``git_sha`` are injectable so tests can pin them; in production
    they're auto-discovered.
    """
    disc_cfg = json.loads((save_dir / "discriminator_config.json").read_text())
    gen_cfg = json.loads((save_dir / "generator_config.json").read_text())

    if snowgan_version is None:
        try:
            import snowgan  # noqa: WPS433 — runtime introspection
            snowgan_version = getattr(snowgan, "__version__", "unknown")
        except Exception:
            snowgan_version = "unknown"

    if git_sha is None:
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parent.parent),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            git_sha = "unknown"

    artifacts = sorted(
        p.name for p in save_dir.iterdir()
        if p.is_file() and (p.suffix in (".h5", ".json") or p.name == MANIFEST_FILENAME)
    )
    artifact_lines = "\n".join(f"- `{a}`" for a in artifacts)

    splits = (
        f"- trained_pool: {len(disc_cfg.get('trained_pool') or [])} groups\n"
        f"- validation_pool: {len(disc_cfg.get('validation_pool') or [])} groups\n"
        f"- test_pool: {len(disc_cfg.get('test_pool') or [])} groups"
    )

    return f"""# snowGAN release {tag}

Snapshot of a snowGAN training run, packaged for downstream consumers
(AvAI etc.). Consume with `snowgan.weights.fetch("{{repo}}", "{tag}")`.

## Provenance
- snowGAN package version: `{snowgan_version}`
- snowGAN git SHA at release: `{git_sha}`
- Training dataset: `{gen_cfg.get('dataset')}`
- Source save_dir: `{save_dir.as_posix()}`

## Model architecture
- depth: {gen_cfg.get('depth')}
- resolution: {gen_cfg.get('resolution')}
- modality: {gen_cfg.get('modality')}
- channels: {gen_cfg.get('channels')}
- latent_dim: {gen_cfg.get('latent_dim')}
- filter_counts (gen): {gen_cfg.get('filter_counts')}
- filter_counts (disc): {disc_cfg.get('filter_counts')}
- kernel_size / kernel_stride: {gen_cfg.get('kernel_size')} / {gen_cfg.get('kernel_stride')}

## Training state at release
- fade_step: {gen_cfg.get('fade_step')} (gen) / {disc_cfg.get('fade_step')} (disc)
- fade_steps target: {gen_cfg.get('fade_steps')}
- current_epoch: {gen_cfg.get('current_epoch')}
- training_steps: gen={gen_cfg.get('training_steps')}, disc={disc_cfg.get('training_steps')}
- lambda_gp: {disc_cfg.get('lambda_gp')}

## Advanced training options
- spectral_norm: {disc_cfg.get('spectral_norm')}
- augment: {disc_cfg.get('augment')}
- multiscale_disc: {disc_cfg.get('multiscale_disc')}
- ema_decay: {gen_cfg.get('ema_decay')}
- lr_decay: {gen_cfg.get('lr_decay')} (lr_min={gen_cfg.get('lr_min')})
- ada_target: {gen_cfg.get('ada_target')}
- adaptive_steps: {gen_cfg.get('adaptive_steps')}
- grad_clip_norm: {gen_cfg.get('grad_clip_norm')}
- fid_interval: {gen_cfg.get('fid_interval')}

## Persisted dataset splits
{splits}

## Artifacts
{artifact_lines}

## Notes
{notes or '(none)'}
"""


def build_model_card(
    save_dir: Path,
    tag: str,
    repo: str,
    *,
    intended_use: str = "",
    limitations: str = "",
    notes: str = "",
    quality_evidence: str = "",
) -> str:
    """Generate the HF Hub model card (README.md) body for a release.

    The model card is the *public-facing* surface of the HF Hub repo — it
    renders on the model's page and tells consumers what this is and how to
    use it. Mostly derived from the configs (so future releases stay
    coherent without manual editing); the prose sections (``intended_use``,
    ``limitations``, ``notes``) accept free-text overrides.

    The MANIFEST.md is the *technical* doc (provenance, every flag, every
    artifact) and lives alongside the README. Different audiences:
    MANIFEST = debugging; README = end-users.

    Limitations / intended_use accept multi-line markdown; if empty, the
    sections fall back to a sensible per-modality default.
    """
    disc_cfg = json.loads((save_dir / "discriminator_config.json").read_text())
    gen_cfg = json.loads((save_dir / "generator_config.json").read_text())

    modality = gen_cfg.get("modality", "magnified_profile")
    depth = gen_cfg.get("depth", 1)
    resolution = gen_cfg.get("resolution", [1024, 1024])
    res_str = f"{resolution[0]}x{resolution[1]}"
    filter_counts = disc_cfg.get("filter_counts", [])
    feature_dim = depth * (resolution[0] // (2 ** len(filter_counts))) ** 2 * (filter_counts[-1] if filter_counts else 0)
    fade_step = max(int(gen_cfg.get("fade_step") or 0), int(disc_cfg.get("fade_step") or 0))
    trained_count = len(disc_cfg.get("trained_pool") or [])
    val_count = len(disc_cfg.get("validation_pool") or [])
    test_count = len(disc_cfg.get("test_pool") or [])

    # YAML frontmatter — what HF Hub reads to populate the model page sidebar
    # (license, tags, dataset link, etc.). Keep keys/values quoted to avoid
    # ambiguity with the YAML parser.
    tags = [
        "gan", "wgan-gp", "image-generation", "snowpack",
        f"modality-{modality}",
    ]
    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    frontmatter = (
        "---\n"
        "license: apache-2.0\n"
        "language:\n  - en\n"
        f"library_name: snowgan\n"
        f"pipeline_tag: image-generation\n"
        f"tags:\n{tags_yaml}\n"
        f"datasets:\n  - {gen_cfg.get('dataset', 'rmdig/rocky_mountain_snowpack')}\n"
        "---\n\n"
    )

    # No usefulness claim by default. The v0.1.0 cards said the backbone was
    # "usable for transfer learning" with no measurement behind it, and
    # snowGradient's 2026-10 probes then found a randomly initialised copy of
    # the network beats it. A claim belongs in --intended-use / --quality-evidence
    # only when a measurement backs it.
    default_intended = (
        f"Research artifact. The discriminator's `features` layer (tap via "
        f"`model.get_layer(\"features\").output`) and the {modality} generator are "
        f"published so results can be reproduced and tested. This release does not "
        f"measure whether the features help any downstream task: compare against a "
        f"randomly initialised copy of the same architecture before relying on them."
    )
    intended_block = intended_use.strip() or default_intended

    default_limitations = (
        f"- Trained exclusively on the `rmdig/rocky_mountain_snowpack` dataset; "
        f"generalization to other snow image distributions is untested.\n"
        f"- Single-modality backbone (`{modality}`); paired-modality features must be "
        f"composed on the consumer side (see [AvAI's `load_backbones`](https://github.com/dennys246/AvAI))."
    )
    limitations_block = limitations.strip() or default_limitations

    notes_block = f"\n## Release notes\n\n{notes}\n" if notes.strip() else ""

    quality_block = quality_evidence.strip() or (
        "None recorded for this release. Training settings (spectral norm, EMA, FID "
        "interval, ...) are not evidence of quality. The in-training FID uses 64 "
        "samples against 2048-dimensional features and is not a valid estimate."
    )

    # Whether the split pools were held out from GAN training. Before
    # honor_splits existed, batch() ignored the pools and the GAN trained on
    # its own validation and test groups (UPGRADES #54) -- the v0.1.0 cards said
    # the opposite. Absent key means the pre-fix behaviour.
    if disc_cfg.get("honor_splits") is True:
        splits_intro = (
            "Splits are deterministic at the `(site, column, core)` group level (seed=42), "
            "persisted in both sidecar configs, and the validation and test pools were "
            "excluded from GAN training (`honor_splits=True`):"
        )
    else:
        splits_intro = (
            "Splits are persisted in both sidecar configs at the `(site, column, core)` "
            "group level, but **they were not held out from this model**: it was trained "
            "without `honor_splits`, so the GAN saw its validation and test groups too "
            "([UPGRADES #54](https://github.com/dennys246/snowGAN/blob/main/docs/UPGRADES.md)):"
        )

    return f"""{frontmatter}# snowGAN — {modality} backbone ({tag})

WGAN-GP trained on the
[Rocky Mountain Snowpack dataset](https://huggingface.co/datasets/{gen_cfg.get('dataset', 'rmdig/rocky_mountain_snowpack')}),
single-modality (`{modality}`), depth={depth}, {res_str} resolution.
Published from the [snowGAN](https://github.com/dennys246/snowGAN) project as a
research artifact.

> **Not for safety decisions.** Nothing here predicts avalanche danger, slope
> stability or snowpack breakability. Do not use it for any backcountry decision.

## How to use

The canonical consumer-side path uses [`snowgan.weights.fetch`](https://github.com/dennys246/snowGAN/blob/main/src/snowgan/weights.py)
to download + cache, then `snowgan.models.Discriminator` to rebuild and load:

```python
from snowgan.weights import fetch
from snowgan.config import build
from snowgan.models.discriminator import Discriminator

# Fetch the release (cached locally after first call).
path = fetch("{repo}", "{tag}")

# Reconstruct the model from its sidecar config, then load weights.
cfg = build(str(path / "discriminator_config.json"))
disc = Discriminator(cfg)
disc.model.build((None, cfg.depth, cfg.resolution[0], cfg.resolution[1], cfg.channels))
disc.model.load_weights(str(path / "discriminator.weights.h5"))

# Tap the named features layer — the cross-repo contract with downstream consumers.
features = disc.model.get_layer("features")
print("backbone features:", features.output.shape)  # (None, {feature_dim})
```

Requires `pip install snowgan[hub]` (pulls in `huggingface_hub`). Without the
`[hub]` extra, `fetch()` raises a clean ImportError naming the missing dep.

## Intended use

{intended_block}

## Architecture

| Field | Value |
| --- | --- |
| Modality | `{modality}` (depth={depth}) |
| Resolution | {res_str} |
| Channels | {gen_cfg.get('channels', 3)} |
| Latent dim | {gen_cfg.get('latent_dim', 100)} |
| Generator filter counts | `{gen_cfg.get('filter_counts')}` |
| Discriminator filter counts | `{filter_counts}` |
| Conv kernel / stride | `{gen_cfg.get('kernel_size')}` / `{gen_cfg.get('kernel_stride')}` |
| Backbone (Flatten "features") dim | {feature_dim} |
| Final activation | `{gen_cfg.get('final_activation', 'tanh')}` |

The discriminator's `Conv3D` layers use `ksize=(1, kH, kW)`, so the depth axis is
broadcast — kernels themselves are depth-agnostic. This is the contract that lets
downstream consumers compose multiple single-modality backbones into a depth=N model
(e.g. profile + core stacked at depth=2 for paired-modality transfer).

## Training

Trained with WGAN-GP loss (Wasserstein + gradient penalty, λ_gp={disc_cfg.get('lambda_gp')})
on the {modality} samples of [`{gen_cfg.get('dataset', 'rmdig/rocky_mountain_snowpack')}`](https://huggingface.co/datasets/{gen_cfg.get('dataset', 'rmdig/rocky_mountain_snowpack')}).
At release: `fade_step={fade_step}`.

| Stabilizer | Setting |
| --- | --- |
| Spectral norm | `{disc_cfg.get('spectral_norm')}` |
| Differentiable augmentation | `{disc_cfg.get('augment')}` |
| Adaptive augmentation (ADA) target | `{gen_cfg.get('ada_target')}` |
| Adaptive disc/gen step ratio | `{gen_cfg.get('adaptive_steps')}` |
| EMA decay (generator shadow) | `{gen_cfg.get('ema_decay')}` |
| Multi-scale discriminator | `{disc_cfg.get('multiscale_disc')}` |
| Gradient clip (global norm) | `{gen_cfg.get('grad_clip_norm')}` |
| LR decay schedule | `{gen_cfg.get('lr_decay')}` (lr_min=`{gen_cfg.get('lr_min')}`) |
| FID eval interval | `{gen_cfg.get('fid_interval')}` steps |

### Dataset splits

{splits_intro}

- `trained_pool`: {trained_count} groups
- `validation_pool`: {val_count} groups
- `test_pool`: {test_count} groups

## Quality evidence

{quality_block}

## Limitations

{limitations_block}

## Files in this release

- `discriminator.weights.h5` — main discriminator weights.
- `discriminator_config.json` — architecture sidecar; pass to `snowgan.models.Discriminator(cfg)`.
- `generator.weights.h5` + `generator_config.json` — generator weights and sidecar.
- `generator_ema.weights.h5` — EMA shadow weights (only if EMA was enabled during training).
- `generator_fade_endpoints.weights.h5` — progressive-fade toRGB endpoints (only if fade was used).
- `discriminator_lowres.weights.h5` — multi-scale 256×256 critic (only if multiscale_disc was on).
- `MANIFEST.md` — full provenance dump (git SHA, every training flag, every artifact). Read this for debugging.
- `README.md` — this file.

## License

Apache 2.0 — see the [snowGAN repository](https://github.com/dennys246/snowGAN) for the full license text.

## Cross-references

- **Source code**: [github.com/dennys246/snowGAN](https://github.com/dennys246/snowGAN)
- **Downstream project**: [github.com/dennys246/AvAI](https://github.com/dennys246/AvAI)
- **Companion modality backbones**:
  [`RMDig/snowGAN-magnified-profile`](https://huggingface.co/RMDig/snowGAN-magnified-profile),
  [`RMDig/snowGAN-core`](https://huggingface.co/RMDig/snowGAN-core)
- **Training dataset**: [`{gen_cfg.get('dataset', 'rmdig/rocky_mountain_snowpack')}`](https://huggingface.co/datasets/{gen_cfg.get('dataset', 'rmdig/rocky_mountain_snowpack')})
{notes_block}"""


def _tag_exists(api, repo: str, tag: str) -> bool:
    """True if ``repo`` already carries ``tag``. A repo that does not exist yet has no tags."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        refs = api.list_repo_refs(repo_id=repo, repo_type="model")
    except RepositoryNotFoundError:
        return False
    return any(ref.name == tag for ref in refs.tags)


def _validate_required(save_dir: Path) -> list[str]:
    return [a for a in REQUIRED_ARTIFACTS if not (save_dir / a).exists()]


def _list_release_files(save_dir: Path) -> list[str]:
    files = list(REQUIRED_ARTIFACTS) + [MANIFEST_FILENAME, MODEL_CARD_FILENAME]
    files.extend(a for a in OPTIONAL_ARTIFACTS if (save_dir / a).exists())
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--save-dir", required=True,
                        help="Local training save_dir containing the artifacts.")
    parser.add_argument("--repo", required=True,
                        help="HF Hub model repo id (e.g. RMDig/snowGAN-core).")
    parser.add_argument("--tag", required=True,
                        help="Release tag (e.g. v0.1.0). Becomes a git tag on the HF repo.")
    parser.add_argument("--notes", default="",
                        help="Free-text notes appended to MANIFEST.md and the model card's "
                             "Release notes section. Use for release-specific commentary "
                             "(e.g. \"trained to fade_step 428k with all advanced flags\").")
    parser.add_argument("--intended-use", default="",
                        help="Override the model card's Intended use section. "
                             "If omitted, a sensible default referencing transfer learning is generated.")
    parser.add_argument("--limitations", default="",
                        help="Override the model card's Limitations section. "
                             "If omitted, a generic per-modality default is generated. Multi-line markdown OK.")
    parser.add_argument("--quality-evidence", default="",
                        help="Markdown for the model card's Quality evidence section: measured results "
                             "only (KID with its real-vs-real floor, memorisation check, probes). If "
                             "omitted, the card states that none was recorded.")
    parser.add_argument("--create-repo", action="store_true",
                        help="Create the model repo on HF Hub if it doesn't exist.")
    parser.add_argument("--private", action="store_true",
                        help="If creating the repo, make it private.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the manifest + model card and list files; do not upload.")
    args = parser.parse_args(argv)

    save_dir = Path(args.save_dir).resolve()
    if not save_dir.is_dir():
        print(f"error: {save_dir} does not exist or is not a directory", file=sys.stderr)
        return 1

    missing = _validate_required(save_dir)
    if missing:
        print(f"error: missing required artifacts in {save_dir}:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    # Resolve the hub client and refuse an existing tag BEFORE writing
    # MANIFEST.md / README.md into save_dir, so a refused release leaves
    # nothing modified.
    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print(
                "error: huggingface_hub not installed. "
                "Install with `pip install snowgan[hub]` or `pip install huggingface_hub`.",
                file=sys.stderr,
            )
            return 1
        api = HfApi()
        if _tag_exists(api, args.repo, args.tag):
            print(
                f"error: {args.repo} already has tag {args.tag}. Released tags are "
                f"immutable: consumers pin them, and moving one leaves two different "
                f"snapshots under one name (v0.1.0 was moved this way on 2026-06-03). "
                f"Release under a new tag instead.",
                file=sys.stderr,
            )
            return 1

    manifest_body = build_manifest(save_dir, args.tag, args.notes).replace(
        "{repo}", args.repo
    )
    manifest_path = save_dir / MANIFEST_FILENAME
    # Force UTF-8 on Windows (default cp1252 chokes on non-ASCII like `λ` in the
    # model card's lambda_gp display). HF Hub reads files as UTF-8.
    manifest_path.write_text(manifest_body, encoding="utf-8")
    print(f"[release] wrote {manifest_path}")

    model_card_body = build_model_card(
        save_dir, args.tag, args.repo,
        intended_use=args.intended_use,
        limitations=args.limitations,
        notes=args.notes,
        quality_evidence=args.quality_evidence,
    )
    model_card_path = save_dir / MODEL_CARD_FILENAME
    model_card_path.write_text(model_card_body, encoding="utf-8")
    print(f"[release] wrote {model_card_path}")

    files_to_upload = _list_release_files(save_dir)
    total_mb = sum((save_dir / f).stat().st_size for f in files_to_upload) / 1024 / 1024
    print(f"[release] {args.repo}@{args.tag} -> {len(files_to_upload)} files ({total_mb:.1f} MB):")
    for f in files_to_upload:
        size_mb = (save_dir / f).stat().st_size / 1024 / 1024
        print(f"  - {f}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("[release] dry-run: not uploading.")
        return 0

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo, repo_type="model",
            exist_ok=True, private=args.private,
        )
        print(f"[release] ensured repo {args.repo} exists (private={args.private})")

    # Upload as a single commit on the main branch, then create the tag.
    # allow_patterns is the explicit whitelist — anything else in save_dir
    # (batch_* snapshots, loss txts, history.png) is intentionally NOT shipped.
    api.upload_folder(
        folder_path=str(save_dir),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=list(files_to_upload),
        commit_message=f"Release {args.tag}",
    )
    api.create_tag(
        repo_id=args.repo, tag=args.tag, repo_type="model",
        tag_message=f"snowGAN release {args.tag}",
    )
    print(f"[release] tagged {args.repo}@{args.tag} on HF Hub")
    print(f"[release] consumers can now: snowgan.weights.fetch('{args.repo}', '{args.tag}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
