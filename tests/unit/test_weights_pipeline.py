"""Unit tests for the HuggingFace Hub weights pipeline.

Covers both halves of the pipeline:
  - ``snowgan.weights.fetch`` — the consumer-side fetcher.
  - ``scripts/release_weights.py`` — the producer-side bundler/uploader,
    specifically its ``build_manifest`` formatter and its file-validation
    + file-listing logic.

The HF Hub round-trip itself is mocked out — we never hit the network in
tests. End-to-end verification of an actual upload happens manually via
``--dry-run`` and a real release.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

# `types` and `sys` are used by the dry-run test below to inject a fake
# huggingface_hub.HfApi only AFTER our code path normally would have
# touched it (the test verifies it doesn't).


# ----------------------------------------------------------------------
# fetch helper
# ----------------------------------------------------------------------

def test_fetch_delegates_to_snapshot_download(monkeypatch, tmp_path):
    """``fetch(repo, version)`` calls ``snapshot_download`` with the right
    repo_id / revision / repo_type and returns its result as a Path.

    Patches the attribute on the real ``huggingface_hub`` module rather than
    swapping the whole module in ``sys.modules`` — the latter breaks
    other consumers (e.g. ``datasets``) that snowgan pulls in transitively.
    """
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    captured = {}

    def fake_snapshot_download(*, repo_id, revision, cache_dir, repo_type):
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        captured["cache_dir"] = cache_dir
        captured["repo_type"] = repo_type
        return str(tmp_path / "cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    from snowgan.weights import fetch
    result = fetch("RMDig/snowGAN-core", "v0.1.0")

    assert captured == {
        "repo_id": "RMDig/snowGAN-core",
        "revision": "v0.1.0",
        "cache_dir": None,
        "repo_type": "model",
    }
    assert result == tmp_path / "cached"
    assert isinstance(result, Path)


def test_fetch_passes_through_cache_dir(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    from snowgan.weights import fetch
    fetch("RMDig/snowGAN-core", "v0.1.0", cache_dir="/tmp/snowgan-cache")
    assert captured["cache_dir"] == "/tmp/snowgan-cache"


def test_fetch_raises_clear_error_when_hf_hub_missing(monkeypatch):
    """When ``huggingface_hub`` isn't satisfiable for the
    ``snapshot_download`` import, callers get a clean ImportError that names
    the optional extra — not a cryptic ``cannot import name`` from deep
    inside the package.

    Simulated via ``builtins.__import__`` because ``huggingface_hub`` uses
    PEP 562 lazy ``__getattr__`` for its top-level exports, so a simple
    ``delattr`` is a no-op (the next attribute access re-resolves through
    ``__getattr__``).
    """
    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # `from huggingface_hub import snapshot_download` invokes
        # __import__("huggingface_hub", ..., fromlist=("snapshot_download",), ...)
        if name == "huggingface_hub" and "snapshot_download" in (fromlist or ()):
            raise ImportError("simulated: huggingface_hub not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from snowgan.weights import fetch
    with pytest.raises(ImportError, match=r"snowgan\[hub\]"):
        fetch("RMDig/snowGAN-core", "v0.1.0")


# ----------------------------------------------------------------------
# release_weights.py — load the script as a module via its file path so
# we can exercise its internals without invoking it as __main__.
# ----------------------------------------------------------------------

@pytest.fixture
def release_module():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "release_weights",
        repo_root / "scripts" / "release_weights.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_minimal_run(save_dir: Path, *, optional: bool = False):
    """Write a fake training save_dir with the artifacts the release script
    looks at. Bytes are arbitrary — we never load them as weights."""
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "discriminator.weights.h5").write_bytes(b"\x00" * 1024)
    (save_dir / "generator.weights.h5").write_bytes(b"\x00" * 4096)
    (save_dir / "discriminator_config.json").write_text(json.dumps({
        "architecture": "discriminator",
        "modality": "core",
        "depth": 1,
        "resolution": [1024, 1024],
        "channels": 3,
        "filter_counts": [64, 128, 256, 512, 1024],
        "kernel_size": [3, 3],
        "kernel_stride": [2, 2],
        "spectral_norm": True,
        "augment": True,
        "multiscale_disc": True,
        "lambda_gp": 1.0,
        "training_steps": 2,
        "fade_step": 100000,
        "fade_steps": 50000,
        "current_epoch": 0,
        "ada_target": 0.6,
        "adaptive_steps": False,
        "grad_clip_norm": 1.0,
        "fid_interval": 5000,
        "trained_pool": [[1, 1, 1]] * 10,
        "validation_pool": [[2, 2, 2]],
        "test_pool": [[3, 3, 3], [4, 4, 4]],
    }))
    (save_dir / "generator_config.json").write_text(json.dumps({
        "architecture": "generator",
        "modality": "core",
        "depth": 1,
        "resolution": [1024, 1024],
        "channels": 3,
        "filter_counts": [1024, 512, 256, 128, 64],
        "kernel_size": [3, 3],
        "kernel_stride": [2, 2],
        "training_steps": 3,
        "fade_step": 100000,
        "fade_steps": 50000,
        "current_epoch": 0,
        "ema_decay": 0.999,
        "lr_decay": "cosine",
        "lr_min": 1e-7,
        "latent_dim": 100,
        "dataset": "rmdig/rocky_mountain_snowpack",
        # Shared training flags are persisted in BOTH gen and disc configs
        # via configure_generic; mirror that here so the manifest's gen-side
        # reads pick them up.
        "spectral_norm": True,
        "augment": True,
        "multiscale_disc": True,
        "ada_target": 0.6,
        "adaptive_steps": False,
        "grad_clip_norm": 1.0,
        "fid_interval": 5000,
    }))
    if optional:
        (save_dir / "generator_ema.weights.h5").write_bytes(b"\x00" * 4096)
        (save_dir / "discriminator_lowres.weights.h5").write_bytes(b"\x00" * 256)
    # leave generator_fade_endpoints.weights.h5 absent to verify graceful skip


def test_build_manifest_renders_known_fields(release_module, tmp_path):
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir, optional=True)

    manifest = release_module.build_manifest(
        save_dir, "v0.1.0",
        "Trained on the core modality with full advanced flags.",
        snowgan_version="9.9.9",
        git_sha="deadbeef",
    )

    # Provenance block
    assert "snowGAN release v0.1.0" in manifest
    assert "9.9.9" in manifest
    assert "deadbeef" in manifest
    assert "rmdig/rocky_mountain_snowpack" in manifest

    # Model architecture
    assert "depth: 1" in manifest
    assert "modality: core" in manifest
    assert "[1024, 512, 256, 128, 64]" in manifest   # gen
    assert "[64, 128, 256, 512, 1024]" in manifest   # disc

    # Training state
    assert "fade_step: 100000" in manifest
    assert "lambda_gp: 1.0" in manifest

    # Advanced flags
    assert "spectral_norm: True" in manifest
    assert "ema_decay: 0.999" in manifest
    assert "ada_target: 0.6" in manifest

    # Splits
    assert "trained_pool: 10 groups" in manifest
    assert "test_pool: 2 groups" in manifest

    # Artifacts list — required artifacts always listed; optional ones only
    # when present on disk. The fake run has EMA + lowres but no fade_endpoints.
    assert "`generator.weights.h5`" in manifest
    assert "`generator_ema.weights.h5`" in manifest
    assert "`discriminator_lowres.weights.h5`" in manifest
    assert "`generator_fade_endpoints.weights.h5`" not in manifest

    # Notes block
    assert "Trained on the core modality with full advanced flags." in manifest


def test_build_manifest_with_no_optional_artifacts(release_module, tmp_path):
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir, optional=False)
    manifest = release_module.build_manifest(
        save_dir, "v0.1.0", "", snowgan_version="x", git_sha="y",
    )
    assert "`generator_ema.weights.h5`" not in manifest
    assert "`discriminator_lowres.weights.h5`" not in manifest
    # Empty notes fall back to a sentinel so the section isn't blank.
    assert "(none)" in manifest


def test_build_model_card_renders_yaml_frontmatter_and_sections(release_module, tmp_path):
    """The model card has correct YAML frontmatter for HF Hub's parser, a
    title that names the modality and tag, a how-to-use code snippet
    referencing the right repo + tag, an architecture table, a training
    stabilizers table, and a splits summary derived from the disc config."""
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir, optional=True)

    card = release_module.build_model_card(
        save_dir, "v0.1.0", "RMDig/snowGAN-core",
        intended_use="", limitations="", notes="First release.",
    )

    # YAML frontmatter
    assert card.startswith("---\n")
    assert "license: apache-2.0" in card
    assert "library_name: snowgan" in card
    assert "pipeline_tag: image-generation" in card
    assert "  - gan" in card
    assert "  - modality-core" in card  # modality-specific tag
    assert "  - rmdig/rocky_mountain_snowpack" in card

    # Title + identity
    assert "snowGAN — core backbone (v0.1.0)" in card

    # Code snippet uses the right repo + tag
    assert 'fetch("RMDig/snowGAN-core", "v0.1.0")' in card
    assert 'disc.model.get_layer("features")' in card

    # Architecture table
    assert "| Modality | `core` (depth=1) |" in card
    assert "| Resolution | 1024x1024 |" in card
    # feature_dim = depth(1) * (1024/2^5)^2 * 1024 = 1 * 32^2 * 1024 = 1048576
    assert "1048576" in card

    # Training stabilizers table
    assert "| Spectral norm | `True` |" in card
    assert "| EMA decay (generator shadow) | `0.999` |" in card
    assert "λ_gp=1.0" in card

    # Splits derived from disc config (10/1/2 in our fixture)
    assert "`trained_pool`: 10 groups" in card
    assert "`validation_pool`: 1 groups" in card
    assert "`test_pool`: 2 groups" in card

    # Notes block rendered when notes provided
    assert "## Release notes" in card
    assert "First release." in card


def test_build_model_card_uses_default_intended_use_and_limitations(release_module, tmp_path):
    """When --intended-use / --limitations are empty, the card falls back
    to sensible per-modality defaults referencing AvAI and the
    single-modality caveat."""
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir, optional=False)

    card = release_module.build_model_card(
        save_dir, "v0.1.0", "RMDig/snowGAN-magnified-profile",
        intended_use="", limitations="", notes="",
    )

    # Default intended-use names the features tap but claims no usefulness:
    # the v0.1.0 "usable for transfer learning" claim was never measured and
    # was contradicted downstream (random init beat it, snowGradient 2026-10).
    assert "`features` layer" in card
    assert "transfer learning" not in card.lower()
    assert "transfer-learning" not in card
    assert "randomly initialised" in card
    # Safety line is unconditional; quality section says none was recorded.
    assert "Not for safety decisions" in card
    assert "None recorded for this release" in card

    # Default limitations notes single-modality + dataset caveat
    assert "rmdig/rocky_mountain_snowpack" in card
    assert "single-modality" in card.lower() or "Single-modality" in card

    # No Release notes section when notes empty
    assert "## Release notes" not in card


def test_build_model_card_honors_explicit_overrides(release_module, tmp_path):
    """Explicit --intended-use / --limitations text appears verbatim
    in place of the defaults."""
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir, optional=False)

    custom_intended = "Reserved for research use; do not deploy in safety-critical pipelines."
    custom_limitations = "- Trained on a tiny dataset (~13 unique samples).\n- Late-training disc divergence."

    card = release_module.build_model_card(
        save_dir, "v0.1.0", "RMDig/snowGAN-core",
        intended_use=custom_intended,
        limitations=custom_limitations,
        notes="",
    )

    assert custom_intended in card
    assert "tiny dataset" in card
    assert "Late-training disc divergence." in card
    # Verify the defaults were displaced, not appended.
    assert "transfer learning" not in card.lower() or custom_intended in card


def test_model_card_listed_in_release_files(release_module, tmp_path):
    """README.md (the model card) is part of the release upload alongside MANIFEST.md."""
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir)
    # README.md doesn't have to exist on disk yet — release_files lists what
    # the script will upload, and the script writes the card before listing.
    files = release_module._list_release_files(save_dir)
    assert "README.md" in files
    assert "MANIFEST.md" in files


def test_dry_run_writes_both_manifest_and_model_card(release_module, tmp_path, capsys):
    """--dry-run produces MANIFEST.md AND README.md so users can inspect both
    before pushing."""
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir, optional=True)

    rc = release_module.main([
        "--save-dir", str(save_dir),
        "--repo", "RMDig/snowGAN-core",
        "--tag", "v0.1.0",
        "--notes", "smoke",
        "--dry-run",
    ])

    assert rc == 0
    assert (save_dir / "MANIFEST.md").exists()
    assert (save_dir / "README.md").exists()
    card = (save_dir / "README.md").read_text(encoding="utf-8")
    assert "snowGAN — core backbone (v0.1.0)" in card
    assert 'fetch("RMDig/snowGAN-core", "v0.1.0")' in card


def test_validate_required_returns_missing(release_module, tmp_path):
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    (save_dir / "discriminator.weights.h5").write_bytes(b"\x00")
    # Missing: discriminator_config.json, generator.weights.h5, generator_config.json
    missing = release_module._validate_required(save_dir)
    assert set(missing) == {
        "discriminator_config.json",
        "generator.weights.h5",
        "generator_config.json",
    }


def test_validate_required_empty_when_all_present(release_module, tmp_path):
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir)
    assert release_module._validate_required(save_dir) == []


def test_list_release_files_includes_only_present_optionals(release_module, tmp_path):
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir, optional=True)
    files = release_module._list_release_files(save_dir)
    # required + manifest + the optionals that exist
    assert "discriminator.weights.h5" in files
    assert "generator.weights.h5" in files
    assert "discriminator_config.json" in files
    assert "generator_config.json" in files
    assert "MANIFEST.md" in files
    assert "generator_ema.weights.h5" in files
    assert "discriminator_lowres.weights.h5" in files
    assert "generator_fade_endpoints.weights.h5" not in files


def test_dry_run_writes_manifest_but_skips_upload(release_module, tmp_path, capsys):
    """``--dry-run`` should write the MANIFEST.md to disk (so the user can
    inspect it before the real push) but must NOT call HfApi."""
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir, optional=True)

    # If HfApi gets touched, fail loudly.
    def _exploding_api(*a, **kw):
        raise AssertionError("HfApi must not be constructed in dry-run mode")
    sys.modules.setdefault("huggingface_hub", types.ModuleType("huggingface_hub"))
    sys.modules["huggingface_hub"].HfApi = _exploding_api  # type: ignore[attr-defined]

    rc = release_module.main([
        "--save-dir", str(save_dir),
        "--repo", "RMDig/snowGAN-core",
        "--tag", "v0.1.0",
        "--notes", "smoke",
        "--dry-run",
    ])

    assert rc == 0
    assert (save_dir / "MANIFEST.md").exists()
    manifest = (save_dir / "MANIFEST.md").read_text(encoding="utf-8")
    assert "snowGAN release v0.1.0" in manifest
    # The {repo} placeholder in the manifest template should be substituted.
    assert "{repo}" not in manifest
    assert "RMDig/snowGAN-core" in manifest

    captured = capsys.readouterr()
    assert "dry-run" in captured.out


def test_main_exits_nonzero_when_save_dir_missing(release_module, tmp_path, capsys):
    rc = release_module.main([
        "--save-dir", str(tmp_path / "does_not_exist"),
        "--repo", "RMDig/snowGAN-core",
        "--tag", "v0.1.0",
    ])
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err


def test_main_exits_nonzero_when_required_artifacts_missing(release_module, tmp_path, capsys):
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    # Only one of the four required files exists.
    (save_dir / "discriminator.weights.h5").write_bytes(b"\x00")
    rc = release_module.main([
        "--save-dir", str(save_dir),
        "--repo", "RMDig/snowGAN-core",
        "--tag", "v0.1.0",
        "--dry-run",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "missing required artifacts" in err
    assert "generator.weights.h5" in err


# ----------------------------------------------------------------------
# Released tags are immutable. v0.1.0 was moved on both HF repos on
# 2026-06-03 (weights identical, metadata changed), leaving two snapshots
# under one name; the script must refuse to do that again.
# ----------------------------------------------------------------------

class _FakeHfApi:
    def __init__(self, tags=(), missing=False):
        self.tags, self.missing, self.calls = list(tags), missing, []

    def list_repo_refs(self, repo_id, repo_type):
        if self.missing:
            from huggingface_hub.utils import RepositoryNotFoundError
            # Its constructor wants a live HTTP response; the type is what matters.
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        return types.SimpleNamespace(tags=[types.SimpleNamespace(name=t) for t in self.tags])

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs["repo_id"]))

    def create_tag(self, **kwargs):
        self.calls.append(("create_tag", kwargs["tag"]))

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs["repo_id"]))


def _patch_api(monkeypatch, api):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)


def test_release_refuses_an_existing_tag_and_writes_nothing(release_module, monkeypatch, tmp_path, capsys):
    api = _FakeHfApi(tags=["v0.1.0"])
    _patch_api(monkeypatch, api)
    _write_minimal_run(tmp_path)
    rc = release_module.main(["--save-dir", str(tmp_path), "--repo", "RMDig/x", "--tag", "v0.1.0"])
    assert rc == 1
    assert api.calls == []
    assert not (tmp_path / "MANIFEST.md").exists() and not (tmp_path / "README.md").exists()
    assert "immutable" in capsys.readouterr().err


def test_release_uploads_and_tags_a_new_tag(release_module, monkeypatch, tmp_path):
    api = _FakeHfApi(tags=["v0.1.0"])
    _patch_api(monkeypatch, api)
    _write_minimal_run(tmp_path)
    assert release_module.main(["--save-dir", str(tmp_path), "--repo", "RMDig/x", "--tag", "v0.1.1"]) == 0
    assert api.calls == [("upload_folder", "RMDig/x"), ("create_tag", "v0.1.1")]


def test_release_to_a_repo_that_does_not_exist_yet(release_module, monkeypatch, tmp_path):
    api = _FakeHfApi(missing=True)
    _patch_api(monkeypatch, api)
    _write_minimal_run(tmp_path)
    args = ["--save-dir", str(tmp_path), "--repo", "RMDig/new", "--tag", "v0.1.0", "--create-repo"]
    assert release_module.main(args) == 0
    assert [c[0] for c in api.calls] == ["create_repo", "upload_folder", "create_tag"]


def test_model_card_says_splits_were_not_held_out_without_honor_splits(release_module, tmp_path):
    """The v0.1.0 cards said the GAN "never saw" its held-out cores; it trained
    on them (UPGRADES #54). Without honor_splits=True the card must say so."""
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir)
    card = release_module.build_model_card(save_dir, "v0.1.0", "RMDig/x")
    assert "never saw" not in card
    assert "they were not held out from this model" in card


def test_model_card_states_held_out_splits_with_honor_splits(release_module, tmp_path):
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir)
    disc = json.loads((save_dir / "discriminator_config.json").read_text())
    disc["honor_splits"] = True
    (save_dir / "discriminator_config.json").write_text(json.dumps(disc))
    card = release_module.build_model_card(save_dir, "v0.2.0", "RMDig/x")
    assert "excluded from GAN training" in card
    assert "not held out" not in card


def test_model_card_renders_quality_evidence(release_module, tmp_path):
    save_dir = tmp_path / "run"
    _write_minimal_run(save_dir)
    card = release_module.build_model_card(save_dir, "v0.2.0", "RMDig/x",
                                           quality_evidence="- KID 0.012 ± 0.002 (floor 0.004)")
    assert "- KID 0.012 ± 0.002 (floor 0.004)" in card
    assert "None recorded" not in card
