"""Released v0.1.0 sidecars must resolve to the architecture their weights have.

The v0.1.0 releases were cut at fde5671, before the generator's architecture
fields existed. Under v1.0.0 defaults their sidecars resolved to
resize-upsampling with two convs per resolution, and the released weights failed
to load ("11 objects could not be loaded"); snowGradient had to hand-pass
gen_upsampler="transpose", gen_convs_per_resolution=1. The fixture is the
released magnified-profile generator_config.json, byte-for-byte (HF
RMDig/snowGAN-magnified-profile @ v0.1.0 = dc935c1).

Verified once against the real weights on 2026-10-02 (both releases, both
models): LOADED with this resolution, FAILED without it.
"""

import json
from pathlib import Path

import pytest

from snowgan.config import build, config_template
from snowgan.models.generator import Generator

FIXTURE = Path(__file__).parent.parent / "fixtures" / "released_v0.1.0_magnified_profile_generator_config.json"


def _released(tmp_path, **overrides):
    data = json.loads(FIXTURE.read_text())
    data.update(overrides)
    path = tmp_path / "generator_config.json"
    path.write_text(json.dumps(data))
    return path


def test_fixture_is_an_era_1_sidecar():
    data = json.loads(FIXTURE.read_text())
    for key in ("gen_norm", "gen_upsampler", "gen_convs_per_resolution"):
        assert key not in data


def test_released_sidecar_resolves_to_the_fde5671_generator(tmp_path):
    cfg = build(str(_released(tmp_path)))
    assert cfg.gen_upsampler == "transpose"
    assert cfg.gen_convs_per_resolution == 1
    assert cfg.gen_norm == "none"  # derived from batch_norm=False, as before


def test_released_sidecar_builds_one_transpose_per_resolution(tmp_path):
    # Architecture is what the era rule decides; shrink only the sizes so the
    # build is cheap. The rule does not depend on them.
    cfg = build(str(_released(tmp_path, filter_counts=[8, 4], latent_dim=8)))
    model = Generator(cfg).model
    kinds = [type(l).__name__ for l in model.layers]
    assert kinds.count("Conv3DTranspose") == len(cfg.filter_counts) + 1  # blocks + toRGB
    assert "UpSampling3D" not in kinds
    assert "Conv3D" not in kinds
    # Output size is 16 * 2**(len(filter_counts) + 1); `resolution` is not read.
    assert model.output_shape == (None, 1, 128, 128, 3)


def test_later_eras_keep_their_own_defaults(tmp_path):
    # b4e0677..1a925d4 wrote gen_norm but not gen_upsampler, and that generator
    # really was resize + 2 convs. The rule must not touch it.
    cfg = build(str(_released(tmp_path, gen_norm="pixel")))
    assert (cfg.gen_upsampler, cfg.gen_convs_per_resolution) == ("resize", 2)


def test_explicit_fields_win(tmp_path):
    path = tmp_path / "generator_config.json"
    data = json.loads(FIXTURE.read_text())
    data.update(gen_upsampler="resize", gen_convs_per_resolution=2)
    path.write_text(json.dumps(data))
    cfg = build(str(path))
    assert (cfg.gen_upsampler, cfg.gen_convs_per_resolution) == ("resize", 2)


def test_template_path_is_untouched(tmp_path):
    cfg = build(str(tmp_path / "absent.json"))
    assert cfg.gen_upsampler == config_template["gen_upsampler"]
    assert cfg.gen_convs_per_resolution == config_template["gen_convs_per_resolution"]


def test_resolution_is_written_back_so_it_is_not_era_dependent_again(tmp_path):
    path = _released(tmp_path)
    cfg = build(str(path))
    cfg.save_config(str(path))
    data = json.loads(path.read_text())
    assert (data["gen_norm"], data["gen_upsampler"], data["gen_convs_per_resolution"]) == ("none", "transpose", 1)


@pytest.mark.parametrize("modality", ["core", "magnified_profile", "merged"])
def test_datatype_mirrors_modality(tmp_path, modality):
    # The core release's sidecar said datatype=magnified_profile beside
    # modality=core: datatype was never set by anything.
    cfg = build(str(_released(tmp_path, modality=modality, datatype="magnified_profile")))
    assert cfg.datatype == modality
    assert cfg.dump()["datatype"] == modality


def test_datatype_follows_modality_set_after_configure(tmp_path):
    # configure_gen/configure_disc set modality from --modality after configure();
    # a fresh run's first save must not write the template's datatype.
    path = tmp_path / "generator_config.json"
    cfg = build(str(path))
    cfg.modality = "core"
    cfg.save_config(str(path))
    assert json.loads(path.read_text())["datatype"] == "core"


def test_legacy_resolution_is_announced(tmp_path, capsys):
    # It runs on trainer resumes too; a silent architecture change is a confound.
    build(str(_released(tmp_path)))
    assert "Legacy sidecar" in capsys.readouterr().out


def test_current_sidecars_are_not_announced(tmp_path, capsys):
    build(str(_released(tmp_path, gen_norm="pixel")))
    assert "Legacy sidecar" not in capsys.readouterr().out
