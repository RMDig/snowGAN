"""
snowGAN package
Exposes the main functions for generating and training diffusion models.
"""

# Import functions/classes from internal modules
from snowgan.modality import Modality
from snowgan.models.discriminator import Discriminator, load_discriminator
from snowgan.models.generator import Generator, load_generator

from snowgan.trainer import Trainer as trainer
from snowgan.generate import generate
from snowgan.config import configure_gen, configure_disc
from snowgan.config import build

# Define a clean public API
__all__ = [
    "Modality",
    "Generator",
    "load_generator",
    "Discriminator",
    "load_discriminator",
    "trainer",
    "generate",
    "build",
    "configure_gen",
    "configure_disc"
]

# Version, exposed so a schema-drift failure in a downstream consumer carries a
# signal. Without it, a sidecar written by a newer snowgan surfaces as a bare
# TypeError deep inside the consumer's model loader and reads as their bug.
# Sourced from installed package metadata so it cannot drift from pyproject.
try:  # pragma: no cover - trivial
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("snowgan")
except Exception:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"
