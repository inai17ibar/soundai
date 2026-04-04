"""Audio embedding using LAION-CLAP.

CLAP (Contrastive Language-Audio Pretraining) produces embeddings that
align audio and text in a shared latent space, enabling both similarity
search and natural language queries.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

# Suppress noisy warnings from laion_clap on import
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import laion_clap  # type: ignore

SUPPORTED_EXTENSIONS = {".wav", ".aiff", ".aif", ".mp3", ".flac", ".ogg"}

# Music-optimised HTSAT-base checkpoint.
# Downloaded to the laion_clap package directory on first use.
_LAION_CLAP_PKG_DIR = os.path.dirname(os.path.realpath(laion_clap.__file__))
DEFAULT_MODEL = os.path.join(_LAION_CLAP_PKG_DIR, "630k-audioset-best.pt")


class CLAPEmbedder:
    """Wraps LAION-CLAP to produce L2-normalised 512-dim embeddings.

    Parameters
    ----------
    model_name:
        CLAP checkpoint name.  Pass ``None`` to use the default.
    use_cuda:
        Force CUDA even when auto-detection would choose CPU.
    """

    DIM = 512

    def __init__(
        self,
        model_name: str | None = None,
        use_cuda: bool = False,
    ) -> None:
        self._model_name = model_name or DEFAULT_MODEL
        self._model: laion_clap.CLAP_Module | None = None
        self._use_cuda = use_cuda

    # ------------------------------------------------------------------
    # Lazy initialisation – model is ~400 MB so we load on first use
    # ------------------------------------------------------------------
    @property
    def model(self) -> laion_clap.CLAP_Module:
        if self._model is None:
            self._model = laion_clap.CLAP_Module(
                enable_fusion=False,
                amodel="HTSAT-tiny",
            )
            self._model.load_ckpt(self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_files(self, paths: list[Path | str]) -> np.ndarray:
        """Return (N, 512) float32 array of L2-normalised audio embeddings.

        Parameters
        ----------
        paths:
            List of audio file paths.  Unsupported formats are silently
            skipped; the returned array rows correspond to *valid* files
            only.
        """
        valid = [Path(p) for p in paths if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS]
        if not valid:
            return np.empty((0, self.DIM), dtype=np.float32)

        str_paths = [str(p) for p in valid]
        embeddings = self.model.get_audio_embedding_from_filelist(
            x=str_paths, use_tensor=False
        )
        return self._normalise(np.array(embeddings, dtype=np.float32))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return (N, 512) float32 array of L2-normalised text embeddings."""
        embeddings = self.model.get_text_embedding(texts, use_tensor=False)
        return self._normalise(np.array(embeddings, dtype=np.float32))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return x / norms
