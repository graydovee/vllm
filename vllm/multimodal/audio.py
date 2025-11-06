# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from vllm.utils import PlaceholderModule

from .base import MediaIO

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


def resample_audio_librosa(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def resample_audio_scipy(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
):
    # lazy import scipy.signal, otherwise it will crash doc build.
    import scipy.signal

    if orig_sr > target_sr:
        return scipy.signal.resample_poly(audio, 1, orig_sr // target_sr)
    elif orig_sr < target_sr:
        return scipy.signal.resample_poly(audio, target_sr // orig_sr, 1)
    return audio


class AudioResampler:
    """Resample audio data to a target sample rate."""

    def __init__(
        self,
        target_sr: Optional[float] = None,
        method: Literal["librosa", "scipy"] = "librosa",
    ):
        self.target_sr = target_sr
        self.method = method

    def resample(
        self,
        audio: npt.NDArray[np.floating],
        *,
        orig_sr: float,
    ) -> npt.NDArray[np.floating]:
        if self.target_sr is None:
            raise RuntimeError("Audio resampling is not supported when "
                               "`target_sr` is not provided")
        if self.method == "librosa":
            return resample_audio_librosa(audio,
                                          orig_sr=orig_sr,
                                          target_sr=self.target_sr)
        elif self.method == "scipy":
            return resample_audio_scipy(audio,
                                        orig_sr=orig_sr,
                                        target_sr=self.target_sr)
        else:
            raise ValueError(f"Invalid resampling method: {self.method}. "
                             "Supported methods are 'librosa' and 'scipy'.")


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs
        
        # PCM format default parameters
        self.pcm_sample_rate = kwargs.get('pcm_sample_rate', 16000)
        self.pcm_bit_depth = kwargs.get('pcm_bit_depth', 16)
        self.pcm_channels = kwargs.get('pcm_channels', 1)

    def _load_pcm_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        """Load PCM audio data from raw bytes.
        
        PCM format has no header, so we need to specify the format parameters.
        Default: 16kHz, 16-bit, mono
        """
        # Determine numpy dtype based on bit depth
        if self.pcm_bit_depth == 16:
            dtype = np.int16
            max_val = 32768.0
        elif self.pcm_bit_depth == 8:
            dtype = np.int8
            max_val = 128.0
        elif self.pcm_bit_depth == 24:
            dtype = np.int32
            max_val = 8388608.0
        elif self.pcm_bit_depth == 32:
            dtype = np.int32
            max_val = 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {self.pcm_bit_depth}")
        
        # Load PCM data as numpy array
        audio_data = np.frombuffer(data, dtype=dtype)
        
        # Handle multi-channel audio
        if self.pcm_channels > 1:
            # Reshape to (samples, channels) and take mean across channels (convert to mono)
            audio_data = audio_data.reshape(-1, self.pcm_channels)
            audio_data = audio_data.mean(axis=1)
        
        # Normalize to float32 in range [-1.0, 1.0]
        audio_data = audio_data.astype(np.float32) / max_val
        
        return audio_data, float(self.pcm_sample_rate)

    def load_bytes(self, data: bytes, media_type: Optional[str] = None) -> tuple[npt.NDArray, float]:
        # Check if it's PCM format
        if media_type and 'pcm' in media_type.lower():
            return self._load_pcm_bytes(data)
        
        # Try to load with librosa first
        try:
            return librosa.load(BytesIO(data), sr=None)
        except Exception as e:
            # If librosa fails, try PCM format as fallback
            try:
                return self._load_pcm_bytes(data)
            except Exception as pcm_error:
                # If both fail, raise the original error
                raise e

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data), media_type=media_type)

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        # Check if it's a PCM file by extension
        if filepath.suffix.lower() in ['.pcm', '.raw']:
            with open(filepath, 'rb') as f:
                data = f.read()
            return self._load_pcm_bytes(data)
        
        # Otherwise use librosa
        return librosa.load(filepath, sr=None)

    def encode_base64(self, media: tuple[npt.NDArray, float]) -> str:
        audio, sr = media

        with BytesIO() as buffer:
            soundfile.write(buffer, audio, sr, format="WAV")
            data = buffer.getvalue()

        return base64.b64encode(data).decode('utf-8')
