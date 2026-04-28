# SPDX-FileCopyrightText: © 2026 ACID Contributors <https://doi.org/10.5281/zenodo.15722902>
# SPDX-License-Identifier: CC-BY-SA-4.0 OR LGPL-3.0-or-later

import io
import json
import zipfile

import numpy as np
from numpy.typing import NDArray


def dump_npy(name: str, zf: zipfile.ZipFile, array: NDArray):
    """Dump a NumPy array to a ZIP file as a .npy file."""
    zi = default_zipinfo(name)
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    zf.writestr(zi, buf.getvalue())


def default_zipinfo(name: str) -> zipfile.ZipInfo:
    """Create a ZipInfo object with a fixed date."""
    zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    zi.external_attr = 0
    zi.create_system = 0
    return zi


def dump_meta(name: str, zf: zipfile.ZipFile, data):
    """Dump metadata to a ZIP file as a .json file."""
    zi = default_zipinfo(name)
    zf.writestr(zi, json.dumps(data))


def lookup_integer(sequence: NDArray[float], std: float, table: NDArray[float]) -> NDArray[int]:
    r"""Lookup to which integer the floats should be mapped according to the lookup table.
    This lookup table is based on the mapping of the sequence to a cumulative distribution function,
    belonging to a Gaussian distribtion with standard deviation ``std``.

    Parameters
    ----------
    sequence
        The input sequences, which is an array with shape ``(nindep, nstep)``.
        Each row is a time-dependent sequence.
    std
        The standard deviation of the sequence.
    table
        The lookup table to map the floats to integers.

    Returns
    -------
    An array that contains the original floats mapped to integers

    """
    return np.searchsorted(table, sequence / std)


def compute_amplitudes(sequences: NDArray[float], timestep: float = 1.0) -> NDArray[float]:
    r"""Compute the amplitudes of a batch of sequences as follows:

    .. math::

    C_k = \frac{1}{M}\sum_{m=1}^M \frac{h}{N} \left|
        \sum_{n=0}^{N-1} x^{(m)}_n \exp\left(-i \frac{2 \pi n k}{N}\right)
    \right|^2

    where:

    - :math:`h` is the timestep,
    - :math:`N` is the number of time steps in the input sequences,
    - :math:`M` is the number of independent sequences,
    - :math:`x^{(m)}_n` is the value of the :math:`m`-th sequence at time step :math:`n`,
    - :math:`k` is the frequency index.

    This normalization differs from conventional discrete Fourier transforms,
    where the factor :math:`\frac{1}{N}` is typically applied in the inverse transform.
    Applying the normalization directly in the forward transform ensures that the resulting spectrum
    is an intensive property,
    which is important in the context of transport properties,
    where the zero-frequency limit is the quantity of interest.
    Likewise, the factor :math:`\frac{1}{M}` ensures that the averaged spectrum is also intensive
    with respect to the number of independent sequences :math:`M`.

    Parameters
    ----------
    sequences
        The input sequences, which is an array with shape ``(nindep, nstep)``.
        Each row is a time-dependent sequence.
    timestep
        The time step of the input sequence.

    Returns
    -------
    amplitudes
        A numpy array that contains the amplitudes of the spectrum.
    """
    nindep = sequences.shape[0]
    nstep = sequences.shape[1]
    return timestep * (abs(np.fft.rfft(sequences, axis=1)) ** 2).sum(axis=0) / (nstep * nindep)
