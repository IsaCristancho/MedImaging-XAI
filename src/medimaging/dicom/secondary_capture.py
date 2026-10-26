
"""Utilities for writing and reading synthetic DICOM images."""

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)


def write_secondary_capture(
    image: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write a 2D uint16 image as a DICOM Secondary Capture file."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Image must be a non-empty 2D array.")

    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError("Image must contain integer pixel values.")

    if image.min() < 0 or image.max() > 65535:
        raise ValueError("Pixel values must be in the uint16 range.")

    pixels = image.astype(np.uint16)
    output_path = Path(output_path)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(
        str(output_path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "SYNTHETIC^TEST"
    dataset.PatientID = "SYNTHETIC001"
    dataset.Modality = "OT"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.StudyDate = "20260101"
    dataset.ContentDate = "20260101"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(output_path, enforce_file_format=True)
    return output_path


def read_dicom_image(path: str | Path) -> tuple[np.ndarray, Any]:
    """Read a DICOM image and return its pixel array and dataset."""
    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array
    return pixels, dataset
