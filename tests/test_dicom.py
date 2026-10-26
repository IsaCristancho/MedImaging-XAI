
import numpy as np
import pytest

from medimaging.dicom.secondary_capture import (
    read_dicom_image,
    write_secondary_capture,
)


def test_dicom_round_trip_preserves_pixels_and_metadata(tmp_path):
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    output_path = tmp_path / "synthetic_image.dcm"

    write_secondary_capture(image, output_path)
    restored, dataset = read_dicom_image(output_path)

    np.testing.assert_array_equal(restored, image)
    assert dataset.Rows == 8
    assert dataset.Columns == 8
    assert dataset.Modality == "OT"
    assert dataset.PatientID == "SYNTHETIC001"


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((0, 0), dtype=np.uint16),
        np.zeros((8, 8, 3), dtype=np.uint16),
        np.zeros((8, 8), dtype=np.float32),
        np.full((8, 8), -1, dtype=np.int32),
        np.full((8, 8), 65536, dtype=np.uint32),
    ],
)
def test_dicom_writer_rejects_invalid_images(tmp_path, image):
    with pytest.raises(ValueError):
        write_secondary_capture(image, tmp_path / "invalid.dcm")
