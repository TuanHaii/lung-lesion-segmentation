"""Unit and integration tests for Hardened SafePathResolver V2.

Covers all 14 required validation scenarios:
1. Ancestor directory containing the word 'infection'.
2. CT inside actual CT directory.
3. Mask inside actual mask directory.
4. Different CT and mask filenames (e.g. case009.nii vs coronacases_009.nii.gz).
5. Multiple CT candidates (raises AmbiguityError).
6. Multiple mask candidates (raises AmbiguityError).
7. Missing CT (raises FileNotFoundError).
8. Missing mask (raises FileNotFoundError).
9. Same-file CT-mask resolution (raises PairIntegrityError).
10. Invalid extensions (.dcm, .png, .txt rejected).
11. Zero-byte files rejected.
12. Shape mismatch (raises PairIntegrityError).
13. Affine mismatch (raises PairIntegrityError).
14. Incorrect patient mapping (raises PairIntegrityError).
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib

from src.data.v2.resolver import (
    SafePathResolver,
    PathResolverError,
    AmbiguityError,
    PairIntegrityError,
    is_nii_file,
    parse_patient_id,
)


def create_dummy_nii(
    path: Path,
    shape=(16, 16, 8),
    affine=None,
    data=None,
    is_ct: bool = True,
):
    """Helper to write a tiny valid NIfTI file for geometric testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4)
    if data is None:
        if is_ct:
            # Range -1000 to 400 for realistic CT intensity (non-binary)
            data = np.random.uniform(-1000.0, 400.0, size=shape).astype(np.float32)
        else:
            # Binary mask {0, 1}
            data = (np.random.rand(*shape) > 0.8).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))
    return path


class TestSafePathResolverV2(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

        # Standard layout
        self.ct_dir = self.root / "COVID-19-CT-Seg_20cases"
        self.mask_dir = self.root / "Infection_Mask"
        self.ct_dir.mkdir(parents=True)
        self.mask_dir.mkdir(parents=True)

        self.resolver = SafePathResolver(raw_data_dir=str(self.root), strict_nifti=True)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # ---------------------------------------------------------------------
    # Scenario 1: Ancestor directory containing the word 'infection'
    # ---------------------------------------------------------------------
    def test_01_ancestor_directory_containing_infection(self):
        """Ancestor directory containing 'infection' (e.g. Kaggle mount) must NOT disqualify valid CT."""
        mount_root = self.root / "kaggle_infection_dataset_mount"
        kaggle_ct_dir = mount_root / "COVID-19-CT-Seg_20cases"
        ct_file = kaggle_ct_dir / "coronacases_001.nii.gz"
        create_dummy_nii(ct_file, is_ct=True)

        resolver = SafePathResolver(raw_data_dir=str(mount_root), strict_nifti=True)
        self.assertTrue(resolver._is_valid_type(ct_file, "ct"))
        resolved = resolver.resolve_patient_file("P_coronacases_001", file_type="ct")
        self.assertEqual(resolved.resolve(), ct_file.resolve())

    # ---------------------------------------------------------------------
    # Scenario 2: CT inside actual CT directory
    # ---------------------------------------------------------------------
    def test_02_ct_inside_actual_ct_directory(self):
        """CT files located in approved CT directory must be properly recognized."""
        ct_file = self.ct_dir / "coronacases_002.nii.gz"
        create_dummy_nii(ct_file, is_ct=True)

        self.assertTrue(self.resolver._is_valid_type(ct_file, "ct"))
        resolved = self.resolver.resolve_patient_file("P_coronacases_002", file_type="ct")
        self.assertEqual(resolved.resolve(), ct_file.resolve())

    # ---------------------------------------------------------------------
    # Scenario 3: Mask inside actual mask directory
    # ---------------------------------------------------------------------
    def test_03_mask_inside_actual_mask_directory(self):
        """Mask files in approved mask directory must be properly recognized."""
        mask_file = self.mask_dir / "coronacases_002.nii.gz"
        create_dummy_nii(mask_file, is_ct=False)

        self.assertTrue(self.resolver._is_valid_type(mask_file, "mask"))
        resolved = self.resolver.resolve_patient_file("P_coronacases_002", file_type="mask")
        self.assertEqual(resolved.resolve(), mask_file.resolve())

    # ---------------------------------------------------------------------
    # Scenario 4: Different CT and mask filenames
    # ---------------------------------------------------------------------
    def test_04_different_ct_and_mask_filenames(self):
        """Resolver must pair CT and Mask for patient even with differing filenames (e.g. case009.nii vs coronacases_009.nii.gz)."""
        ct_file = self.ct_dir / "case009.nii"
        mask_file = self.mask_dir / "coronacases_009.nii.gz"
        create_dummy_nii(ct_file, shape=(16, 16, 8), is_ct=True)
        create_dummy_nii(mask_file, shape=(16, 16, 8), is_ct=False)

        resolved_ct, resolved_mask = self.resolver.resolve_patient("P_coronacases_009")
        self.assertEqual(resolved_ct.resolve(), ct_file.resolve())
        self.assertEqual(resolved_mask.resolve(), mask_file.resolve())
        self.assertNotEqual(resolved_ct.resolve(), resolved_mask.resolve())

    # ---------------------------------------------------------------------
    # Scenario 5: Multiple CT candidates raises AmbiguityError
    # ---------------------------------------------------------------------
    def test_05_multiple_ct_candidates_raises_ambiguity_error(self):
        """If multiple distinct CT files exist for the same patient, raises AmbiguityError."""
        cand1 = self.ct_dir / "case009.nii"
        cand2 = self.ct_dir / "coronacases_009.nii.gz"
        create_dummy_nii(cand1, is_ct=True)
        create_dummy_nii(cand2, is_ct=True)

        with self.assertRaises(AmbiguityError):
            self.resolver.resolve_patient_file("P_coronacases_009", file_type="ct")

    # ---------------------------------------------------------------------
    # Scenario 6: Multiple mask candidates raises AmbiguityError
    # ---------------------------------------------------------------------
    def test_06_multiple_mask_candidates_raises_ambiguity_error(self):
        """If multiple distinct Mask files exist for the same patient, raises AmbiguityError."""
        cand1 = self.mask_dir / "coronacases_009.nii"
        cand2 = self.mask_dir / "coronacases_009.nii.gz"
        create_dummy_nii(cand1, is_ct=False)
        create_dummy_nii(cand2, is_ct=False)

        with self.assertRaises(AmbiguityError):
            self.resolver.resolve_patient_file("P_coronacases_009", file_type="mask")

    # ---------------------------------------------------------------------
    # Scenario 7: Missing CT raises FileNotFoundError
    # ---------------------------------------------------------------------
    def test_07_missing_ct_raises_filenotfound(self):
        """Missing CT file must raise FileNotFoundError, not fallback to arbitrary file."""
        mask_file = self.mask_dir / "coronacases_003.nii.gz"
        create_dummy_nii(mask_file, is_ct=False)

        with self.assertRaises(FileNotFoundError):
            self.resolver.resolve_patient("P_coronacases_003")

    # ---------------------------------------------------------------------
    # Scenario 8: Missing mask raises FileNotFoundError
    # ---------------------------------------------------------------------
    def test_08_missing_mask_raises_filenotfound(self):
        """Missing mask file must raise FileNotFoundError, not fallback to arbitrary file."""
        ct_file = self.ct_dir / "coronacases_003.nii.gz"
        create_dummy_nii(ct_file, is_ct=True)

        with self.assertRaises(FileNotFoundError):
            self.resolver.resolve_patient("P_coronacases_003")

    # ---------------------------------------------------------------------
    # Scenario 9: Same-file CT-mask resolution raises PairIntegrityError
    # ---------------------------------------------------------------------
    def test_09_same_file_ct_mask_raises_pair_integrity_error(self):
        """CT and Mask resolving to the same file path violates non-identity rule and raises PairIntegrityError."""
        dummy_file = self.ct_dir / "coronacases_004.nii.gz"
        create_dummy_nii(dummy_file, is_ct=True)

        with self.assertRaises(PairIntegrityError):
            self.resolver.validate_pair(dummy_file, dummy_file, patient_id="P_coronacases_004")

    # ---------------------------------------------------------------------
    # Scenario 10: Invalid extensions (.dcm, .png, .txt) rejected
    # ---------------------------------------------------------------------
    def test_10_invalid_extensions_rejected(self):
        """Only .nii and .nii.gz files are accepted. Other formats are rejected."""
        dcm_file = self.ct_dir / "coronacases_005.dcm"
        png_file = self.ct_dir / "coronacases_005.png"
        txt_file = self.ct_dir / "coronacases_005.txt"
        dcm_file.write_bytes(b"dummy dcm")
        png_file.write_bytes(b"dummy png")
        txt_file.write_bytes(b"dummy txt")

        self.assertFalse(is_nii_file(dcm_file))
        self.assertFalse(is_nii_file(png_file))
        self.assertFalse(is_nii_file(txt_file))

        with self.assertRaises(PathResolverError):
            self.resolver.resolve_file(str(dcm_file), file_type="ct")

    # ---------------------------------------------------------------------
    # Scenario 11: Zero-byte files rejected
    # ---------------------------------------------------------------------
    def test_11_zero_byte_file_rejected(self):
        """Corrupted 0-byte NIfTI files must be rejected."""
        empty_ct = self.ct_dir / "coronacases_006.nii.gz"
        empty_ct.write_bytes(b"")

        with self.assertRaises((PathResolverError, FileNotFoundError)):
            self.resolver.resolve_patient_file("P_coronacases_006", file_type="ct")

    # ---------------------------------------------------------------------
    # Scenario 12: Shape mismatch raises PairIntegrityError
    # ---------------------------------------------------------------------
    def test_12_shape_mismatch_raises_pair_integrity_error(self):
        """CT and Mask with differing 3D spatial dimensions must raise PairIntegrityError."""
        ct_file = self.ct_dir / "coronacases_007.nii.gz"
        mask_file = self.mask_dir / "coronacases_007.nii.gz"
        create_dummy_nii(ct_file, shape=(32, 32, 10), is_ct=True)
        create_dummy_nii(mask_file, shape=(32, 32, 12), is_ct=False)  # Z dimension mismatch

        with self.assertRaises(PairIntegrityError):
            self.resolver.validate_pair(ct_file, mask_file, patient_id="P_coronacases_007")

    # ---------------------------------------------------------------------
    # Scenario 13: Affine mismatch raises PairIntegrityError
    # ---------------------------------------------------------------------
    def test_13_affine_mismatch_raises_pair_integrity_error(self):
        """CT and Mask with differing affine grids beyond tolerance (1e-3) must raise PairIntegrityError."""
        ct_file = self.ct_dir / "coronacases_008.nii.gz"
        mask_file = self.mask_dir / "coronacases_008.nii.gz"

        affine_ct = np.eye(4)
        affine_mask = np.eye(4)
        affine_mask[0, 0] = 2.5  # Significant voxel spacing / scale difference

        create_dummy_nii(ct_file, shape=(16, 16, 8), affine=affine_ct, is_ct=True)
        create_dummy_nii(mask_file, shape=(16, 16, 8), affine=affine_mask, is_ct=False)

        with self.assertRaises(PairIntegrityError):
            self.resolver.validate_pair(ct_file, mask_file, patient_id="P_coronacases_008")

    # ---------------------------------------------------------------------
    # Scenario 14: Incorrect patient mapping raises PairIntegrityError
    # ---------------------------------------------------------------------
    def test_14_incorrect_patient_mapping_raises_error(self):
        """Pairing a CT of patient 1 with Mask of patient 2 must raise PairIntegrityError."""
        ct_p1 = self.ct_dir / "coronacases_001.nii.gz"
        mask_p2 = self.mask_dir / "coronacases_002.nii.gz"
        create_dummy_nii(ct_p1, shape=(16, 16, 8), is_ct=True)
        create_dummy_nii(mask_p2, shape=(16, 16, 8), is_ct=False)

        with self.assertRaises(PairIntegrityError):
            self.resolver.validate_pair(ct_p1, mask_p2, patient_id="P_coronacases_001")

    # ---------------------------------------------------------------------
    # Additional Source: Radiopaedia Patient Resolution
    # ---------------------------------------------------------------------
    def test_15_radiopaedia_patient_resolution(self):
        """Resolver correctly parses and resolves multi-token Radiopaedia patient cases."""
        ct_radio = self.ct_dir / "radiopaedia_10_85902_1.nii.gz"
        mask_radio = self.mask_dir / "radiopaedia_10_85902_1.nii.gz"
        create_dummy_nii(ct_radio, shape=(20, 20, 5), is_ct=True)
        create_dummy_nii(mask_radio, shape=(20, 20, 5), is_ct=False)

        resolved_ct, resolved_mask = self.resolver.resolve_patient("P_radiopaedia_10_85902_1")
        self.assertEqual(resolved_ct.resolve(), ct_radio.resolve())
        self.assertEqual(resolved_mask.resolve(), mask_radio.resolve())

    # ---------------------------------------------------------------------
    # Scenario 16: Coronacases Case 007 and Case 010 actual Kaggle variants
    # ---------------------------------------------------------------------
    def test_16_coronacases_case007_and_010_variants(self):
        """Resolver correctly pairs coronacases_org_case007 and coronacases_org_case010."""
        ct_007 = self.ct_dir / "coronacases_org_case007.nii"
        mask_007 = self.mask_dir / "coronacases_007.nii"
        create_dummy_nii(ct_007, shape=(16, 16, 8), is_ct=True)
        create_dummy_nii(mask_007, shape=(16, 16, 8), is_ct=False)

        resolved_ct, resolved_mask = self.resolver.resolve_patient("P_coronacases_007")
        self.assertEqual(resolved_ct.resolve(), ct_007.resolve())
        self.assertEqual(resolved_mask.resolve(), mask_007.resolve())

        ct_010 = self.ct_dir / "coronacases_org_case010.nii"
        mask_010 = self.mask_dir / "coronacases_010.nii"
        create_dummy_nii(ct_010, shape=(16, 16, 8), is_ct=True)
        create_dummy_nii(mask_010, shape=(16, 16, 8), is_ct=False)

        resolved_ct_10, resolved_mask_10 = self.resolver.resolve_patient("P_coronacases_010")
        self.assertEqual(resolved_ct_10.resolve(), ct_010.resolve())
        self.assertEqual(resolved_mask_10.resolve(), mask_010.resolve())

    # ---------------------------------------------------------------------
    # Scenario 17: Coronacases Case 009 variant (case009.nii)
    # ---------------------------------------------------------------------
    def test_17_coronacases_case009_variants(self):
        """Resolver correctly handles case009.nii CT paired with coronacases_009.nii mask."""
        ct_009 = self.ct_dir / "case009.nii"
        mask_009 = self.mask_dir / "coronacases_009.nii"
        create_dummy_nii(ct_009, shape=(16, 16, 8), is_ct=True)
        create_dummy_nii(mask_009, shape=(16, 16, 8), is_ct=False)

        resolved_ct, resolved_mask = self.resolver.resolve_patient("P_coronacases_009")
        self.assertEqual(resolved_ct.resolve(), ct_009.resolve())
        self.assertEqual(resolved_mask.resolve(), mask_009.resolve())

    # ---------------------------------------------------------------------
    # Scenario 18: Radiopaedia actual Kaggle volume filenames
    # ---------------------------------------------------------------------
    def test_18_radiopaedia_real_kaggle_variants(self):
        """Resolver handles 'radiopaedia_org_covid-19-pneumonia-<case>_<study>_<series>-dcm.nii'."""
        ct_names = [
            "radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii",
            "radiopaedia_org_covid-19-pneumonia-10_85902_3-dcm.nii",
            "radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm.nii",
            "radiopaedia_org_covid-19-pneumonia-27_86410_0-dcm.nii",
            "radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii",
        ]
        pids = [
            "P_radiopaedia_10_85902_1",
            "P_radiopaedia_10_85902_3",
            "P_radiopaedia_14_85914_0",
            "P_radiopaedia_27_86410_0",
            "P_radiopaedia_4_85506_1",
        ]
        mask_names = [
            "radiopaedia_10_85902_1.nii",
            "radiopaedia_10_85902_3.nii",
            "radiopaedia_14_85914_0.nii",
            "radiopaedia_27_86410_0.nii",
            "radiopaedia_4_85506_1.nii",
        ]

        for ct_n, pid, mask_n in zip(ct_names, pids, mask_names):
            ct_p = self.ct_dir / ct_n
            mask_p = self.mask_dir / mask_n
            create_dummy_nii(ct_p, shape=(16, 16, 8), is_ct=True)
            create_dummy_nii(mask_p, shape=(16, 16, 8), is_ct=False)

            res_ct, res_mask = self.resolver.resolve_patient(pid)
            self.assertEqual(res_ct.resolve(), ct_p.resolve())
            self.assertEqual(res_mask.resolve(), mask_p.resolve())

    # ---------------------------------------------------------------------
    # Scenario 19: Nested directory layout (as found on Kaggle)
    # ---------------------------------------------------------------------
    def test_19_nested_directory_resolution(self):
        """Resolver handles Kaggle nested directory structure (<case>.nii/<actual>.nii)."""
        nested_subfolder = self.ct_dir / "coronacases_007.nii"
        nested_subfolder.mkdir(parents=True, exist_ok=True)
        actual_ct = nested_subfolder / "coronacases_org_case007.nii"
        actual_mask = self.mask_dir / "coronacases_007.nii"

        create_dummy_nii(actual_ct, shape=(16, 16, 8), is_ct=True)
        create_dummy_nii(actual_mask, shape=(16, 16, 8), is_ct=False)

        res_ct, res_mask = self.resolver.resolve_patient("P_coronacases_007")
        self.assertEqual(res_ct.resolve(), actual_ct.resolve())
        self.assertEqual(res_mask.resolve(), actual_mask.resolve())

    # ---------------------------------------------------------------------
    # Scenario 20: Cross-source mismatch raises PairIntegrityError
    # ---------------------------------------------------------------------
    def test_20_wrong_source_mismatch_raises_error(self):
        """Pairing coronacases CT with radiopaedia mask must raise PairIntegrityError."""
        ct_corona = self.ct_dir / "coronacases_org_001.nii"
        mask_radio = self.mask_dir / "radiopaedia_10_85902_1.nii"
        create_dummy_nii(ct_corona, shape=(16, 16, 8), is_ct=True)
        create_dummy_nii(mask_radio, shape=(16, 16, 8), is_ct=False)

        with self.assertRaises(PairIntegrityError):
            self.resolver.validate_pair(ct_corona, mask_radio, patient_id="P_coronacases_001")

    # ---------------------------------------------------------------------
    # Scenario 21: Radiopaedia series mismatch raises PairIntegrityError
    # ---------------------------------------------------------------------
    def test_21_radiopaedia_series_mismatch_raises_error(self):
        """Pairing study series 1 with series 3 must raise PairIntegrityError."""
        ct_ser1 = self.ct_dir / "radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii"
        mask_ser3 = self.mask_dir / "radiopaedia_10_85902_3.nii"
        create_dummy_nii(ct_ser1, shape=(16, 16, 8), is_ct=True)
        create_dummy_nii(mask_ser3, shape=(16, 16, 8), is_ct=False)

        with self.assertRaises(PairIntegrityError):
            self.resolver.validate_pair(ct_ser1, mask_ser3, patient_id="P_radiopaedia_10_85902_1")

    # ---------------------------------------------------------------------
    # Scenario 22: Ambiguous Radiopaedia candidates raises AmbiguityError
    # ---------------------------------------------------------------------
    def test_22_ambiguous_radiopaedia_candidates_raises_ambiguity(self):
        """Multiple distinct CT files for the same Radiopaedia series raises AmbiguityError."""
        cand1 = self.ct_dir / "radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii"
        cand2 = self.ct_dir / "radiopaedia_10_85902_1.nii"
        create_dummy_nii(cand1, is_ct=True)
        create_dummy_nii(cand2, is_ct=True)

        with self.assertRaises(AmbiguityError):
            self.resolver.resolve_patient_file("P_radiopaedia_10_85902_1", file_type="ct")


if __name__ == "__main__":
    unittest.main()
