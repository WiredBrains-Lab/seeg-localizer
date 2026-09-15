"""Tests for scientifically meaningful joins and scalar handling (no GUI needed)."""

import csv
import math
from pathlib import Path
import tempfile
import unittest

from viewer3d.cohort_lookup import (
    LESION_COLUMNS, cohort_contact_is_visible, load_cohort_lookup, match_cohort_lookup,
    subject_alias,
)
from viewer3d.mni_cohort import load_mni_cohort


class CohortLookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "lookup.csv"

    def table(self, rows, fields=None):
        fields = fields or ["subject", "electrode_name", "value"]
        with self.path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        return load_cohort_lookup(self.path)

    def test_subject_and_contact_joint_key_with_recording_aliases(self):
        lookup = self.table([
            ["sub-001", "ns5_1_LAMY01", -0.1], ["sub-002", "nf3_42_LAMY01", 0.2],
            ["sub-001", "ns5_2_RAMY01", 0.3],
        ])
        electrodes = [
            {"cohort_patient": " SUB-001 ", "name": "lamy1"},
            {"cohort_patient": "sub-002", "map_contact": "LAMY01"},
            {"cohort_patient": "sub-003", "name": "LAMY01"},
            {"cohort_patient": "sub-001", "saved_metadata": {"electrode_name": "RAMY01"}},
        ]
        matches = match_cohort_lookup(electrodes, lookup)
        self.assertEqual(matches.row_indices, [0, 1, None, 2])
        self.assertEqual(matches.unmatched_rows, [])

    def test_ambiguous_alias_is_not_guessed_but_exact_name_matches(self):
        lookup = self.table([["s1", "ns5_1_LA01", 1], ["s1", "nf3_2_LA01", 2]])
        matches = match_cohort_lookup([
            {"cohort_patient": "s1", "name": "LA1"},
            {"cohort_patient": "s1", "name": "ns5_1_LA01"},
        ], lookup)
        self.assertEqual(matches.row_indices, [None, 0])
        self.assertEqual(matches.ambiguous_contacts, [0])
        self.assertEqual(matches.unmatched_rows, [1])

    def test_conflicting_saved_identifiers_and_duplicate_locations_are_ambiguous(self):
        lookup = self.table([["s1", "LA01", 1], ["s1", "LA02", 2]])
        matches = match_cohort_lookup([
            {"cohort_patient": "s1", "name": "LA01", "map_contact": "LA02"},
            {"cohort_patient": "s1", "name": "LA02"},
            {"cohort_patient": "s1", "name": "LA02"},
        ], lookup)
        self.assertEqual(matches.row_indices, [None, None, None])
        self.assertEqual(matches.ambiguous_contacts, [0, 1, 2])

    def test_no_hemisphere_or_subject_number_guessing(self):
        lookup = self.table([["sub-001", "ns5_1_LA01", 1]])
        matches = match_cohort_lookup([
            {"cohort_patient": "sub-001", "name": "RA01"},
            {"cohort_patient": "sub-1", "name": "LA01"},
            {"cohort_patient": "sub-001", "name": "A01"},
            {"cohort_patient": "sub-001"},
        ], lookup)
        self.assertEqual(matches.row_indices, [None] * 4)

    def test_duplicate_keys_rejected_including_case_and_spaces(self):
        with self.assertRaisesRegex(ValueError, "Duplicate subject/electrode_name"):
            self.table([["s1", "LA01", 1], [" S1 ", "la01", 2]])

    def test_subj_subject_normalization(self):
        for name in ("SUBJ_009", "SUBJ09", "subj9", " SUBJ_09 "):
            self.assertEqual(subject_alias(name), "subj09")
        self.assertEqual(subject_alias("SUBJ_010"), "subj10")
        self.assertEqual(subject_alias("sub-009"), "sub-009")
        self.assertEqual(subject_alias("SUBJ_009_extra"), "subj_009_extra")

    def test_subject_alias_duplicates_are_rejected_before_indexing(self):
        with self.assertRaisesRegex(ValueError, "Duplicate subject/electrode_name"):
            self.table([["SUBJ09", "LA01", 1], ["SUBJ_009", "LA01", 2]])

    def test_final_name_component_is_used_for_any_recording_prefix(self):
        lookup = self.table([
            ["SUBJ09", "session_recording_123_LAMY01", -0.1],
            ["SUBJ10", "another_stream_456_LAMY01", 0.2],
        ])
        matches = match_cohort_lookup([
            {"cohort_patient": "SUBJ_009", "name": "LAMY01"},
            {"cohort_patient": "SUBJ_010", "name": "LAMY01"},
        ], lookup)
        self.assertEqual(matches.row_indices, [0, 1])

    def test_ct_csv_without_subject_column_joins_using_subject_folders(self):
        root = Path(self.temp.name)
        directories = [root / "SUBJ_009", root / "SUBJ_010" / "registration"]
        for directory in directories:
            directory.mkdir(parents=True)
            with (directory / "ct_elec_info.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["name", "shaft", "mni_x", "mni_y", "mni_z"])
                writer.writerow(["LAMY01", "LAMY", -20, 10, 5])
        lookup = self.table([
            ["SUBJ09", "ns5_1_LAMY01", -0.1], ["SUBJ10", "ns5_1_LAMY01", 0.2],
        ])
        electrodes, summary = load_mni_cohort(root)
        self.assertEqual(summary["patient_ids"], ["SUBJ_009", "SUBJ_010"])
        self.assertEqual(match_cohort_lookup(electrodes, lookup).row_indices, [0, 1])
        for selected_root in (directories[0], directories[0] / "ct_elec_info.csv"):
            electrodes, summary = load_mni_cohort(selected_root)
            self.assertEqual(summary["patient_ids"], ["SUBJ_009"])
            self.assertEqual(match_cohort_lookup(electrodes, lookup).row_indices, [0])

    def test_missing_key_empty_data_bad_headers_and_ragged_rows_rejected(self):
        for rows, fields in (
            ([["s1", 1]], ["subject", "value"]),
            ([["s1", "", 1]], None),
            ([], None),
            ([["s1", "LA01", "unknown"]], None),
            ([["s1", "LA01", 1, 2]], None),
            ([["s1", "LA01", 1, 2]], ["subject", "electrode_name", "value", "value"]),
        ):
            with self.subTest(rows=rows, fields=fields), self.assertRaises(ValueError):
                self.table(rows, fields)

    def test_missing_nonfinite_and_invalid_values_are_not_zero(self):
        lookup = self.table([["s1", f"LA{i}", value] for i, value in enumerate(
            ["0", "", "NaN", "inf", "-inf", "not measured", "-0.004"]
        )])
        self.assertEqual(lookup.values["value"][0], 0.0)
        self.assertTrue(all(math.isnan(value) for value in lookup.values["value"][1:6]))
        self.assertEqual(lookup.values["value"][-1], -0.004)

    def test_priority_columns_and_shared_limits_use_only_matched_rows(self):
        lookup = self.table([
            ["s1", "LA01", 10, -0.1, 0.4, 0.2],
            ["s2", "LA01", 10, -99, 99, 99],
        ], ["subject", "electrode_name", "n_splits", *LESION_COLUMNS])
        self.assertEqual(lookup.columns[:3], list(LESION_COLUMNS))
        self.assertEqual(lookup.color_limits(LESION_COLUMNS, [0, None]), (-0.4, 0.4))
        self.assertEqual(lookup.color_limits([LESION_COLUMNS[0]], [0]), (-0.1, 0.1))

    def test_zero_missing_or_no_matches_have_finite_nondegenerate_limits(self):
        lookup = self.table([["s1", "LA01", 0], ["s1", "LA02", ""]])
        for rows in ([0], [1], [None], []):
            self.assertEqual(lookup.color_limits(["value"], rows), (-1.0, 1.0))

    def test_visibility_combines_patient_selection_saved_visibility_and_valid_mni(self):
        electrode = dict(cohort_patient="s1", mni_x=1, mni_y=2, mni_z=3)
        self.assertTrue(cohort_contact_is_visible(electrode))
        self.assertTrue(cohort_contact_is_visible(electrode, {"s1"}))
        self.assertFalse(cohort_contact_is_visible(electrode, set()))
        self.assertFalse(cohort_contact_is_visible(electrode, {"s2"}))
        self.assertFalse(cohort_contact_is_visible(dict(electrode, visible=False)))
        self.assertFalse(cohort_contact_is_visible(dict(electrode, mni_x="nan")))


if __name__ == "__main__":
    unittest.main()
