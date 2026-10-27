"""
test_pipeline.py - Comprehensive Test Suite for DataLens

Tests cover universal data type detection, consistent missing-value normalization,
descriptive statistics, robust cleaning, non-destructive safety, CSV export,
and Streamlit AppTest integration across 19 synthetic scenarios and edge cases.
"""

import io
import os
import sys
import csv
import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from analyzer import (
    read_csv_file,
    detect_duplicate_columns,
    normalize_series_missing_markers,
    infer_column_semantic_type,
    normalize_dataframe,
    get_column_info,
    get_missing_value_summary,
    count_duplicate_rows,
    get_data_quality_summary,
    get_numeric_columns,
    get_numeric_statistics,
    get_non_numeric_statistics,
    remove_duplicate_rows,
    handle_missing_values,
    clean_dataset,
    export_dataframe_to_csv,
    MalformedCSVError,
    diagnose_csv_structure,
)

APP_PATH = os.path.join(WORKSPACE_DIR, "app.py")


class TestUniversalDataPipeline(unittest.TestCase):
    """Unit tests for universal data type detection, quality metrics, and cleaning."""

    def test_01_clean_numeric_and_categorical(self):
        """Scenario 1: Clean numeric and categorical data."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "score": [88.5, 92.0, 79.5, 95.0],
            "city": ["New York", "London", "Paris", "Tokyo"],
        })
        norm_df, meta = normalize_dataframe(df)
        self.assertEqual(meta["semantic_types"]["age"], "Numeric (Integer)")
        self.assertEqual(meta["semantic_types"]["score"], "Numeric (Float)")
        self.assertEqual(meta["semantic_types"]["city"], "Categorical")
        self.assertEqual(get_numeric_columns(norm_df), ["age", "score"])

    def test_02_numeric_stored_as_strings(self):
        """Scenario 2: Numeric values stored as strings with commas/whitespace."""
        df = pd.DataFrame({
            "price": [" $100 ", "$200", "$300"],  # will test below
            "salary": [" 50,000 ", "65,000.50", " 80,000 "],
            "count": ["10", "20", "30"],
        })
        norm_df, meta = normalize_dataframe(df)
        self.assertEqual(meta["semantic_types"]["salary"], "Numeric (Float)")
        self.assertEqual(meta["semantic_types"]["count"], "Numeric (Integer)")
        self.assertIn("salary", get_numeric_columns(norm_df))
        self.assertIn("count", get_numeric_columns(norm_df))

    def test_03_mixed_numeric_and_text(self):
        """Scenario 3: Mixed numeric and arbitrary text values."""
        df = pd.DataFrame({
            "mixed_col": ["10", "20", "pending_review", "40"],
        })
        norm_df, meta = normalize_dataframe(df)
        # Should not destroy "pending_review"
        self.assertEqual(meta["semantic_types"]["mixed_col"], "Mixed/Ambiguous")
        self.assertEqual(norm_df["mixed_col"].iloc[2], "pending_review")
        self.assertNotIn("mixed_col", get_numeric_columns(norm_df))

    def test_04_empty_and_whitespace_cells(self):
        """Scenario 4: Empty strings and whitespace-only cells."""
        df = pd.DataFrame({
            "val": ["10", "   ", "", "40"],
            "label": ["A", "  ", "B", ""],
        })
        norm_df, meta = normalize_dataframe(df)
        self.assertTrue(pd.isna(norm_df["val"].iloc[1]))
        self.assertTrue(pd.isna(norm_df["val"].iloc[2]))
        self.assertTrue(pd.isna(norm_df["label"].iloc[1]))
        self.assertTrue(pd.isna(norm_df["label"].iloc[3]))
        self.assertEqual(norm_df["val"].isnull().sum(), 2)

    def test_05_common_null_markers_configurable(self):
        """Scenario 5: Common null markers with configurable recognition."""
        df = pd.DataFrame({
            "numeric_with_dash": ["10.5", "20.2", "-", "40.1"],
            "custom_marker_col": ["alpha", "beta", "???", "gamma"],
        })
        # Standard run: '-' in otherwise numeric column contextually inferred as missing
        norm_df, meta = normalize_dataframe(df, custom_markers=["???"])
        self.assertTrue(pd.isna(norm_df["numeric_with_dash"].iloc[2]))
        self.assertEqual(meta["semantic_types"]["numeric_with_dash"], "Numeric (Float)")
        self.assertTrue(pd.isna(norm_df["custom_marker_col"].iloc[2]))

    def test_06_categorical_and_free_text(self):
        """Scenario 6: Categorical and free-text columns."""
        df = pd.DataFrame({
            "status": ["active", "pending", "active", "active", "closed"] * 5,
            "comment": [f"This is unique user comment number {i} with free text details" for i in range(25)],
        })
        norm_df, meta = normalize_dataframe(df)
        self.assertEqual(meta["semantic_types"]["status"], "Categorical")
        self.assertEqual(meta["semantic_types"]["comment"], "Text")

    def test_07_boolean_columns(self):
        """Scenario 7: Boolean columns in diverse representations."""
        df = pd.DataFrame({
            "bool_raw": [True, False, True, False],
            "bool_str": ["true", "false", "True", "False"],
            "bool_yes_no": ["yes", "no", "yes", "no"],
            "bool_tf": ["t", "f", "t", "f"],
        })
        norm_df, meta = normalize_dataframe(df)
        for col in df.columns:
            self.assertEqual(meta["semantic_types"][col], "Boolean")
            self.assertNotIn(col, get_numeric_columns(norm_df))

    def test_08_datetime_columns(self):
        """Scenario 8: Date/time columns in multiple formats."""
        df = pd.DataFrame({
            "iso_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "slash_date": ["01/15/2022", "02/20/2022", "03/25/2022"],
            "timestamp": ["2024-05-10 14:30:00", "2024-05-11 15:00:00", "2024-05-12 16:15:00"],
        })
        norm_df, meta = normalize_dataframe(df)
        for col in df.columns:
            self.assertEqual(meta["semantic_types"][col], "Date/Time")
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(norm_df[col]))

    def test_09_missing_values_in_several_columns(self):
        """Scenario 9: Missing values across several columns and quality metrics."""
        df = pd.DataFrame({
            "A": [1, None, 3, 4],
            "B": ["x", "y", None, "w"],
            "C": [10.5, None, None, 40.0],
        })
        norm_df, _ = normalize_dataframe(df)
        dq = get_data_quality_summary(norm_df)
        self.assertEqual(dq["total_rows"], 4)
        self.assertEqual(dq["total_columns"], 3)
        self.assertEqual(dq["total_missing"], 4)
        self.assertEqual(dq["rows_with_missing"], 2)  # row 1 has A and C missing, row 2 has B and C missing
        self.assertEqual(dq["columns_with_missing"], 3)

    def test_10_duplicate_rows(self):
        """Scenario 10: Completely duplicated rows detection and removal."""
        df = pd.DataFrame({
            "A": [1, 2, 2, 3],
            "B": ["a", "b", "b", "c"],
        })
        norm_df, _ = normalize_dataframe(df)
        self.assertEqual(count_duplicate_rows(norm_df), 1)
        cleaned, removed = remove_duplicate_rows(norm_df)
        self.assertEqual(removed, 1)
        self.assertEqual(len(cleaned), 3)

    def test_11_all_missing_column(self):
        """Scenario 11: Columns containing only missing values."""
        df = pd.DataFrame({
            "valid": [1, 2, 3],
            "all_na": [None, np.nan, "NA"],
        })
        norm_df, meta = normalize_dataframe(df)
        self.assertEqual(meta["semantic_types"]["all_na"], "All Missing")
        # Cleaning with mean should NOT invent an arbitrary value for all_na
        cleaned, summary = clean_dataset(df, remove_duplicates=False, missing_strategy="Fill numeric with mean")
        self.assertTrue(cleaned["all_na"].isnull().all())

    def test_12_empty_and_header_only_csv(self):
        """Scenario 12: Empty and header-only CSV files."""
        csv_header_only = "col1,col2,col3\n"
        df = read_csv_file(io.StringIO(csv_header_only))
        self.assertEqual(len(df), 0)
        self.assertEqual(len(df.columns), 3)

        dq = get_data_quality_summary(df)
        self.assertEqual(dq["total_rows"], 0)
        self.assertEqual(dq["total_missing"], 0)

        miss = get_missing_value_summary(df)
        self.assertEqual(miss["Missing-Value Percentage (%)"].iloc[0], 0.0)

        stats = get_numeric_statistics(df)
        self.assertEqual(len(stats), 0)

    def test_13_single_row_dataset(self):
        """Scenario 13: Single-row dataset."""
        df = pd.DataFrame({
            "id": [101],
            "score": [95.5],
            "category": ["Alpha"],
        })
        norm_df, _ = normalize_dataframe(df)
        stats = get_numeric_statistics(norm_df)
        self.assertEqual(len(stats), 2)
        # std on 1 observation is 0.0 or nan without error
        self.assertEqual(stats.loc[stats["Column Name"] == "score", "Count"].values[0], 1)

    def test_14_infinite_and_non_finite_values(self):
        """Scenario 14: Infinite and non-finite numeric values."""
        df = pd.DataFrame({
            "val": [10.0, np.inf, 30.0, -np.inf, 50.0],
        })
        norm_df, _ = normalize_dataframe(df)
        stats = get_numeric_statistics(norm_df)
        # Finite count should be 3
        count = stats.loc[stats["Column Name"] == "val", "Count"].values[0]
        self.assertEqual(count, 3)
        mean_val = stats.loc[stats["Column Name"] == "val", "Mean"].values[0]
        self.assertEqual(mean_val, 30.0)

    def test_15_constant_columns(self):
        """Scenario 15: Columns with constant values."""
        df = pd.DataFrame({
            "const_num": [42.0, 42.0, 42.0, 42.0],
            "const_cat": ["Fixed", "Fixed", "Fixed", "Fixed"],
        })
        norm_df, _ = normalize_dataframe(df)
        stats = get_numeric_statistics(norm_df)
        std_val = stats.loc[stats["Column Name"] == "const_num", "Standard Deviation"].values[0]
        self.assertEqual(std_val, 0.0)

    def test_16_column_names_with_spaces_and_special_characters(self):
        """Scenario 16: Column names with spaces, slashes, and symbols."""
        df = pd.DataFrame({
            "Total Price ($)": [100.5, 200.0, 300.75],
            "Date/Time (UTC)": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Status # Code": ["OK", "WARN", "OK"],
        })
        norm_df, meta = normalize_dataframe(df)
        self.assertIn("Total Price ($)", meta["semantic_types"])
        self.assertEqual(meta["semantic_types"]["Total Price ($)"], "Numeric (Float)")

    def test_17_ambiguous_columns_preserved_safely(self):
        """Scenario 17: Ambiguous columns that must not be misclassified."""
        df = pd.DataFrame({
            "phone_code": ["123-456-7890", "987-654-3210", "555-123-4567"],
            "hyphenated_words": ["pre-commit", "user-friendly", "full-time"],
        })
        norm_df, meta = normalize_dataframe(df)
        # Must not be parsed as dates or numbers
        self.assertEqual(meta["semantic_types"]["hyphenated_words"], "Categorical")
        self.assertNotIn("hyphenated_words", get_numeric_columns(norm_df))

    def test_18_cleaning_pipeline_integrity(self):
        """Scenario 18: Full cleaning pipeline integrity and tracking."""
        df = pd.DataFrame({
            "A": ["10", "20", "20", None],
            "B": ["x", "y", "y", "z"],
            "C": [10.5, 20.0, 20.0, np.nan],
        })
        cleaned_df, summary = clean_dataset(
            df, remove_duplicates=True, missing_strategy="Fill numeric with mean"
        )
        self.assertEqual(summary["initial_rows"], 4)
        self.assertEqual(summary["final_rows"], 3)
        self.assertEqual(summary["duplicate_rows_removed"], 1)
        self.assertEqual(summary["missing_cells_filled"], 2)
        self.assertTrue(summary["changes_made"])
        # Original df must be completely unchanged
        self.assertEqual(len(df), 4)

        # CSV export
        csv_bytes = export_dataframe_to_csv(cleaned_df)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertIn(b"10", csv_bytes)

    def test_19_quoted_commas(self):
        """Scenario 19: Valid CSV containing quoted commas."""
        csv_text = 'Name,City,Age\n"Smith, John","New York, NY",30\n"Doe, Jane","San Francisco, CA",28\n'
        df = read_csv_file(io.StringIO(csv_text))
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["Name", "City", "Age"])
        self.assertEqual(df["Name"].iloc[0], "Smith, John")
        self.assertEqual(df["City"].iloc[1], "San Francisco, CA")

    def test_20_escaped_quotes(self):
        """Scenario 20: Valid CSV containing RFC 4180 escaped quotes."""
        csv_text = 'ID,Quote\n1,"He said, ""Hello World!"""\n2,"It\'s a ""great"" day"\n'
        df = read_csv_file(io.StringIO(csv_text))
        self.assertEqual(len(df), 2)
        self.assertEqual(df["Quote"].iloc[0], 'He said, "Hello World!"')
        self.assertEqual(df["Quote"].iloc[1], 'It\'s a "great" day')

    def test_21_utf8_bom(self):
        """Scenario 21: CSV with UTF-8 BOM byte sequence."""
        bom_bytes = b"\xef\xbb\xbfPrefix,Assignment,Tutorial\n05,57.14,34.09\n"
        df = read_csv_file(io.BytesIO(bom_bytes))
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), ["Prefix", "Assignment", "Tutorial"])
        self.assertFalse(df.columns[0].startswith("\ufeff"))

    def test_22_utf16_encoding(self):
        """Scenario 22: CSV encoded with UTF-16."""
        u16_bytes = "Col1,Col2,Col3\n10,20,30\n40,50,60\n".encode("utf-16")
        df = read_csv_file(io.BytesIO(u16_bytes))
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["Col1", "Col2", "Col3"])
        self.assertEqual(df["Col1"].iloc[0], 10)

    def test_23_different_newlines(self):
        """Scenario 23: Different newline conventions (CRLF, LF, CR)."""
        # CRLF (Windows)
        crlf = "A,B\r\n1,2\r\n3,4\r\n"
        df_crlf = read_csv_file(io.StringIO(crlf))
        self.assertEqual(len(df_crlf), 2)

        # LF (Unix)
        lf = "A,B\n1,2\n3,4\n"
        df_lf = read_csv_file(io.StringIO(lf))
        self.assertEqual(len(df_lf), 2)

        # CR (Legacy Mac)
        cr = b"A,B\r1,2\r3,4\r"
        df_cr = read_csv_file(io.BytesIO(cr))
        self.assertEqual(len(df_cr), 2)

    def test_24_inconsistent_fields_error_diagnostics(self):
        """Scenario 24: Inconsistent field counts raise MalformedCSVError with exact line details."""
        csv_text = (
            "Prefix,Assignment,Tutorial,Midterm,TakeHome,Final\n"
            "05,57.14,34.09,64.38,51.48,52.50\n"
            "08,63.40,86.21,63.12,72.78,,\n"  # Line 3: 7 fields instead of 6
        )
        with self.assertRaises(MalformedCSVError) as ctx:
            read_csv_file(io.StringIO(csv_text), allow_recovery=False)

        err_msg = str(ctx.exception)
        self.assertIn("Expected 6 fields", err_msg)
        self.assertIn("saw 7", err_msg)
        self.assertIn("Line 3", err_msg)
        self.assertTrue(ctx.exception.diagnostics.get("is_recoverable"))

    def test_25_inconsistent_fields_safe_recovery(self):
        """Scenario 25: Safe recovery repairs extra trailing fields without skipping rows."""
        csv_text = (
            "Prefix,Assignment,Tutorial,Midterm,TakeHome,Final\n"
            "05,57.14,34.09,64.38,51.48,52.50\n"
            "08,63.40,86.21,63.12,72.78,,\n"  # Extra empty trailing comma
            "07,80.44,90.20,46.25,91.48,72.22\n"
        )
        df = read_csv_file(io.StringIO(csv_text), allow_recovery=True)
        self.assertEqual(len(df), 3)  # None skipped!
        self.assertEqual(list(df.columns), ["Prefix", "Assignment", "Tutorial", "Midterm", "TakeHome", "Final"])
        self.assertIn("recovery_report", df.attrs)
        report = df.attrs["recovery_report"]
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["line"], 3)
        self.assertEqual(report[0]["data_lost"], [])

    def test_26_class_grades_investigation_and_recovery(self):
        """Scenario 26: Actual or simulated class-grades.csv reproduction."""
        class_grades_path = r"D:\class-grades.csv"
        if os.path.exists(class_grades_path):
            with open(class_grades_path, "rb") as f:
                content = f.read()
        else:
            # Synthetic equivalent of class-grades.csv
            lines = ["Prefix,Assignment,Tutorial,Midterm,TakeHome,Final"]
            for i in range(2, 101):
                if i in [22, 40, 62]:
                    lines.append("08,63.40,86.21,63.12,72.78,,")
                else:
                    lines.append("05,57.14,34.09,64.38,51.48,52.50")
            content = "\n".join(lines).encode("utf-8")

        # Strict mode: must raise MalformedCSVError pinpointing lines 22, 40, 62
        with self.assertRaises(MalformedCSVError) as ctx:
            read_csv_file(io.BytesIO(content), allow_recovery=False)

        err_msg = str(ctx.exception)
        self.assertIn("Line 22", err_msg)
        self.assertIn("Line 40", err_msg)
        self.assertIn("Line 62", err_msg)
        self.assertIn("saw 7", err_msg)

        # Recovery mode: must load all 99 rows without skipping
        recovered_df = read_csv_file(io.BytesIO(content), allow_recovery=True)
        self.assertEqual(len(recovered_df), 99)
        self.assertEqual(len(recovered_df.attrs["recovery_report"]), 3)

    def test_27_malformed_quoting_error(self):
        """Scenario 27: Malformed quoting (unclosed quote) raises informative error."""
        bad_quotes = 'Col1,Col2\n"Unclosed string,123\n'
        with self.assertRaises((MalformedCSVError, ValueError)) as ctx:
            read_csv_file(io.StringIO(bad_quotes), allow_recovery=False)
        self.assertIn("CSV parsing error", str(ctx.exception))

    def test_28_stable_row_id_non_missing_preservation_class_grades(self):
        """Scenario 28: Using stable row identifier, verify zero non-missing values are altered."""
        class_grades_path = r"D:\class-grades.csv"
        if os.path.exists(class_grades_path):
            with open(class_grades_path, "rb") as f:
                content = f.read()
        else:
            lines = ["Prefix,Assignment,Tutorial,Midterm,TakeHome,Final"]
            for i in range(2, 101):
                if i == 9:
                    lines.append("07,75.27,85.29,66.88,,77.78")
                elif i in [22, 40, 62]:
                    lines.append("08,63.40,86.21,63.12,72.78,,")
                else:
                    lines.append("05,57.14,34.09,64.38,51.48,52.50")
            content = "\n".join(lines).encode("utf-8")

        raw_text = content.decode("utf-8")
        raw_reader = csv.reader(io.StringIO(raw_text))
        raw_rows = list(raw_reader)
        header = raw_rows[0]
        raw_records = raw_rows[1:]

        recovered_df = read_csv_file(io.BytesIO(content), allow_recovery=True)
        self.assertEqual(len(recovered_df), len(raw_records))

        row_ids = [f"rec_{i+2}" for i in range(len(recovered_df))]
        recovered_df["_temp_id"] = row_ids

        for strat in ["Fill numeric with mean", "Fill numeric with median", "Fill with mode"]:
            cleaned_df, summary = clean_dataset(
                recovered_df.drop(columns=["_temp_id"]),
                remove_duplicates=False,
                missing_strategy=strat,
            )
            cleaned_df["_temp_id"] = row_ids

            numeric_cols = ["Assignment", "Tutorial", "Midterm", "TakeHome", "Final"]
            for i, r_orig in enumerate(raw_records):
                line_no = i + 2
                r_cleaned = cleaned_df[cleaned_df["_temp_id"] == f"rec_{line_no}"].iloc[0]
                for col_idx, col in enumerate(header):
                    if col in numeric_cols:
                        orig_val = r_orig[col_idx].strip()
                        if orig_val != "":
                            self.assertAlmostEqual(
                                float(orig_val),
                                float(r_cleaned[col]),
                                places=4,
                                msg=f"Value changed at line {line_no}, col {col}: {orig_val} vs {r_cleaned[col]} in {strat}",
                            )

            for malformed_line, exp_takehome in [(22, 72.78), (40, 90.0), (62, 24.77)]:
                mal_row = cleaned_df[cleaned_df["_temp_id"] == f"rec_{malformed_line}"].iloc[0]
                self.assertAlmostEqual(float(mal_row["TakeHome"]), exp_takehome, places=2)
                self.assertFalse(pd.isna(mal_row["Midterm"]))
                self.assertFalse(pd.isna(mal_row["Assignment"]))
                self.assertFalse(pd.isna(mal_row["Tutorial"]))

            export_bytes = export_dataframe_to_csv(cleaned_df.drop(columns=["_temp_id"]))
            exp_header = export_bytes.decode("utf-8").splitlines()[0]
            self.assertNotIn("_temp_id", exp_header)

    def test_29_missing_cells_imputation_values_calculated_dynamically(self):
        """Scenario 29: Imputation replaces only the four missing cells with dynamically computed values."""
        class_grades_path = r"D:\class-grades.csv"
        if os.path.exists(class_grades_path):
            with open(class_grades_path, "rb") as f:
                content = f.read()
        else:
            lines = ["Prefix,Assignment,Tutorial,Midterm,TakeHome,Final"]
            for i in range(2, 101):
                if i == 9:
                    lines.append("07,75.27,85.29,66.88,,77.78")
                elif i in [22, 40, 62]:
                    lines.append("08,63.40,86.21,63.12,72.78,,")
                else:
                    lines.append("05,57.14,34.09,64.38,51.48,52.50")
            content = "\n".join(lines).encode("utf-8")

        recovered_df = read_csv_file(io.BytesIO(content), allow_recovery=True)
        norm_df, _ = normalize_dataframe(recovered_df)

        th_valid = norm_df["TakeHome"].dropna()
        fin_valid = norm_df["Final"].dropna()
        self.assertEqual(len(th_valid), 98)
        self.assertEqual(len(fin_valid), 96)

        expected_th_mean = th_valid.mean()
        expected_th_median = th_valid.median()
        expected_fin_mean = fin_valid.mean()
        expected_fin_median = fin_valid.median()

        # Mean Imputation
        cleaned_mean, summary_mean = clean_dataset(
            recovered_df, remove_duplicates=False, missing_strategy="Fill numeric with mean"
        )
        self.assertEqual(summary_mean["missing_cells_filled"], 4)
        self.assertEqual(summary_mean["missing_cells_remaining"], 0)
        self.assertAlmostEqual(float(cleaned_mean.loc[7, "TakeHome"]), expected_th_mean, places=3)
        self.assertAlmostEqual(float(cleaned_mean.loc[20, "Final"]), expected_fin_mean, places=3)
        self.assertAlmostEqual(float(cleaned_mean.loc[38, "Final"]), expected_fin_mean, places=3)
        self.assertAlmostEqual(float(cleaned_mean.loc[60, "Final"]), expected_fin_mean, places=3)

        # Median Imputation
        cleaned_med, summary_med = clean_dataset(
            recovered_df, remove_duplicates=False, missing_strategy="Fill numeric with median"
        )
        self.assertEqual(summary_med["missing_cells_filled"], 4)
        self.assertEqual(summary_med["missing_cells_remaining"], 0)
        self.assertAlmostEqual(float(cleaned_med.loc[7, "TakeHome"]), expected_th_median, places=4)
        self.assertAlmostEqual(float(cleaned_med.loc[20, "Final"]), expected_fin_median, places=4)
        self.assertAlmostEqual(float(cleaned_med.loc[38, "Final"]), expected_fin_median, places=4)
        self.assertAlmostEqual(float(cleaned_med.loc[60, "Final"]), expected_fin_median, places=4)

    def test_30_row_removal_and_immutability_class_grades(self):
        """Scenario 30: Row removal drops exactly the 4 rows with missing cells; raw_df remains immutable."""
        class_grades_path = r"D:\class-grades.csv"
        if os.path.exists(class_grades_path):
            with open(class_grades_path, "rb") as f:
                content = f.read()
        else:
            lines = ["Prefix,Assignment,Tutorial,Midterm,TakeHome,Final"]
            for i in range(2, 101):
                if i == 9:
                    lines.append("07,75.27,85.29,66.88,,77.78")
                elif i in [22, 40, 62]:
                    lines.append("08,63.40,86.21,63.12,72.78,,")
                else:
                    lines.append("05,57.14,34.09,64.38,51.48,52.50")
            content = "\n".join(lines).encode("utf-8")

        recovered_df = read_csv_file(io.BytesIO(content), allow_recovery=True)
        raw_df_copy = recovered_df.copy(deep=True)

        cleaned_drop, summary_drop = clean_dataset(
            recovered_df, remove_duplicates=False, missing_strategy="Remove rows with missing values"
        )

        self.assertEqual(summary_drop["initial_rows"], 99)
        self.assertEqual(summary_drop["final_rows"], 95)
        self.assertEqual(summary_drop["missing_rows_removed"], 4)
        self.assertEqual(len(cleaned_drop), 95)
        self.assertTrue(recovered_df.equals(raw_df_copy))

        csv_bytes = export_dataframe_to_csv(cleaned_drop)
        exp_lines = [l for l in csv_bytes.decode("utf-8").splitlines() if l.strip()]
        self.assertEqual(len(exp_lines), 96)
        self.assertNotIn("Unnamed", exp_lines[0])


class TestStreamlitAppIntegration(unittest.TestCase):
    """Integration tests running Streamlit AppTest against app.py."""

    def test_app_full_workflow(self):
        """Scenario 19: Full Streamlit AppTest workflow."""
        raw_csv = (
            "Name,Score,Salary,Active\n"
            "Alice,85, 75,000 ,true\n"
            "Bob,90, 50,000 ,false\n"
            "Bob,90, 50,000 ,false\n"
            "Charlie,-, 65,000 ,yes\n"
            "David,88, NA ,no\n"
        )
        csv_bytes = raw_csv.encode("utf-8")

        # 1. Launch app without upload
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(len(at.info) > 0)

        # 2. Upload CSV via sidebar
        at.sidebar.file_uploader[0].upload("employees.csv", csv_bytes)
        at.run()
        self.assertFalse(at.exception)

        # 3. Check executive KPI metrics
        metric_labels = [m.label for m in at.metric]
        self.assertIn("Total Rows", metric_labels)
        self.assertIn("Total Columns", metric_labels)
        self.assertIn("Missing Values", metric_labels)
        self.assertIn("Duplicate Rows", metric_labels)

        # 4. Check all 5 tabs exist
        tab_labels = [t.label for t in at.tabs]
        for t in ["Overview", "Data Quality", "Statistics", "Visualizations", "Data Cleaning & Export"]:
            self.assertIn(t, tab_labels)

        # 5. Check Visualizations tab
        charts = at.get("plotly_chart")
        self.assertTrue(len(charts) >= 1)

        # 6. Check Data Cleaning tab
        clean_buttons = [b for b in at.button if b.label == "Clean Dataset"]
        self.assertEqual(len(clean_buttons), 1)

        # Click Clean Dataset
        clean_buttons[0].click().run()
        self.assertFalse(at.exception)

        # Verify cleaned_df and download button
        self.assertIsNotNone(at.session_state["cleaned_df"])
        download_buttons = [b for b in at.download_button if "Download Cleaned CSV" in b.label]
        self.assertEqual(len(download_buttons), 1)

        # 7. Check reset on new upload
        new_csv = "ColA,ColB\n1,2\n3,4\n"
        at.sidebar.file_uploader[0].upload("new.csv", new_csv.encode("utf-8"))
        at.run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["cleaned_df"])

    def test_app_malformed_csv_and_safe_recovery(self):
        """Scenario 32: AppTest uploads malformed CSV, displays exact error, and safely recovers when enabled."""
        csv_text = (
            "Prefix,Assignment,Tutorial,Midterm,TakeHome,Final\n"
            "05,57.14,34.09,64.38,51.48,52.50\n"
            "08,63.40,86.21,63.12,72.78,,\n"
            "07,80.44,90.20,46.25,91.48,72.22\n"
        )
        csv_bytes = csv_text.encode("utf-8")

        # 1. Upload malformed CSV without recovery enabled (default)
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.file_uploader[0].upload("class_grades_sample.csv", csv_bytes)
        at.run()

        # Should present a clear error identifying Expected 6 fields, saw 7 at line 3
        self.assertTrue(len(at.error) > 0)
        err_msg = at.error[0].value
        self.assertIn("Expected 6 fields", err_msg)
        self.assertIn("saw 7", err_msg)
        self.assertIn("Line 3", err_msg)

        # 2. Toggle safe recovery checkbox in sidebar
        safe_cb = [cb for cb in at.sidebar.checkbox if "Safe CSV Recovery" in cb.label][0]
        safe_cb.check().run()

        # Error should now be gone, and warnings should show safe recovery applied
        self.assertEqual(len(at.error), 0)
        self.assertFalse(at.exception)
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("Safe CSV Recovery Applied" in w for w in warnings))

        # Check KPIs reflect all 3 rows (none skipped)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics.get("Total Rows"), "3")
        self.assertEqual(metrics.get("Total Columns"), "6")


if __name__ == "__main__":
    unittest.main()

