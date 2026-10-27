"""
analyzer.py - Data Analysis Module for DataLens

This module provides reusable helper functions for inspecting, summarizing,
analyzing data quality, normalizing types, and cleaning tabular dataset files (CSV).
"""

import io
import csv
import numpy as np
import pandas as pd

# Standard case-insensitive null value tokens recognized across CSV files
DEFAULT_NA_MARKERS = {
    "",
    " ",
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    "#n/a",
    "nil",
    "undefined",
    "n.a.",
    "not available",
}


class MalformedCSVError(ValueError):
    """
    Raised when a CSV file has inconsistent field counts or malformed quoting.
    Contains detailed diagnostics about every problematic row.
    """

    def __init__(self, message: str, diagnostics: dict = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def decode_csv_bytes(content: bytes) -> str:
    """
    Decodes raw CSV bytes using robust multi-encoding detection.
    Supports UTF-16 (with LE/BE BOM), UTF-8 with BOM (utf-8-sig),
    standard UTF-8, Windows-1252 (cp1252), and Latin-1 (iso-8859-1).

    Parameters:
        content: Raw bytes from a CSV file.

    Returns:
        str: Decoded string content.

    Raises:
        ValueError: If the byte sequence cannot be decoded.
    """
    if content.startswith(b"\xff\xfe") or content.startswith(b"\xfe\xff"):
        return content.decode("utf-16")
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig")

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Character encoding error: Unable to decode file as UTF-8, UTF-16, CP1252, or Latin-1."
    )


def diagnose_csv_structure(text: str) -> dict:
    """
    Parses CSV text row-by-row using python's csv.reader to detect
    inconsistent field counts, malformed quoting, and potential recovery actions.

    Parameters:
        text: Decoded CSV text string.

    Returns:
        dict: Detailed diagnostic report.
    """
    diagnostics = {
        "is_valid": True,
        "header": [],
        "expected_fields": 0,
        "total_rows": 0,
        "malformed_rows": [],
        "is_recoverable": False,
        "all_extra_are_empty": True,
        "error_message": "",
    }

    if not text.strip():
        diagnostics["is_valid"] = False
        diagnostics["error_message"] = "The CSV file is completely empty."
        return diagnostics

    try:
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if not header:
            diagnostics["is_valid"] = False
            diagnostics["error_message"] = "The CSV file contains no columns or data."
            return diagnostics

        diagnostics["header"] = header
        diagnostics["expected_fields"] = len(header)
        expected = len(header)

        malformed = []
        all_extra_are_empty = True

        for line_idx, row in enumerate(reader, start=2):
            diagnostics["total_rows"] += 1
            if len(row) != expected:
                diff = len(row) - expected
                extra_fields = row[expected:] if diff > 0 else []
                has_non_empty_extra = any(str(v).strip() != "" for v in extra_fields)
                if has_non_empty_extra:
                    all_extra_are_empty = False

                malformed.append(
                    {
                        "line": line_idx,
                        "expected": expected,
                        "seen": len(row),
                        "content": ",".join(row),
                        "extra_fields": extra_fields,
                        "missing_count": abs(diff) if diff < 0 else 0,
                        "has_non_empty_extra": has_non_empty_extra,
                    }
                )

        if malformed:
            diagnostics["is_valid"] = False
            diagnostics["malformed_rows"] = malformed
            diagnostics["is_recoverable"] = True
            diagnostics["all_extra_are_empty"] = all_extra_are_empty

            first = malformed[0]
            lines_summary = ", ".join(
                f"Line {m['line']} ({m['seen']} fields)" for m in malformed[:5]
            )
            if len(malformed) > 5:
                lines_summary += f", and {len(malformed) - 5} more lines"

            extra_explanation = (
                " (caused by extra trailing empty commas/delimiters)"
                if all_extra_are_empty
                else ""
            )

            diagnostics["error_message"] = (
                f"CSV parsing error: Expected {expected} fields in line {first['line']}, saw {first['seen']}{extra_explanation}.\n\n"
                f"Problematic rows detected ({len(malformed)} total):\n"
                + "\n".join(
                    f"• Line {m['line']}: Expected {expected} fields, saw {m['seen']} fields"
                    for m in malformed[:10]
                )
            )

        return diagnostics

    except csv.Error as e:
        diagnostics["is_valid"] = False
        diagnostics["is_recoverable"] = False
        diagnostics["error_message"] = f"CSV parsing error: Malformed quoting or syntax ({e})."
        return diagnostics


def read_csv_file(file, allow_recovery: bool = False) -> pd.DataFrame:
    """
    Reads an uploaded CSV file-like object into a pandas DataFrame.
    Supports valid CSV files containing quoted commas, escaped quotes,
    different newline conventions (CRLF, LF, CR), UTF-8 BOM, and supported text encodings.

    If the file has inconsistent field counts and allow_recovery is False,
    raises a MalformedCSVError detailing the exact lines and fields.
    If allow_recovery is True, safely repairs rows by trimming extra trailing
    fields or padding missing fields without skipping rows, and attaches
    a detailed recovery report to df.attrs['recovery_report'].

    Parameters:
        file: A file-like object (e.g., Streamlit's UploadedFile or BytesIO) or raw bytes.
        allow_recovery: Whether to safely recover malformed rows with inconsistent field counts.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        ValueError / MalformedCSVError: If the file is empty, malformed, or cannot be parsed.
    """
    if file is None:
        raise ValueError("No file was provided.")

    if hasattr(file, "seek"):
        file.seek(0)

    if hasattr(file, "read"):
        raw_content = file.read()
    else:
        raw_content = file

    if hasattr(file, "seek"):
        file.seek(0)

    # Decode bytes using robust multi-encoding detection
    if isinstance(raw_content, bytes):
        text = decode_csv_bytes(raw_content)
    else:
        text = str(raw_content)

    if not text.strip():
        raise ValueError("The uploaded CSV file is completely empty (no data or headers found).")

    # If recovery is not requested, parse strictly
    if not allow_recovery:
        try:
            df = pd.read_csv(io.StringIO(text))
            if df.empty and len(df.columns) == 0:
                raise ValueError("The uploaded CSV file contains no columns or data.")
            return df
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded CSV file is completely empty (no data or headers found).")
        except pd.errors.ParserError as pe:
            diag = diagnose_csv_structure(text)
            err_msg = diag.get("error_message") or str(pe)
            raise MalformedCSVError(err_msg, diagnostics=diag)
        except Exception as e:
            raise ValueError(f"Unable to read the CSV file. Details: {e}")

    # If allow_recovery is True:
    # First test if standard parse succeeds without any repair needed
    try:
        df = pd.read_csv(io.StringIO(text))
        if df.empty and len(df.columns) == 0:
            raise ValueError("The uploaded CSV file contains no columns or data.")
        return df
    except pd.errors.ParserError:
        pass

    # Perform safe row-by-row recovery without discarding entire rows
    diag = diagnose_csv_structure(text)
    if not diag["header"]:
        raise ValueError("The uploaded CSV file contains no columns or data.")

    header = diag["header"]
    expected = diag["expected_fields"]

    reader = csv.reader(io.StringIO(text))
    _ = next(reader, None)  # skip header

    recovered_rows = []
    recovery_report = []

    for line_idx, row in enumerate(reader, start=2):
        if len(row) == expected:
            recovered_rows.append(row)
        elif len(row) > expected:
            extra = row[expected:]
            omitted_non_empty = [v for v in extra if str(v).strip() != ""]
            recovered_rows.append(row[:expected])
            recovery_report.append(
                {
                    "line": line_idx,
                    "action": f"Trimmed {len(extra)} extra trailing field(s)",
                    "extra_values": extra,
                    "data_lost": omitted_non_empty,
                }
            )
        else:
            missing = expected - len(row)
            recovered_rows.append(row + [""] * missing)
            recovery_report.append(
                {
                    "line": line_idx,
                    "action": f"Padded {missing} missing field(s) with empty values",
                    "extra_values": [],
                    "data_lost": [],
                }
            )

    # Re-parse recovered rows through pd.read_csv so the recovered dataset
    # has standard pandas type inference and NA handling identical to clean files
    output_io = io.StringIO()
    writer = csv.writer(output_io)
    writer.writerow(header)
    writer.writerows(recovered_rows)
    output_io.seek(0)
    try:
        df = pd.read_csv(output_io)
    except Exception:
        df = pd.DataFrame(recovered_rows, columns=header)

    df.attrs["recovery_report"] = recovery_report
    df.attrs["recovery_diagnostics"] = diag
    return df


def detect_duplicate_columns(file) -> list:
    """
    Inspects the header line of a CSV file to check for duplicate column names.
    Properly handles UTF-8 BOM, UTF-16, and various encodings.

    Parameters:
        file: A file-like object.

    Returns:
        list: A list of column names that appear more than once in the header.
    """
    if file is None:
        return []

    try:
        if hasattr(file, "seek"):
            file.seek(0)

        raw_line = file.readline()

        # Reset seek position after reading
        if hasattr(file, "seek"):
            file.seek(0)

        if isinstance(raw_line, bytes):
            if raw_line.startswith(b"\xff\xfe") or raw_line.startswith(b"\xfe\xff"):
                line_text = raw_line.decode("utf-16")
            elif raw_line.startswith(b"\xef\xbb\xbf"):
                line_text = raw_line.decode("utf-8-sig")
            else:
                try:
                    line_text = raw_line.decode("utf-8-sig")
                except UnicodeDecodeError:
                    line_text = raw_line.decode("latin1", errors="replace")
        else:
            line_text = str(raw_line)

        if not line_text.strip():
            return []

        # Parse the header row
        csv_reader = csv.reader(io.StringIO(line_text))
        headers = next(csv_reader, [])

        seen = set()
        duplicates = []
        for col in headers:
            col_name = col.strip()
            if col_name in seen:
                if col_name not in duplicates:
                    duplicates.append(col_name)
            else:
                seen.add(col_name)

        return duplicates
    except Exception:
        if hasattr(file, "seek"):
            file.seek(0)
        return []


def normalize_series_missing_markers(
    series: pd.Series, custom_markers: set | list = None
) -> tuple[pd.Series, int]:
    """
    Replaces missing-value markers in a Series with np.nan.
    Recognizes empty cells, whitespace-only strings, standard NA tokens,
    and contextually infers placeholder markers (such as single dashes '-')
    in columns that are otherwise entirely numeric or date/time.

    Parameters:
        series: Input pandas Series.
        custom_markers: Optional collection of extra strings to treat as missing.

    Returns:
        tuple: (cleaned_series: pd.Series, markers_replaced_count: int)
    """
    if series.empty:
        return series.copy(), 0

    # Ensure custom markers are a lowercased set of stripped strings
    extra = set(str(m).strip().lower() for m in custom_markers) if custom_markers else set()
    all_markers = DEFAULT_NA_MARKERS | extra

    # If series is already boolean or datetime, check standard isnull
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return series.copy(), 0

    s = series.copy()
    initial_null_count = int(s.isnull().sum())

    # Only string/object series can contain text null markers like "NA" or " "
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        # 1. Identify whitespace or exact marker matches
        def is_marker(val):
            if pd.isna(val):
                return False
            if isinstance(val, str):
                stripped = val.strip()
                if stripped == "":
                    return True
                return stripped.lower() in all_markers
            return False

        mask_marker = s.map(is_marker)

        # 2. Contextual check for placeholder tokens like '-', '--', '?'
        # If not explicitly in custom_markers, only treat as missing if all other
        # non-null values convert to numbers or dates.
        contextual_candidates = {"-", "--", "?"} - all_markers
        if contextual_candidates:
            has_candidate = s.map(
                lambda v: isinstance(v, str) and v.strip() in contextual_candidates
            )
            if has_candidate.any():
                # Test remaining values if we treat candidate as null
                tentative_remaining = s[~mask_marker & ~has_candidate].dropna()
                if len(tentative_remaining) > 0:
                    # Check if all remaining values are reliably numeric
                    cleaned_test = (
                        tentative_remaining.astype(str)
                        .str.strip()
                        .str.replace(",", "", regex=False)
                    )
                    numeric_test = pd.to_numeric(cleaned_test, errors="coerce")
                    if numeric_test.notnull().all():
                        mask_marker = mask_marker | has_candidate
                    else:
                        # Check if all remaining values are date/time
                        if cleaned_test.str.contains(r"[-/:T]", regex=True).all():
                            dt_test = pd.to_datetime(cleaned_test, errors="coerce")
                            if dt_test.notnull().all():
                                mask_marker = mask_marker | has_candidate

        s = s.mask(mask_marker, np.nan)

    final_null_count = int(s.isnull().sum())
    markers_replaced = max(0, final_null_count - initial_null_count)
    return s, markers_replaced


def infer_column_semantic_type(series: pd.Series) -> tuple[str, pd.Series]:
    """
    Infers the universal semantic type of a column and safely converts it to
    the appropriate pandas data type without destroying data.

    Semantic Types:
        - "Numeric (Integer)"
        - "Numeric (Float)"
        - "Boolean"
        - "Date/Time"
        - "Categorical"
        - "Text"
        - "Mixed/Ambiguous"
        - "All Missing"

    Parameters:
        series: Input pandas Series (assumes missing markers have been normalized).

    Returns:
        tuple: (semantic_type: str, converted_series: pd.Series)
    """
    s_clean = series.dropna()

    if len(s_clean) == 0:
        return "All Missing", series.copy()

    # 1. Check Boolean
    if pd.api.types.is_bool_dtype(series):
        return "Boolean", series.copy()

    if series.dtype == object or pd.api.types.is_string_dtype(series):
        # Check string representations of booleans
        str_vals = s_clean.astype(str).str.strip().str.lower()
        unique_vals = set(str_vals.unique())
        if unique_vals.issubset({"true", "false"}):
            mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
            return "Boolean", mapped.astype("boolean")
        if unique_vals.issubset({"yes", "no"}):
            mapped = series.astype(str).str.strip().str.lower().map({"yes": True, "no": False})
            return "Boolean", mapped.astype("boolean")
        if unique_vals.issubset({"t", "f"}):
            mapped = series.astype(str).str.strip().str.lower().map({"t": True, "f": False})
            return "Boolean", mapped.astype("boolean")

    # 2. Check Numeric
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        # Exclude infinite values when checking if integer
        finite_vals = s_clean[~np.isinf(s_clean)]
        if len(finite_vals) > 0 and (finite_vals % 1 == 0).all():
            try:
                return "Numeric (Integer)", series.astype("Int64")
            except Exception:
                return "Numeric (Integer)", series.copy()
        return "Numeric (Float)", series.astype(float)

    if series.dtype == object or pd.api.types.is_string_dtype(series):
        # Check for numeric strings (handling thousands commas like 1,234.50)
        cleaned_strs = s_clean.astype(str).str.strip()
        cleaned_no_commas = cleaned_strs.str.replace(",", "", regex=False)
        numeric_attempt = pd.to_numeric(cleaned_no_commas, errors="coerce")

        if numeric_attempt.notnull().all():
            # 100% of non-null values converted to numbers
            full_cleaned = series.astype(str).str.strip().str.replace(",", "", regex=False)
            full_numeric = pd.to_numeric(full_cleaned, errors="coerce")
            finite_vals = full_numeric.dropna()
            finite_vals = finite_vals[~np.isinf(finite_vals)]
            if len(finite_vals) > 0 and (finite_vals % 1 == 0).all():
                try:
                    return "Numeric (Integer)", full_numeric.astype("Int64")
                except Exception:
                    return "Numeric (Integer)", full_numeric
            return "Numeric (Float)", full_numeric.astype(float)

    # 3. Check Date/Time
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time", series.copy()

    if series.dtype == object or pd.api.types.is_string_dtype(series):
        cleaned_strs = s_clean.astype(str).str.strip()
        # Avoid treating pure numbers as dates
        if not cleaned_strs.str.match(r"^-?\d+(\.\d+)?$").all():
            # Must contain date/time delimiters
            if cleaned_strs.str.contains(r"[-/:T]", regex=True).all():
                try:
                    dt_attempt = pd.to_datetime(cleaned_strs, errors="coerce", format="mixed")
                    if dt_attempt.notnull().all():
                        years = dt_attempt.dt.year
                        if (years >= 1800).all() and (years <= 2200).all():
                            full_dt = pd.to_datetime(series, errors="coerce", format="mixed")
                            return "Date/Time", full_dt
                except Exception:
                    pass

    # 4. Check Mixed vs Categorical vs Free Text
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        # If some values convert to numbers and others are arbitrary non-missing words
        cleaned_strs = s_clean.astype(str).str.strip().str.replace(",", "", regex=False)
        num_attempt = pd.to_numeric(cleaned_strs, errors="coerce")
        has_some_numbers = num_attempt.notnull().any()
        has_some_non_numbers = num_attempt.isnull().any()

        if has_some_numbers and has_some_non_numbers:
            return "Mixed/Ambiguous", series.copy()

        # Categorical vs Text check
        n_unique = s_clean.nunique()
        total_non_null = len(s_clean)
        str_clean = s_clean.astype(str).str.strip()
        avg_char_len = str_clean.str.len().mean() if total_non_null > 0 else 0
        avg_word_count = str_clean.str.split().str.len().mean() if total_non_null > 0 else 0
        uniqueness_ratio = n_unique / max(total_non_null, 1)

        # Free text columns feature multi-word phrases, sentences, longer strings, or high uniqueness
        is_free_text = (avg_char_len > 25 or avg_word_count > 3) and (uniqueness_ratio > 0.5 or total_non_null <= 10)

        if is_free_text or (uniqueness_ratio > 0.6 and total_non_null > 10):
            return "Text", series.copy()
        elif n_unique <= 30 or uniqueness_ratio <= 0.5:
            return "Categorical", series.copy()
        else:
            return "Text", series.copy()

    return "Categorical", series.copy()


def normalize_dataframe(
    df: pd.DataFrame, custom_markers: set | list = None
) -> tuple[pd.DataFrame, dict]:
    """
    Normalizes a DataFrame by detecting and standardizing missing-value markers,
    inferring universal semantic types, and safely converting column dtypes.
    Never modifies the original input DataFrame.

    Parameters:
        df: Input pandas DataFrame.
        custom_markers: Optional list or set of custom missing-value markers.

    Returns:
        tuple: (normalized_df: pd.DataFrame, metadata: dict)
            metadata contains:
                - 'semantic_types': dict[col, str]
                - 'missing_markers_recognized': dict[col, int]
                - 'total_missing_markers_recognized': int
                - 'columns_changed': list[str]
    """
    if df.empty or len(df.columns) == 0:
        return df.copy(), {
            "semantic_types": {},
            "missing_markers_recognized": {},
            "total_missing_markers_recognized": 0,
            "columns_changed": [],
        }

    normalized_df = pd.DataFrame(index=df.index)
    semantic_types = {}
    missing_markers_recognized = {}
    total_markers = 0
    columns_changed = []

    for col in df.columns:
        col_str = str(col)
        raw_series = df[col]

        # Step 1: Normalize missing-value markers
        s_no_markers, replaced_count = normalize_series_missing_markers(
            raw_series, custom_markers=custom_markers
        )
        missing_markers_recognized[col_str] = replaced_count
        total_markers += replaced_count

        # Step 2: Infer semantic type and convert
        sem_type, converted_series = infer_column_semantic_type(s_no_markers)
        semantic_types[col_str] = sem_type
        normalized_df[col] = converted_series

        # Track if column was changed
        if replaced_count > 0 or str(converted_series.dtype) != str(raw_series.dtype):
            columns_changed.append(col_str)

    metadata = {
        "semantic_types": semantic_types,
        "missing_markers_recognized": missing_markers_recognized,
        "total_missing_markers_recognized": total_markers,
        "columns_changed": columns_changed,
    }

    return normalized_df, metadata


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts column names, detected data types, inferred semantic types,
    and non-null counts into a summary DataFrame.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        pd.DataFrame: Table containing column details.
    """
    if len(df.columns) == 0:
        return pd.DataFrame(
            columns=["Column Name", "Data Type", "Semantic Type", "Non-Null Count", "Null Count"]
        )

    _, meta = normalize_dataframe(df)
    semantic_types = meta.get("semantic_types", {})

    summary_df = pd.DataFrame(
        {
            "Column Name": [str(col) for col in df.columns],
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Semantic Type": [semantic_types.get(str(col), "Unknown") for col in df.columns],
            "Non-Null Count": [int(df[col].count()) for col in df.columns],
            "Null Count": [int(df[col].isnull().sum()) for col in df.columns],
        }
    )
    return summary_df


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates missing value counts and percentages for each column in the DataFrame.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        pd.DataFrame: Table with columns 'Column Name', 'Missing-Value Count',
                      and 'Missing-Value Percentage (%)'.
    """
    total_rows = len(df)

    if len(df.columns) == 0:
        return pd.DataFrame(
            columns=["Column Name", "Missing-Value Count", "Missing-Value Percentage (%)"]
        )

    columns = [str(col) for col in df.columns]
    missing_counts = [int(df[col].isnull().sum()) for col in df.columns]

    if total_rows > 0:
        missing_percentages = [
            round((count / total_rows) * 100, 2) for count in missing_counts
        ]
    else:
        # Avoid division-by-zero when the dataset contains zero rows
        missing_percentages = [0.0 for _ in missing_counts]

    return pd.DataFrame(
        {
            "Column Name": columns,
            "Missing-Value Count": missing_counts,
            "Missing-Value Percentage (%)": missing_percentages,
        }
    )


def count_duplicate_rows(df: pd.DataFrame) -> int:
    """
    Counts completely duplicated rows in the DataFrame.
    A row is treated as a duplicate when all its column values match another row.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        int: Number of completely duplicate rows.
    """
    if len(df) == 0 or len(df.columns) == 0:
        return 0
    return int(df.duplicated().sum())


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """
    Calculates overall data quality metrics for the dataset.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        dict: Summary containing:
            - 'total_rows': int
            - 'total_columns': int
            - 'total_missing': int
            - 'rows_with_missing': int
            - 'columns_with_missing': int
            - 'duplicate_rows': int
    """
    total_rows = len(df)
    total_columns = len(df.columns)

    if total_rows == 0 or total_columns == 0:
        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_missing": 0,
            "rows_with_missing": 0,
            "columns_with_missing": 0,
            "duplicate_rows": 0,
        }

    missing_summary = get_missing_value_summary(df)
    total_missing = int(df.isnull().sum().sum())
    rows_with_missing = int(df.isnull().any(axis=1).sum())
    duplicate_rows = count_duplicate_rows(df)
    columns_with_missing = int((missing_summary["Missing-Value Count"] > 0).sum())

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_missing": total_missing,
        "rows_with_missing": rows_with_missing,
        "columns_with_missing": columns_with_missing,
        "duplicate_rows": duplicate_rows,
    }


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Returns a list of column names that contain numeric data (integers and floats),
    including string-represented numbers that can be reliably converted,
    explicitly excluding boolean columns.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        list: Names of eligible numeric columns.
    """
    if df.empty or len(df.columns) == 0:
        return []

    numeric_cols = []
    for col in df.columns:
        col_series = df[col]
        # Exclude booleans
        if pd.api.types.is_bool_dtype(col_series):
            continue

        if pd.api.types.is_numeric_dtype(col_series):
            numeric_cols.append(str(col))
        elif col_series.dtype == object or pd.api.types.is_string_dtype(col_series):
            # Check if strings can be reliably converted to numeric
            s_clean = col_series.dropna()
            if len(s_clean) > 0:
                cleaned_strs = (
                    s_clean.astype(str).str.strip().str.replace(",", "", regex=False)
                )
                converted = pd.to_numeric(cleaned_strs, errors="coerce")
                if converted.notnull().all():
                    numeric_cols.append(str(col))

    return numeric_cols


def get_numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes descriptive statistics for all eligible numeric columns in the DataFrame.
    Safely handles missing values, constant columns, single observations,
    infinite values, and all-null columns without crashing.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        pd.DataFrame: Table with numeric columns represented as rows, containing
                      count, mean, median, standard deviation, min, 25%, 50%, 75%, and max.
    """
    columns_layout = [
        "Column Name",
        "Count",
        "Mean",
        "Median",
        "Standard Deviation",
        "Min",
        "25%",
        "50%",
        "75%",
        "Max",
    ]

    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=columns_layout)

    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        return pd.DataFrame(columns=columns_layout)

    rows = []
    for col in numeric_cols:
        series = df[col]
        # Convert to numeric if needed, replacing inf/-inf with NaN
        if not pd.api.types.is_numeric_dtype(series):
            cleaned = (
                series.astype(str).str.strip().str.replace(",", "", regex=False)
            )
            s_num = pd.to_numeric(cleaned, errors="coerce")
        else:
            s_num = series.astype(float).copy()

        # Handle infinite values safely
        s_num = s_num.replace([np.inf, -np.inf], np.nan)
        valid = s_num.dropna()

        count_val = len(valid)
        if count_val == 0:
            rows.append(
                {
                    "Column Name": str(col),
                    "Count": 0,
                    "Mean": np.nan,
                    "Median": np.nan,
                    "Standard Deviation": np.nan,
                    "Min": np.nan,
                    "25%": np.nan,
                    "50%": np.nan,
                    "75%": np.nan,
                    "Max": np.nan,
                }
            )
        else:
            mean_val = valid.mean()
            median_val = valid.median()
            std_val = valid.std() if count_val > 1 else 0.0
            min_val = valid.min()
            q25 = valid.quantile(0.25)
            q50 = valid.quantile(0.50)
            q75 = valid.quantile(0.75)
            max_val = valid.max()

            rows.append(
                {
                    "Column Name": str(col),
                    "Count": count_val,
                    "Mean": mean_val,
                    "Median": median_val,
                    "Standard Deviation": std_val,
                    "Min": min_val,
                    "25%": q25,
                    "50%": q50,
                    "75%": q75,
                    "Max": max_val,
                }
            )

    return pd.DataFrame(rows, columns=columns_layout)


def get_non_numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary metrics for categorical, text, boolean, and date/time columns.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        pd.DataFrame: Table with non-numeric column summaries.
    """
    columns_layout = [
        "Column Name",
        "Semantic Type",
        "Count",
        "Unique Values",
        "Top / Mode Value",
        "Frequency",
    ]

    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=columns_layout)

    numeric_cols = set(get_numeric_columns(df))
    non_numeric_cols = [c for c in df.columns if str(c) not in numeric_cols]

    if not non_numeric_cols:
        return pd.DataFrame(columns=columns_layout)

    _, meta = normalize_dataframe(df)
    semantic_types = meta.get("semantic_types", {})

    rows = []
    for col in non_numeric_cols:
        col_str = str(col)
        series = df[col].dropna()
        count_val = len(series)
        unique_val = series.nunique() if count_val > 0 else 0

        if count_val > 0:
            modes = series.mode()
            top_val = str(modes.iloc[0]) if len(modes) > 0 else "N/A"
            freq_val = int(series.value_counts().iloc[0]) if len(modes) > 0 else 0
        else:
            top_val = "N/A"
            freq_val = 0

        rows.append(
            {
                "Column Name": col_str,
                "Semantic Type": semantic_types.get(col_str, "Categorical"),
                "Count": count_val,
                "Unique Values": unique_val,
                "Top / Mode Value": top_val,
                "Frequency": freq_val,
            }
        )

    return pd.DataFrame(rows, columns=columns_layout)


def remove_duplicate_rows(df: pd.DataFrame) -> tuple:
    """
    Removes completely duplicate rows from the DataFrame while keeping the first occurrence.
    Never modifies the original DataFrame.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        tuple: (cleaned_df: pd.DataFrame, rows_removed: int)
    """
    if df.empty or len(df.columns) == 0:
        return df.copy(), 0

    cleaned_df = df.drop_duplicates(keep="first").copy()
    rows_removed = len(df) - len(cleaned_df)
    return cleaned_df, rows_removed


def handle_missing_values(df: pd.DataFrame, strategy: str) -> tuple:
    """
    Applies the specified missing-value handling strategy to a copy of the dataset.
    Never modifies the original DataFrame.

    Supported strategies:
        - 'Keep missing values': Leaves the dataset unchanged.
        - 'Remove rows with missing values': Removes rows with at least one missing cell.
        - 'Fill numeric with mean': Fills missing values in numeric columns with column mean.
        - 'Fill numeric with median': Fills missing values in numeric columns with column median.
        - 'Fill with mode': Fills missing values with the most frequent value.

    Parameters:
        df: Input pandas DataFrame.
        strategy: Strategy name string.

    Returns:
        tuple: (cleaned_df: pd.DataFrame, stats: dict)
    """
    stats = {
        "strategy": strategy,
        "rows_removed": 0,
        "cells_filled": 0,
        "filled_columns": [],
    }

    if df.empty or len(df.columns) == 0:
        return df.copy(), stats

    cleaned_df = df.copy()

    if strategy == "Keep missing values":
        return cleaned_df, stats

    elif strategy == "Remove rows with missing values":
        dropped_df = cleaned_df.dropna(how="any").copy()
        stats["rows_removed"] = len(cleaned_df) - len(dropped_df)
        return dropped_df, stats

    elif strategy == "Fill numeric with mean":
        numeric_cols = get_numeric_columns(cleaned_df)
        for col in numeric_cols:
            series = cleaned_df[col]
            missing_count = int(series.isnull().sum())
            if missing_count > 0:
                # Convert to numeric if stored as string
                if not pd.api.types.is_numeric_dtype(series):
                    s_clean = (
                        series.astype(str).str.strip().str.replace(",", "", regex=False)
                    )
                    s_num = pd.to_numeric(s_clean, errors="coerce")
                else:
                    s_num = series.copy()

                # Filter out infinities for mean calculation
                finite_vals = s_num[~np.isinf(s_num)].dropna()
                if len(finite_vals) > 0:
                    mean_val = finite_vals.mean()
                    if not pd.isna(mean_val):
                        if mean_val % 1 == 0:
                            try:
                                filled_series = s_num.fillna(int(mean_val))
                            except Exception:
                                filled_series = s_num.astype(float).fillna(mean_val)
                        else:
                            filled_series = s_num.astype(float).fillna(round(mean_val, 4))
                        cleaned_df[col] = filled_series
                        stats["cells_filled"] += missing_count
                        stats["filled_columns"].append(str(col))
        return cleaned_df, stats

    elif strategy == "Fill numeric with median":
        numeric_cols = get_numeric_columns(cleaned_df)
        for col in numeric_cols:
            series = cleaned_df[col]
            missing_count = int(series.isnull().sum())
            if missing_count > 0:
                if not pd.api.types.is_numeric_dtype(series):
                    s_clean = (
                        series.astype(str).str.strip().str.replace(",", "", regex=False)
                    )
                    s_num = pd.to_numeric(s_clean, errors="coerce")
                else:
                    s_num = series.copy()

                finite_vals = s_num[~np.isinf(s_num)].dropna()
                if len(finite_vals) > 0:
                    median_val = finite_vals.median()
                    if not pd.isna(median_val):
                        if median_val % 1 == 0:
                            try:
                                filled_series = s_num.fillna(int(median_val))
                            except Exception:
                                filled_series = s_num.astype(float).fillna(median_val)
                        else:
                            filled_series = s_num.astype(float).fillna(median_val)
                        cleaned_df[col] = filled_series
                        stats["cells_filled"] += missing_count
                        stats["filled_columns"].append(str(col))
        return cleaned_df, stats

    elif strategy == "Fill with mode":
        for col in cleaned_df.columns:
            series = cleaned_df[col]
            missing_count = int(series.isnull().sum())
            if missing_count > 0:
                modes = series.dropna().mode()
                if len(modes) > 0:
                    # Deterministic tie-breaking: pick first sorted mode
                    try:
                        sorted_modes = sorted(list(modes))
                        mode_val = sorted_modes[0]
                    except Exception:
                        mode_val = modes.iloc[0]
                    cleaned_df[col] = series.fillna(mode_val)
                    stats["cells_filled"] += missing_count
                    stats["filled_columns"].append(str(col))
        return cleaned_df, stats

    else:
        return cleaned_df, stats


def clean_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool,
    missing_strategy: str,
    custom_markers: set | list = None,
) -> tuple:
    """
    Cleans a dataset by normalizing types, optionally removing duplicate rows,
    and applying a missing-value strategy.
    Never modifies the original DataFrame.

    Parameters:
        df: Input pandas DataFrame.
        remove_duplicates: Whether to remove duplicate rows.
        missing_strategy: Missing-value strategy name string.
        custom_markers: Optional list or set of custom missing markers.

    Returns:
        tuple: (cleaned_df: pd.DataFrame, summary: dict)
    """
    initial_rows = len(df)
    initial_columns = len(df.columns)

    # Step 1: Normalize types and missing-value markers on a fresh copy
    norm_df, norm_meta = normalize_dataframe(df, custom_markers=custom_markers)

    # Step 2: Duplicate row removal
    if remove_duplicates:
        df_step2, duplicate_rows_removed = remove_duplicate_rows(norm_df)
    else:
        df_step2, duplicate_rows_removed = norm_df.copy(), 0

    # Step 3: Missing-value strategy handling
    cleaned_df, missing_stats = handle_missing_values(df_step2, missing_strategy)

    missing_rows_removed = missing_stats["rows_removed"]
    missing_cells_filled = missing_stats["cells_filled"]
    filled_columns = missing_stats["filled_columns"]

    final_rows = len(cleaned_df)
    final_columns = len(cleaned_df.columns)
    total_rows_removed = initial_rows - final_rows

    missing_cells_remaining = int(cleaned_df.isnull().sum().sum())
    missing_markers_recognized = norm_meta.get("total_missing_markers_recognized", 0)

    # Combine all columns modified during normalization and cleaning
    all_changed_cols = set(norm_meta.get("columns_changed", [])) | set(filled_columns)
    columns_changed = sorted(list(all_changed_cols))

    changes_made = (
        (total_rows_removed > 0)
        or (missing_cells_filled > 0)
        or (missing_markers_recognized > 0)
        or (len(norm_meta.get("columns_changed", [])) > 0)
    )

    summary = {
        "initial_rows": initial_rows,
        "initial_columns": initial_columns,
        "final_rows": final_rows,
        "final_columns": final_columns,
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_rows_removed": missing_rows_removed,
        "total_rows_removed": total_rows_removed,
        "missing_cells_filled": missing_cells_filled,
        "missing_cells_remaining": missing_cells_remaining,
        "missing_markers_recognized": missing_markers_recognized,
        "columns_changed": columns_changed,
        "filled_columns": filled_columns,
        "duplicate_removal_applied": remove_duplicates,
        "missing_strategy_applied": missing_strategy,
        "changes_made": changes_made,
    }

    return cleaned_df, summary


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Converts a pandas DataFrame into a UTF-8 encoded CSV byte string without the pandas index.
    Safely handles empty DataFrames and header-only DataFrames.

    Parameters:
        df: Input pandas DataFrame.

    Returns:
        bytes: UTF-8 encoded CSV content.
    """
    if df is None:
        return b""
    return df.to_csv(index=False).encode("utf-8")
