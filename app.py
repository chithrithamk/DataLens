"""
app.py - DataLens: CSV Data Analyzer

A beginner-friendly Streamlit web application for uploading, exploring,
analyzing, visualizing, cleaning, and exporting CSV datasets.
"""

import numpy as np
import plotly.express as px
import streamlit as st
from analyzer import (
    read_csv_file,
    detect_duplicate_columns,
    normalize_dataframe,
    get_column_info,
    get_missing_value_summary,
    get_data_quality_summary,
    get_numeric_statistics,
    get_non_numeric_statistics,
    get_numeric_columns,
    clean_dataset,
    export_dataframe_to_csv,
    MalformedCSVError,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataLens — CSV Data Analyzer",
    page_icon="🔍",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 2. Sidebar: File Upload, Settings, Dataset Info, and Navigation Guide
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ DataLens Controls")

    uploaded_file = st.file_uploader(
        "Choose a CSV file to upload",
        type=["csv"],
        help="Select a CSV (comma-separated values) file from your computer.",
    )

    with st.expander("🛠️ Advanced Settings", expanded=False):
        safe_recovery_opt = st.checkbox(
            "Enable Safe CSV Recovery",
            value=False,
            help="When enabled, safely recovers minor CSV formatting inconsistencies (such as extra trailing commas) without skipping rows. Every adjusted row and any omitted fields are explicitly reported.",
        )
        custom_marker_input = st.text_input(
            "Additional Missing Value Markers",
            value="-",
            help="Comma-separated tokens to treat as missing values (e.g. -, ?, N/A). Standard markers like NA, NULL, and whitespace are always recognized automatically.",
        )
        custom_markers = [
            m.strip() for m in custom_marker_input.split(",") if m.strip()
        ]

    st.divider()
    st.markdown("#### 💡 Quick Instructions")
    st.markdown(
        """
        1. **Upload** a CSV file using the file selector above.
        2. **Overview**: View dimensions, first 10 rows, and column types.
        3. **Data Quality**: Audit missing values, markers, and duplicate rows.
        4. **Statistics**: Explore numerical and non-numerical distributions.
        5. **Visualizations**: Generate bar charts, histograms, and scatter plots.
        6. **Cleaning & Export**: Clean missing values, remove duplicates, and download the cleaned CSV.
        """
    )

# -----------------------------------------------------------------------------
# 3. Main Dashboard Header
# -----------------------------------------------------------------------------
st.title("DataLens")
st.subheader("Upload and explore your CSV datasets with ease")

# If no file is uploaded, display a welcoming onboarding message in the main area
if uploaded_file is None:
    # Reset session state when no file is present
    st.session_state["last_cleaned_file"] = None
    st.session_state["cleaned_df"] = None
    st.session_state["cleaning_summary"] = None

    st.info(
        "👋 **Welcome! To get started:**\n\n"
        "1. Click **Browse files** or drag and drop a `.csv` file into the upload box in the sidebar on the left.\n"
        "2. Ensure your CSV file includes a header row containing column names.\n"
        "3. DataLens will automatically load the data, normalize column types, detect missing values, and inspect data quality."
    )

# -----------------------------------------------------------------------------
# 4. CSV Processing and Tabbed Dashboard
# -----------------------------------------------------------------------------
else:
    # Check for duplicate column names in the raw CSV header
    duplicate_columns = detect_duplicate_columns(uploaded_file)
    if duplicate_columns:
        st.warning(
            f"⚠️ **Duplicate column names detected**: `{', '.join(duplicate_columns)}`.\n\n"
            "Pandas automatically added numerical suffixes (for example, `.1`) to distinguish "
            "the duplicate columns so they can be processed without losing data."
        )

    # Safely load the CSV file using our reusable analyzer function
    try:
        raw_df = read_csv_file(uploaded_file, allow_recovery=safe_recovery_opt)

        # If safe recovery was applied to inconsistent rows, display an explicit audit report
        if raw_df.attrs.get("recovery_report"):
            recovery_report = raw_df.attrs["recovery_report"]
            st.warning(
                f"⚠️ **Safe CSV Recovery Applied**: {len(recovery_report)} row(s) had inconsistent field counts "
                "and were safely recovered without skipping rows."
            )
            with st.expander(
                f"📋 View Safe Recovery Audit Log ({len(recovery_report)} affected rows)",
                expanded=False,
            ):
                for item in recovery_report[:20]:
                    lost_info = (
                        f" — *Omitted non-empty data: `{item['data_lost']}`*"
                        if item.get("data_lost")
                        else " — *No non-empty data lost*"
                    )
                    st.markdown(f"• **Line {item['line']}**: {item['action']}{lost_info}")
                if len(recovery_report) > 20:
                    st.markdown(f"*... and {len(recovery_report) - 20} more rows.*")

        # Reset cleaned dataset state if a different file is uploaded
        file_identifier = getattr(uploaded_file, "name", "uploaded_csv")
        if (
            "last_cleaned_file" not in st.session_state
            or st.session_state["last_cleaned_file"] != file_identifier
        ):
            st.session_state["last_cleaned_file"] = file_identifier
            st.session_state["cleaned_df"] = None
            st.session_state["cleaning_summary"] = None

        total_rows = len(raw_df)
        total_cols = len(raw_df.columns)

        # Normalize a copy of the dataset for consistent quality metrics, statistics, and charts
        df, norm_meta = normalize_dataframe(raw_df, custom_markers=custom_markers)

        # Update sidebar with active dataset information
        with st.sidebar:
            st.divider()
            st.markdown("#### 📄 Active Dataset")
            st.write(f"**Filename:** `{file_identifier}`")
            if hasattr(uploaded_file, "size"):
                size_kb = uploaded_file.size / 1024
                if size_kb < 1024:
                    st.write(f"**File Size:** {size_kb:.1f} KB")
                else:
                    st.write(f"**File Size:** {size_kb / 1024:.2f} MB")
            st.write(f"**Rows:** {total_rows:,}")
            st.write(f"**Columns:** {total_cols:,}")

        # Handle edge case: Header row exists, but there are zero data rows
        if total_rows == 0:
            st.warning(
                "⚠️ **No Data Rows Found**: The uploaded CSV contains column headers, "
                "but does not contain any data rows."
            )

        # Compute data quality metrics for executive dashboard
        dq_summary = get_data_quality_summary(df)

        # Top Executive KPI Metrics Bar
        st.markdown("### 📊 Dataset Overview Summary")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Rows", f"{total_rows:,}")
        with kpi2:
            st.metric("Total Columns", f"{total_cols:,}")
        with kpi3:
            st.metric("Missing Values", f"{dq_summary['total_missing']:,}")
        with kpi4:
            st.metric("Duplicate Rows", f"{dq_summary['duplicate_rows']:,}")

        st.divider()

        # ---------------------------------------------------------------------
        # Tabbed Layout Organization
        # ---------------------------------------------------------------------
        tab_overview, tab_quality, tab_stats, tab_viz, tab_clean = st.tabs(
            [
                "Overview",
                "Data Quality",
                "Statistics",
                "Visualizations",
                "Data Cleaning & Export",
            ]
        )

        # =====================================================================
        # TAB 1: OVERVIEW
        # =====================================================================
        with tab_overview:
            st.subheader("📊 Dataset Dimensions")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Rows", value=f"{total_rows:,}")
            with col2:
                st.metric(label="Total Columns", value=f"{total_cols:,}")
            with col3:
                st.metric(
                    label="Dimensions (Rows × Columns)",
                    value=f"{total_rows:,} × {total_cols:,}",
                )

            st.divider()
            st.subheader("👀 Data Preview (First 10 Rows)")
            if total_rows > 0:
                st.dataframe(raw_df.head(10), use_container_width=True)
            else:
                st.write("No data rows available to preview.")

            st.divider()
            st.subheader("📋 Column Names & Detected Data Types")
            column_summary = get_column_info(raw_df)
            st.dataframe(column_summary, use_container_width=True)

        # =====================================================================
        # TAB 2: DATA QUALITY
        # =====================================================================
        with tab_quality:
            st.subheader("🛡️ Data Quality Report")

            q_col1, q_col2, q_col3, q_col4 = st.columns(4)
            with q_col1:
                st.metric(
                    label="Total Missing Cells",
                    value=f"{dq_summary['total_missing']:,}",
                )
            with q_col2:
                st.metric(
                    label="Duplicate Rows",
                    value=f"{dq_summary['duplicate_rows']:,}",
                )
            with q_col3:
                st.metric(
                    label="Rows with Missing Values",
                    value=f"{dq_summary['rows_with_missing']:,}",
                )
            with q_col4:
                st.metric(
                    label="Columns with Missing Values",
                    value=f"{dq_summary['columns_with_missing']:,}",
                )

            # Contextual alerts
            if total_rows == 0:
                st.info("ℹ️ The uploaded file has zero data rows.")
            else:
                if dq_summary["total_missing"] == 0:
                    st.success("✅ The dataset contains no missing values.")
                else:
                    st.warning(
                        f"⚠️ Found {dq_summary['total_missing']:,} missing cell(s) "
                        f"across {dq_summary['columns_with_missing']} column(s) "
                        f"affecting {dq_summary['rows_with_missing']} row(s)."
                    )

                if dq_summary["duplicate_rows"] == 0:
                    st.success("✅ The dataset contains no duplicate rows.")
                else:
                    st.warning(
                        f"⚠️ Found {dq_summary['duplicate_rows']:,} completely duplicate row(s)."
                    )

                if norm_meta["total_missing_markers_recognized"] > 0:
                    st.info(
                        f"ℹ️ Recognized **{norm_meta['total_missing_markers_recognized']:,}** "
                        f"text/placeholder missing marker(s) during type normalization."
                    )

            st.markdown("##### Missing Values by Column")
            missing_summary_df = get_missing_value_summary(df)
            st.dataframe(missing_summary_df, use_container_width=True)

        # =====================================================================
        # TAB 3: STATISTICS
        # =====================================================================
        with tab_stats:
            st.subheader("📈 Numerical Statistics")

            numeric_stats = get_numeric_statistics(df)

            if len(numeric_stats) > 0 and total_rows > 0:
                st.markdown(
                    """
                    Descriptive statistics summarize the central tendency, dispersion, 
                    and distribution of numerical columns:
                    - **Count**: Number of valid, non-null observations.
                    - **Mean**: Average value of the column.
                    - **Median (50%)**: Middle value splitting the ordered data in half.
                    - **Standard Deviation**: Measure of how spread out values are around the mean.
                    - **Min / Max**: Smallest and largest recorded values.
                    - **25% / 75%**: 1st quartile (25th percentile) and 3rd quartile (75th percentile).
                    """
                )

                st.dataframe(
                    numeric_stats,
                    use_container_width=True,
                    column_config={
                        "Mean": st.column_config.NumberColumn(format="%.2f"),
                        "Median": st.column_config.NumberColumn(format="%.2f"),
                        "Standard Deviation": st.column_config.NumberColumn(format="%.2f"),
                        "Min": st.column_config.NumberColumn(format="%.2f"),
                        "25%": st.column_config.NumberColumn(format="%.2f"),
                        "50%": st.column_config.NumberColumn(format="%.2f"),
                        "75%": st.column_config.NumberColumn(format="%.2f"),
                        "Max": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
            else:
                if total_rows == 0:
                    st.info(
                        "ℹ️ The uploaded file has zero data rows. Descriptive statistics cannot be computed."
                    )
                else:
                    st.info(
                        "ℹ️ No numeric columns were found in this dataset. Descriptive statistics "
                        "are only calculated for numerical features (integer and floating-point values)."
                    )

            st.divider()
            st.subheader("🔤 Categorical & Non-Numeric Summary")
            non_numeric_stats = get_non_numeric_statistics(df)
            if len(non_numeric_stats) > 0 and total_rows > 0:
                st.dataframe(non_numeric_stats, use_container_width=True)
            else:
                if total_rows > 0:
                    st.info("ℹ️ All columns in this dataset are numeric.")

        # =====================================================================
        # TAB 4: VISUALIZATIONS
        # =====================================================================
        with tab_viz:
            st.subheader("📊 Interactive Visualizations")

            if total_rows == 0:
                st.info(
                    "ℹ️ The uploaded file has zero data rows. Visualizations cannot be generated."
                )
            else:
                chart_type = st.selectbox(
                    "Select Chart Type",
                    options=["Bar Chart", "Histogram", "Scatter Plot"],
                    help="Choose a chart type to visualize your dataset with Plotly.",
                )

                numeric_cols = get_numeric_columns(df)
                all_cols = [str(c) for c in df.columns]

                # -------------------------------------------------------------
                # 1. Bar Chart
                # -------------------------------------------------------------
                if chart_type == "Bar Chart":
                    st.markdown("##### 📊 Bar Chart Settings")
                    b_col1, b_col2 = st.columns(2)

                    cat_cols = [c for c in all_cols if c not in numeric_cols]
                    default_x_idx = all_cols.index(cat_cols[0]) if cat_cols else 0

                    with b_col1:
                        x_col = st.selectbox(
                            "Select X-axis (Categories/Grouping)",
                            options=all_cols,
                            index=default_x_idx,
                            help="Select the column to represent categories on the horizontal axis.",
                        )

                    with b_col2:
                        y_options = ["None (Count records)"] + numeric_cols
                        y_col_choice = st.selectbox(
                            "Select Y-axis (Optional Numeric Metric)",
                            options=y_options,
                            index=0,
                            help="Optionally select a numeric column to aggregate, or count rows per category.",
                        )

                    if y_col_choice != "None (Count records)":
                        agg_method = st.selectbox(
                            "Select Aggregation Method",
                            options=["Sum", "Mean", "Count"],
                            index=0,
                            help="Choose how numeric values are aggregated for each category.",
                        )

                        clean_bar_df = df[[x_col, y_col_choice]].dropna()
                        if clean_bar_df.empty:
                            st.info(
                                f"ℹ️ No valid (non-null) data points found for '{x_col}' and '{y_col_choice}'."
                            )
                        else:
                            agg_key = agg_method.lower()
                            agg_label = {
                                "sum": "Sum",
                                "mean": "Average",
                                "count": "Count",
                            }[agg_key]

                            grouped_df = (
                                clean_bar_df.groupby(x_col, as_index=False)[y_col_choice]
                                .agg(agg_key)
                                .rename(columns={y_col_choice: f"{agg_label} of {y_col_choice}"})
                            )

                            fig_bar = px.bar(
                                grouped_df,
                                x=x_col,
                                y=f"{agg_label} of {y_col_choice}",
                                title=f"{agg_label} of {y_col_choice} by {x_col}",
                                labels={
                                    x_col: str(x_col),
                                    f"{agg_label} of {y_col_choice}": f"{agg_label} of {y_col_choice}",
                                },
                                text_auto=".2f" if agg_key == "mean" else True,
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                    else:
                        clean_bar_df = df[[x_col]].dropna()
                        if clean_bar_df.empty:
                            st.info(f"ℹ️ All values in column '{x_col}' are missing.")
                        else:
                            counts_df = clean_bar_df[x_col].value_counts().reset_index()
                            counts_df.columns = [x_col, "Record Count"]

                            fig_bar = px.bar(
                                counts_df,
                                x=x_col,
                                y="Record Count",
                                title=f"Record Counts by {x_col}",
                                labels={x_col: str(x_col), "Record Count": "Number of Records"},
                                text_auto=True,
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                # -------------------------------------------------------------
                # 2. Histogram
                # -------------------------------------------------------------
                elif chart_type == "Histogram":
                    st.markdown("##### 📉 Histogram Settings")

                    if not numeric_cols:
                        st.info(
                            "ℹ️ Histograms require at least one numeric column. "
                            "No numeric columns were found in this dataset."
                        )
                    else:
                        h_col1, h_col2 = st.columns(2)
                        with h_col1:
                            hist_col = st.selectbox(
                                "Select Numeric Column",
                                options=numeric_cols,
                                help="Select the numeric feature whose distribution you want to inspect.",
                            )
                        with h_col2:
                            nbins = st.slider(
                                "Number of Bins",
                                min_value=5,
                                max_value=100,
                                value=20,
                                step=5,
                                help="Select the number of bins to group continuous values into.",
                            )

                        clean_hist_data = (
                            df[[hist_col]]
                            .replace([np.inf, -np.inf], np.nan)
                            .dropna()
                        )
                        if clean_hist_data.empty:
                            st.info(
                                f"ℹ️ Column '{hist_col}' contains no valid numeric values to display."
                            )
                        else:
                            fig_hist = px.histogram(
                                clean_hist_data,
                                x=hist_col,
                                nbins=nbins,
                                title=f"Distribution of {hist_col} ({nbins} Bins)",
                                labels={hist_col: str(hist_col), "count": "Frequency"},
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                # -------------------------------------------------------------
                # 3. Scatter Plot
                # -------------------------------------------------------------
                elif chart_type == "Scatter Plot":
                    st.markdown("##### 🔵 Scatter Plot Settings")

                    if len(numeric_cols) < 2:
                        st.info(
                            "ℹ️ A scatter plot requires at least two distinct numeric columns to plot X and Y coordinates. "
                            f"Found {len(numeric_cols)} numeric column(s) in this dataset."
                        )
                    else:
                        s_col1, s_col2, s_col3 = st.columns(3)
                        with s_col1:
                            x_scatter = st.selectbox(
                                "Select X-axis (Numeric)",
                                options=numeric_cols,
                                index=0,
                                help="Numeric column for horizontal coordinate.",
                            )
                        with s_col2:
                            default_y_idx = 1 if len(numeric_cols) > 1 else 0
                            y_scatter = st.selectbox(
                                "Select Y-axis (Numeric)",
                                options=numeric_cols,
                                index=default_y_idx,
                                help="Numeric column for vertical coordinate.",
                            )
                        with s_col3:
                            color_options = ["None"] + all_cols
                            color_col = st.selectbox(
                                "Color by (Optional Grouping)",
                                options=color_options,
                                index=0,
                                help="Optionally color markers using categories from another column.",
                            )

                        if x_scatter == y_scatter:
                            st.info(
                                f"ℹ️ Both X and Y axes are set to '{x_scatter}'. Data points will align along a diagonal line."
                            )

                        subset_cols = [x_scatter, y_scatter]
                        if color_col != "None":
                            subset_cols.append(color_col)

                        clean_scatter_df = (
                            df[subset_cols]
                            .replace([np.inf, -np.inf], np.nan)
                            .dropna()
                        )
                        if clean_scatter_df.empty:
                            st.info(
                                f"ℹ️ No valid (non-null) data points found for '{x_scatter}' and '{y_scatter}'."
                            )
                        else:
                            color_arg = color_col if color_col != "None" else None
                            chart_title = f"{y_scatter} vs. {x_scatter}"
                            if color_arg:
                                chart_title += f" (Grouped by {color_col})"

                            fig_scatter = px.scatter(
                                clean_scatter_df,
                                x=x_scatter,
                                y=y_scatter,
                                color=color_arg,
                                title=chart_title,
                                labels={x_scatter: str(x_scatter), y_scatter: str(y_scatter)},
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)

        # =====================================================================
        # TAB 5: DATA CLEANING & EXPORT
        # =====================================================================
        with tab_clean:
            st.subheader("🧹 Data Cleaning")

            if total_rows == 0:
                st.info(
                    "ℹ️ The uploaded file has zero data rows. Data cleaning cannot be applied."
                )
            else:
                st.markdown(
                    "Configure data cleaning options to normalize types, remove duplicates, and handle missing values."
                )

                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    remove_duplicates_opt = st.checkbox(
                        "Remove duplicate rows",
                        value=True,
                        help="Identifies and removes completely identical rows, retaining the first occurrence.",
                    )
                with c_col2:
                    missing_strategy_opt = st.selectbox(
                        "Missing Value Treatment",
                        options=[
                            "Keep missing values",
                            "Remove rows with missing values",
                            "Fill numeric with mean",
                            "Fill numeric with median",
                            "Fill with mode",
                        ],
                        index=0,
                        help="Choose how to handle missing values across the dataset.",
                    )

                clean_btn = st.button("Clean Dataset", type="primary")

                if clean_btn:
                    cleaned_data, summary = clean_dataset(
                        raw_df,
                        remove_duplicates=remove_duplicates_opt,
                        missing_strategy=missing_strategy_opt,
                        custom_markers=custom_markers,
                    )
                    st.session_state["cleaned_df"] = cleaned_data
                    st.session_state["cleaning_summary"] = summary

                # Display cleaning results if cleaning has been executed
                if (
                    st.session_state.get("cleaned_df") is not None
                    and st.session_state.get("cleaning_summary") is not None
                ):
                    summary = st.session_state["cleaning_summary"]
                    cleaned_data = st.session_state["cleaned_df"]

                    st.markdown("##### Cleaning Results")
                    if summary["changes_made"]:
                        st.success("✅ **Dataset successfully cleaned!**")
                    else:
                        st.info(
                            "ℹ️ **No cleaning changes were necessary.** "
                            "The dataset contains no duplicate rows or missing values matching the selected strategy."
                        )

                    # Summary metric cards
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Rows Removed", f"{summary['total_rows_removed']:,}")
                    with m2:
                        st.metric("Missing Cells Filled", f"{summary['missing_cells_filled']:,}")
                    with m3:
                        st.metric(
                            "Missing Cells Remaining",
                            f"{summary['missing_cells_remaining']:,}",
                        )
                    with m4:
                        st.metric(
                            "Final Dimensions",
                            f"{summary['final_rows']:,} × {summary['final_columns']:,}",
                        )

                    # Detailed notes
                    details = []
                    if summary["duplicate_rows_removed"] > 0:
                        details.append(
                            f"• **{summary['duplicate_rows_removed']:,}** completely duplicate row(s) removed."
                        )
                    if summary["missing_rows_removed"] > 0:
                        details.append(
                            f"• **{summary['missing_rows_removed']:,}** row(s) dropped due to missing values."
                        )
                    if summary["missing_cells_filled"] > 0:
                        details.append(
                            f"• **{summary['missing_cells_filled']:,}** missing cell(s) filled using `{summary['missing_strategy_applied']}`."
                        )
                    if summary["missing_markers_recognized"] > 0:
                        details.append(
                            f"• **{summary['missing_markers_recognized']:,}** missing marker token(s) identified and normalized."
                        )
                    if summary["columns_changed"]:
                        cols_str = ", ".join(f"`{c}`" for c in summary["columns_changed"][:5])
                        if len(summary["columns_changed"]) > 5:
                            cols_str += f" and {len(summary['columns_changed']) - 5} more"
                        details.append(
                            f"• Columns modified by normalization or imputation: {cols_str}."
                        )
                    if summary["missing_cells_remaining"] > 0:
                        details.append(
                            f"• *Note:* **{summary['missing_cells_remaining']:,}** missing cell(s) remain "
                            f"(columns with non-numeric data or all-missing values)."
                        )

                    if details:
                        st.markdown("\n".join(details))

                    st.markdown("##### Preview of Cleaned Dataset")
                    if len(cleaned_data) > 0:
                        st.dataframe(cleaned_data.head(10), use_container_width=True)
                    else:
                        st.write("Cleaned dataset contains 0 data rows.")

                    # Download Cleaned Dataset Sub-section
                    st.divider()
                    st.subheader("📥 Download Cleaned Dataset")

                    cleaned_export_df = st.session_state["cleaned_df"]
                    export_rows = len(cleaned_export_df)
                    export_cols = len(cleaned_export_df.columns)

                    st.success(
                        "✅ **Ready for download:** Your cleaned dataset is ready. "
                        "The file below contains all applied cleaning transformations without altering your original dataset."
                    )

                    st.write(
                        f"**Cleaned Dataset Dimensions:** {export_rows:,} rows × {export_cols:,} columns"
                    )

                    csv_export_bytes = export_dataframe_to_csv(cleaned_export_df)

                    st.download_button(
                        label="⬇️ Download Cleaned CSV",
                        data=csv_export_bytes,
                        file_name="datalens_cleaned_data.csv",
                        mime="text/csv",
                        help="Click to download your cleaned dataset as a CSV file.",
                    )

    except MalformedCSVError as err:
        st.error(f"❌ **{err}**")
        if err.diagnostics.get("is_recoverable"):
            st.info(
                "💡 **Safe Recovery Available**: The inconsistent fields in this file appear to be minor formatting issues (e.g. extra trailing empty commas). "
                "You can enable **'Enable Safe CSV Recovery'** under **Advanced Settings** in the sidebar to load this dataset safely without skipping rows. Every adjusted row will be explicitly reported."
            )
    except ValueError as err:
        # Gracefully display friendly error messages without exposing Python tracebacks
        st.error(f"❌ **Error processing CSV file:** {err}")
    except Exception as err:
        # Fallback for unexpected failures
        st.error(f"❌ **An unexpected error occurred while processing the file:** {err}")
