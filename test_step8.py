import io
import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
from streamlit.testing.v1 import AppTest
from analyzer import (
    read_csv_file,
    detect_duplicate_columns,
    get_column_info,
    get_missing_value_summary,
    count_duplicate_rows,
    get_data_quality_summary,
    get_numeric_statistics,
    get_numeric_columns,
    remove_duplicate_rows,
    handle_missing_values,
    clean_dataset,
    export_dataframe_to_csv,
)

APP_PATH = os.path.join(os.getcwd(), "app.py")

def test_analyzer_functions():
    # Basic DataFrame
    df = pd.DataFrame({
        "A": [1, 2, 2, None],
        "B": ["x", "y", "y", "z"],
        "C": [10.5, 20.0, 20.0, 40.5]
    })
    
    # Missing summary
    miss = get_missing_value_summary(df)
    assert len(miss) == 3
    assert miss.loc[miss["Column Name"] == "A", "Missing-Value Count"].values[0] == 1
    
    # Duplicate rows
    dups = count_duplicate_rows(df)
    assert dups == 1
    
    # Data quality summary
    dq = get_data_quality_summary(df)
    assert dq["total_missing"] == 1
    assert dq["duplicate_rows"] == 1
    assert dq["columns_with_missing"] == 1
    
    # Numeric stats
    stats = get_numeric_statistics(df)
    assert "A" in stats["Column Name"].values
    assert "C" in stats["Column Name"].values
    assert "B" not in stats["Column Name"].values
    
    # Clean dataset
    cleaned, summary = clean_dataset(df, remove_duplicates=True, missing_strategy="Fill numeric with mean")
    assert summary["duplicate_rows_removed"] == 1
    assert summary["missing_cells_filled"] == 1
    assert len(cleaned) == 3
    
    # CSV export
    csv_bytes = export_dataframe_to_csv(cleaned)
    assert isinstance(csv_bytes, bytes)
    assert b"A,B,C" in csv_bytes
    print("Analyzer unit tests passed successfully.")

def test_app_ui_ux():
    sample_csv = "Name,Age,Salary\nAlice,30,70000\nBob,25,50000\nBob,25,50000\nCharlie,,65000\nDavid,40,\n"
    csv_bytes = sample_csv.encode("utf-8")
    
    # 1. Test initial load (no file uploaded)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception on initial run: {at.exception}"
    assert len(at.info) > 0, "Welcome info message should be displayed"
    assert "Welcome" in at.info[0].value
    
    # 2. Upload file in sidebar
    at.sidebar.file_uploader[0].upload("employees.csv", csv_bytes)
    at.run()
    assert not at.exception, f"App raised exception on upload: {at.exception}"
    
    # Verify sidebar active dataset info
    sidebar_text = " ".join([m.value for m in at.sidebar.markdown] + [w.value for w in at.sidebar.caption if hasattr(w, 'value')])
    # Check that KPI metrics rendered
    metric_labels = [m.label for m in at.metric]
    assert "Total Rows" in metric_labels
    assert "Total Columns" in metric_labels
    assert "Missing Values" in metric_labels
    assert "Duplicate Rows" in metric_labels
    
    # Check tabs exist
    tab_labels = [t.label for t in at.tabs]
    expected_tabs = ["Overview", "Data Quality", "Statistics", "Visualizations", "Data Cleaning & Export"]
    for tab in expected_tabs:
        assert tab in tab_labels, f"Expected tab '{tab}' not found in {tab_labels}"
        
    print("Tabs and Executive KPIs verified successfully.")

def test_chart_rendering():
    sample_csv = "Category,Score,Sales\nA,10,100\nB,20,200\nA,30,300\nC,40,400\n"
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.file_uploader[0].upload("sales.csv", sample_csv.encode("utf-8"))
    at.run()
    assert not at.exception
    
    # Default chart type is Bar Chart
    charts = at.get("plotly_chart")
    assert len(charts) >= 1, "Bar chart should be rendered"
    
    # Switch to Histogram
    chart_selector = [s for s in at.selectbox if s.label == "Select Chart Type"][0]
    chart_selector.select("Histogram").run()
    assert not at.exception
    charts = at.get("plotly_chart")
    assert len(charts) >= 1, "Histogram should be rendered"
    
    # Switch to Scatter Plot
    chart_selector = [s for s in at.selectbox if s.label == "Select Chart Type"][0]
    chart_selector.select("Scatter Plot").run()
    assert not at.exception
    charts = at.get("plotly_chart")
    assert len(charts) >= 1, "Scatter plot should be rendered"
    
    print("Interactive chart rendering verified successfully.")

def test_cleaning_and_export():
    sample_csv = "Name,Age,Salary\nAlice,30,70000\nBob,25,50000\nBob,25,50000\nCharlie,,65000\n"
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.file_uploader[0].upload("team.csv", sample_csv.encode("utf-8"))
    at.run()
    assert not at.exception
    
    # Find Clean Dataset button
    clean_buttons = [b for b in at.button if b.label == "Clean Dataset"]
    assert len(clean_buttons) == 1
    
    # Trigger Clean Dataset
    clean_buttons[0].click().run()
    assert not at.exception
    
    # Cleaned DataFrame should be in session_state
    assert at.session_state["cleaned_df"] is not None
    assert len(at.session_state["cleaned_df"]) == 3 # 1 duplicate removed
    
    # Download button should be rendered
    download_buttons = [b for b in at.download_button if "Download Cleaned CSV" in b.label]
    assert len(download_buttons) == 1, "Download Cleaned CSV button should be rendered"
    
    # Switch chart selectbox to simulate user moving to another tab and changing controls
    chart_selector = [s for s in at.selectbox if s.label == "Select Chart Type"][0]
    chart_selector.select("Histogram").run()
    assert not at.exception
    # Verify cleaned_df persists across reruns
    assert at.session_state["cleaned_df"] is not None
    assert len(at.session_state["cleaned_df"]) == 3
    
    # Uploading a new file should reset cleaned session state
    new_csv = "X,Y\n1,2\n3,4\n"
    at.sidebar.file_uploader[0].upload("other.csv", new_csv.encode("utf-8"))
    at.run()
    assert not at.exception
    assert at.session_state["cleaned_df"] is None, "cleaned_df should be reset on new file upload"
    
    print("Data cleaning, export, and session state persistence verified successfully.")

if __name__ == "__main__":
    test_analyzer_functions()
    test_app_ui_ux()
    test_chart_rendering()
    test_cleaning_and_export()
    print("ALL STEP 8 VERIFICATION TESTS PASSED SUCCESSFULLY!")
