import os
import shutil
from IPython import get_ipython
import pandas as pd


def delete_symbol_csvs(files_paths: dict):
    for path in files_paths.values():
        if os.path.exists(path):
            os.remove(path)


def clear_pycache(start_path="."):
    removed = []
    for root, dirs, files in os.walk(start_path):
        if "__pycache__" in dirs:
            path = os.path.join(root, "__pycache__")
            shutil.rmtree(path)
            removed.append(path)
    print(f"🧹 Removed {len(removed)} __pycache__ directories.")
    return removed


def setup_autoreload():
    ipython = get_ipython()
    if ipython:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
        print("🔁 Autoreload is enabled.")
    else:
        print("⚠️ Not running inside an IPython environment.")


def unnest_conf_indicators(df):
    """
    Unnest the conf_indicators column into separate columns.
    """
    df_copy = df.copy()

    # Initialize new columns
    df_copy["adx_slope"] = None
    df_copy["adx_value"] = None
    df_copy["di_slopes"] = None
    df_copy["di_plus"] = None
    df_copy["di_minus"] = None

    df_copy["volume_slope"] = None
    df_copy["volume_value"] = None
    df_copy["volume_avg"] = None
    df_copy["rsi_ratio"] = None
    df_copy["rsi_value"] = None
    df_copy["rsi_trend_reverse_slope_long"] = None
    df_copy["rsi_trend_reverse_slope_value"] = None
    df_copy["rsi_mean"] = None
    df_copy["rsi_trend_reverse"] = None
    """
    df_copy["macd_line"] = None
    df_copy["macd_signal"] = None
    df_copy["macd_line_prev"] = None
    df_copy["macd_signal_prev"] = None
    df_copy["macd_direction"] = None
    df_copy["bollinger_price"] = None
    df_copy["bollinger_middle"] = None
    df_copy["bollinger_lower"] = None
    df_copy["bollinger_upper"] = None
    df_copy["bollinger_direction"] = None
    """
    df_copy["atr_avg"] = None
    df_copy["atr_value"] = None
    df_copy["atr_slope"] = None
    df_copy["atr_stop_loss_to_price_ratio"] = None

    for idx, row in df_copy.iterrows():
        conf_indicators = row["conf_indicators"]

        if conf_indicators == "no_confirmation_needed":
            df_copy.at[idx, "adx_slope"] = "N/A"
            df_copy.at[idx, "adx_value"] = "N/A"
            df_copy.at[idx, "di_slopes"] = "N/A"
            df_copy.at[idx, "di_plus"] = "N/A"
            df_copy.at[idx, "di_minus"] = "N/A"
            df_copy.at[idx, "volume_slope"] = "N/A"
            df_copy.at[idx, "volume_value"] = "N/A"
            df_copy.at[idx, "volume_avg"] = "N/A"
            df_copy.at[idx, "rsi_ratio"] = "N/A"
            df_copy.at[idx, "rsi_value"] = "N/A"
            df_copy.at[idx, "rsi_trend_reverse_slope_long"] = "N/A"
            df_copy.at[idx, "rsi_trend_reverse_slope_value"] = "N/A"
            df_copy.at[idx, "rsi_mean"] = "N/A"
            df_copy.at[idx, "rsi_trend_reverse"] = "N/A"
            """
            df_copy.at[idx, "macd_line"] = "N/A"
            df_copy.at[idx, "macd_signal"] = "N/A"
            df_copy.at[idx, "macd_line_prev"] = "N/A"
            df_copy.at[idx, "macd_signal_prev"] = "N/A"
            df_copy.at[idx, "macd_direction"] = "N/A"
            df_copy.at[idx, "bollinger_price"] = "N/A"
            df_copy.at[idx, "bollinger_middle"] = "N/A"
            df_copy.at[idx, "bollinger_lower"] = "N/A"
            df_copy.at[idx, "bollinger_upper"] = "N/A"
            df_copy.at[idx, "bollinger_direction"] = "N/A"
            """
            df_copy.at[idx, "atr_avg"] = "N/A"
            df_copy.at[idx, "atr_value"] = "N/A"
            df_copy.at[idx, "atr_slope"] = "N/A"
            df_copy.at[idx, "atr_stop_loss_to_price_ratio"] = "N/A"

        elif isinstance(conf_indicators, dict):
            adx_data = conf_indicators["adx"]
            if isinstance(adx_data, tuple) and len(adx_data) == 2:
                df_copy.at[idx, "adx_slope"] = adx_data[0]
                df_copy.at[idx, "adx_value"] = adx_data[1]

            di_data = conf_indicators["di"]
            if isinstance(di_data, tuple) and len(di_data) >= 3:
                df_copy.at[idx, "di_slopes"] = di_data[0]
                df_copy.at[idx, "di_plus"] = di_data[1]
                df_copy.at[idx, "di_minus"] = di_data[2]

            volume_data = conf_indicators["volume"]
            if isinstance(volume_data, tuple) and len(volume_data) == 2:
                df_copy.at[idx, "volume_slope"] = volume_data[0]
                df_copy.at[idx, "volume_value"] = volume_data[1][0]
                df_copy.at[idx, "volume_avg"] = volume_data[1][1]
            rsi_data = conf_indicators["rsi"]
            if isinstance(rsi_data, tuple) and len(rsi_data) == 2:
                df_copy.at[idx, "rsi_ratio"] = rsi_data[0]
                df_copy.at[idx, "rsi_value"] = rsi_data[1]
            rsi_trend_reverse_slope_data = conf_indicators.get(
                "rsi_trend_reverse_slope", (None, None)
            )
            if (
                isinstance(rsi_trend_reverse_slope_data, tuple)
                and len(rsi_trend_reverse_slope_data) == 2
            ):
                df_copy.at[idx, "rsi_trend_reverse_slope_long"] = (
                    rsi_trend_reverse_slope_data[0]
                )
                df_copy.at[idx, "rsi_trend_reverse_slope_value"] = (
                    rsi_trend_reverse_slope_data[1]
                )
            rsi_trend_reverse_over_data = conf_indicators.get(
                "rsi_trend_reverse_over", (None, None)
            )
            if (
                isinstance(rsi_trend_reverse_over_data, tuple)
                and len(rsi_trend_reverse_over_data) == 2
            ):
                df_copy.at[idx, "rsi_mean"] = (
                    rsi_trend_reverse_over_data[0]
                )
                df_copy.at[idx, "rsi_trend_reverse"] = rsi_trend_reverse_over_data[1]
            
            """
            macd_data = conf_indicators["macd"]
            if isinstance(macd_data, tuple) and len(macd_data) == 2:
                df_copy.at[idx, "macd_line"] = macd_data[0][0]
                df_copy.at[idx, "macd_signal"] = macd_data[0][1]
                df_copy.at[idx, "macd_line_prev"] = macd_data[0][2]
                df_copy.at[idx, "macd_signal_prev"] = macd_data[0][3]
                df_copy.at[idx, "macd_direction"] = macd_data[1]

            bollinger_data = conf_indicators["bollinger"]
            if isinstance(bollinger_data, tuple) and len(bollinger_data) == 2:
                df_copy.at[idx, "bollinger_price"] = bollinger_data[0][0]
                df_copy.at[idx, "bollinger_middle"] = bollinger_data[0][1]
                df_copy.at[idx, "bollinger_lower"] = bollinger_data[0][2]
                df_copy.at[idx, "bollinger_upper"] = bollinger_data[0][3]
                df_copy.at[idx, "bollinger_direction"] = bollinger_data[1]
            """
            
            atr_data = conf_indicators["atr"]
            if isinstance(atr_data, tuple) and len(atr_data) == 2:
                df_copy.at[idx, "atr_avg"] = atr_data[0]
                df_copy.at[idx, "atr_value"] = atr_data[1][0]
                df_copy.at[idx, "atr_slope"] = atr_data[1][1]
                df_copy.at[idx, "atr_stop_loss_to_price_ratio"] = atr_data[1][2]

    df_copy = df_copy.drop(columns=["conf_indicators"])
    return df_copy


def safe_ratio_calc(row):
    """Calculate ratio while handling N/A values"""
    di_plus = row["di_plus"]
    di_minus = row["di_minus"]

    if di_plus == "N/A" or di_minus == "N/A":
        return None

    try:
        return float(di_plus) / float(di_minus)
    except (ValueError, ZeroDivisionError):
        return None


def save_df_to_csv(performance_df, file_path, append=True):
    """
    Save performance summary DataFrame to CSV with optional append functionality.

    Parameters:
    -----------
    performance_df : pd.DataFrame
        Performance summary DataFrame to save
    file_path : str
        Path to the CSV file
    append : bool, default True
        Whether to append to existing file or overwrite

    Returns:
    --------
    bool
        True if successful, raises exception if error

    Raises:
    -------
    ValueError
        If columns don't match when appending to existing file
    """
    import os

    try:
        # Check if file exists and we want to append
        if append and os.path.exists(file_path):
            # Read existing file to check column compatibility
            existing_df = pd.read_csv(file_path)

            # Check if columns match
            existing_cols = set(existing_df.columns)
            new_cols = set(performance_df.columns)

            if existing_cols != new_cols:
                missing_in_existing = new_cols - existing_cols
                missing_in_new = existing_cols - new_cols

                error_msg = "Column mismatch detected:\n"
                if missing_in_existing:
                    error_msg += f"  New columns not in existing file: {list(missing_in_existing)}\n"
                if missing_in_new:
                    error_msg += (
                        f"  Existing columns not in new data: {list(missing_in_new)}\n"
                    )

                raise ValueError(error_msg)

            # Columns match, append the data
            performance_df.to_csv(file_path, mode="a", header=False, index=False)
            print(f"Successfully appended {len(performance_df)} rows to {file_path}")

        else:
            # Create new file or overwrite existing
            performance_df.to_csv(file_path, index=False)
            mode_msg = "overwritten" if os.path.exists(file_path) else "created"
            print(
                f"Successfully {mode_msg} {file_path} with {len(performance_df)} rows"
            )

        return True

    except ValueError as e:
        # Re-raise column mismatch errors
        raise e
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")
        raise e


def save_df_with_metadata(performance_df, file_path, metadata_dict=None, append=True):
    """
    Enhanced version that adds metadata columns before saving.

    Parameters:
    -----------
    performance_df : pd.DataFrame
        Performance summary DataFrame to save
    file_path : str
        Path to the CSV file
    metadata_dict : dict, optional
        Additional metadata to add as columns (e.g., {'run_date': '2025-01-15', 'strategy_version': 'v1.2'})
    append : bool, default True
        Whether to append to existing file or overwrite
    """

    # Create a copy to avoid modifying original
    df_to_save = performance_df.copy()

    # Add metadata columns if provided
    if metadata_dict:
        for key, value in metadata_dict.items():
            df_to_save[key] = value

    # Save using the base function
    return save_df_to_csv(df_to_save, file_path, append)
