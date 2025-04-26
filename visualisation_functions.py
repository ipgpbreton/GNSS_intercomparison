
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def plot_time_series(
        df, 
        station, 
        variable,
        solution_name
):
    """
    Plot time series of observation values for a specific station and variable.

    Parameters:
    df : DataFrame
        The dataframe containing the data
    station : str 
        Station identifier for which to plot data (Capital letters on 9 characters).
    variable : str 
        Variable to plot ('zenith_total_delay', 'precipitable_water_column', 'precipitable_water_column_era5').
    solution_name : str
        Name of the solution ('IGSdaily', 'IGSrepro3', 'EPNrepro2') to be used in the plot title.

    Returns
    -------
    None
        Displays the plot.
    Notes
    -----
    This function assumes that the DataFrame has a multi-index with 'primary_station_id' and 'observed_variable'.
    The 'report_timestamp' column is expected to be in datetime format.
    This function gives a quick overview of the time series data for the specified station and variable.
    """

    plt.rcParams.update({"font.size": 16})

    try:
        selected_data = df.loc[(station, variable)].reset_index()
    except KeyError:
        print(f"Data for {station} with variable {variable} not found.")
        return

    fig, ax = plt.subplots(figsize=(25, 4))
    ax.scatter(
        selected_data["report_timestamp"],
        selected_data["observation_value"],
        label=f"Station {station}",
        s=3,
        color="blue",
        alpha=0.7,
    )

    if "era5" in variable:
        if "repro3" in solution_name:
            product_type = "ERA5 on IGSrepro3"
        elif "daily" in solution_name:
            product_type = "ERA5 on IGSdaily"
        elif "repro2" in solution_name:
            product_type = "ERA5 on EPNrepro2"
    else:
        if "repro3" in solution_name:
            product_type = "IGSrepro3"
        elif "daily" in solution_name:
            product_type = "IGSdaily"
        elif "repro2" in solution_name:
            product_type = "EPNrepro2"

    ax.set_title(f"Station: {station} - {product_type}", fontsize=18)
    if "precipitable_water_column" in variable:
        ylabel = "PWC [kg/m²]"
    elif variable == "zenith_total_delay":
        ylabel = "ZTD [m]"
    else:
        ylabel = "Value"

    ax.set_ylabel(ylabel, fontsize=16)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_diff_time_series(
    dfs_dict,
    station,
    variable,
    equipment_change,
    labels=None,
    ylim=[-30, 30],
    xlim=(pd.to_datetime("2000-01-01"), pd.to_datetime("2019-12-31")),
    save=False,
):
    """
    Plot time series data of differences of ZTD or PWC between various datasets.
    
    Parameters
    ----------
    dfs_dict : dict
        Dictionary containing dfs_dict for different datasets.
        Keys should be dataset names and values should be the corresponding dfs_dict.
    station : str
        Station identifier for which to plot data (Capital letters on 9 characters).
    variable : str
        Variable to plot ('PWC_diff', 'ZTD_diff', 'PWC_era5', 'PWC_daily', 'PWC_repro3', 'ZTD_daily', 'ZTD_repro3').
    equipment_change : DataFrame
        DataFrame containing equipment change information with columns 'Station', 'Change_Date', and 'Type'.
    labels : list, optional
        List of specific dataset labels to include in the plot.
    ylim : list, default=[-30, 30] (change for plotting ZTD)
        Y-axis limits for the plot.
    xlim : tuple, default=(2000-01-01, 2019-12-31 ; can go up to 2023-12-31 for REPRO3daily)
        X-axis limits for the plot as datetime objects.
    save : bool, default=False
        If True, saves the plot to the given file path.
        
    Returns
    -------
    None
        Displays the plot.
        
    Notes
    -----
    This function is designed to visualize differences in ZTD or PWC data
    between different datasets (e.g., IGSdaily, IGSrepro3, ERA5).
    It provides a quick overview of the differences and highlights equipment changes.
    The function assumes that the input DataFrames are already preprocessed and contain the necessary columns. (df_diff_pwc_daily_era5, df_diff_pwc_repro3_era5, df_diff_pwc_daily_repro3, df_diff_ztd_daily_repro3)
    """
    plt.rcParams.update({"font.size": 16})

    if labels:
        dfs_dict = {label: df for label, df in dfs_dict.items() if label in labels}

    fig = plt.figure(figsize=(25, 4))
    ax = plt.gca()
    for label, df in dfs_dict.items():
        df_station = df[df.index.get_level_values("Station") == station]
        variable_column = variable

        variable_stats_m = {
            "mean": df_station[variable_column].mean(),
            "std": df_station[variable_column].std(),
            "min": df_station[variable_column].min(),
            "max": df_station[variable_column].max(),
            "median": df_station[variable_column].median(),
            "IQR/1.35": (df_station[variable_column].quantile(0.75) - df_station[variable_column].quantile(0.25))/1.35,
            "Number of point out of median +/- 3*sigma": df_station[(
                    df_station[variable_column] > df_station[variable_column].median()+ 3* (df_station[variable_column].quantile(0.75) - df_station[variable_column].quantile(0.25))/ 1.35
                ) | (
                    df_station[variable_column] < df_station[variable_column].median()- 3* (df_station[variable_column].quantile(0.75) - df_station[variable_column].quantile(0.25))/ 1.35
                )].shape[0],
            "Total number of points": len(df_station[variable_column]),
        }

        stats_text = f"in [{'kg/m2' if 'PWC' in variable else 'm'}] : Median = {variable_stats_m['median']:.3f}, IQR/1.35 = {variable_stats_m['IQR/1.35']:.3f}, Mean = {variable_stats_m['mean']:.3f}, Std = {variable_stats_m['std']:.3f}, NP : {variable_stats_m['Total number of points']}, NP>3*sigma : {variable_stats_m['Number of point out of median +/- 3*sigma']} ({(variable_stats_m['Number of point out of median +/- 3*sigma'] / len(df_station[variable])) * 100:.2f}%), min = {variable_stats_m['min']:.3f}, max = {variable_stats_m['max']:.3f}\n"

        ax.text(
            0.02,
            0.09,
            stats_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

        median = df_station[variable_column].median()
        iqr = df_station[variable_column].quantile(0.75) - df_station[variable_column].quantile(0.25)
        sigma = iqr / 1.35
        lower_bound = median - 3 * sigma
        upper_bound = median + 3 * sigma

        ax.axhline(
            y=lower_bound,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label="+/- 3*sigma",
        )
        ax.axhline(y=upper_bound, color="orange", linestyle="--", linewidth=1.5)
        ax.axhline(y=median, color="red", linestyle="-", linewidth=1.5, label="Median")

        if df_station.empty:
            print(f"No data available for station {station} in {label}.")
            continue
        color = "blue"
        plt.scatter(df_station["Date"], df_station[variable_column], s=1, color=color)

    if not equipment_change.empty:
        for date in equipment_change[equipment_change["Station"] == station[:4].lower()][
            "Change_Date"
        ]:
            # Get the equipment type for this date
            eq_type = equipment_change[
                (equipment_change["Station"] == station[:4].lower())
                & (equipment_change["Change_Date"] == date)
            ]["Type"].values[0]
            if eq_type == "A":
                color = "green"
                label_text = "Antenna Changing Date"
            elif eq_type == "R":
                color = "magenta"
                label_text = "Receiver Changing Date"
            elif eq_type == "RA":
                color = "red"
                label_text = "Receiver+Antenna Changing Date"
            else:
                color = "gray"
                label_text = "Unknown Equipment Change"

            # Only add label for the first occurrence of each type
            label = (
                label_text
                if date == equipment_change[
                    (equipment_change["Station"] == station[:4].lower())
                    & (equipment_change["Type"] == eq_type)
                ]["Change_Date"].iloc[0]
                else ""
            )
            ax.axvline(
                pd.to_datetime(date), color=color, linestyle="--", alpha=0.8, label=label
            )

    plt.title(f"Station {station} - {labels[0][5:-1]}")
    plt.ylabel(f'{labels[0][:4]} [{"kg/m2" if "PWC" in variable else "m"}]')
    plt.grid(axis="y")

    plt.ylim(ylim)
    plt.xlim(xlim)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    years = [
        tick
        for tick in ax.get_xticks()
        if tick >= ax.get_xlim()[0] and tick <= ax.get_xlim()[1]
    ]
    ax.set_xticks(years)

    ax.tick_params(axis="x", which="major", labelsize=16, rotation=0)

    handles, labels = [], []
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

    fig.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.65, 1.17),
        frameon=False,
    )

    plt.tight_layout()
    if save == True:
        plt.savefig(f"/home/hbreton/Documents/Script/Plots/era5_comparison/{label}/{station}_{variable}.png", dpi=600, bbox_inches='tight')
    plt.show()

def plot_scatter_joint_stats(
    station_stats,
    equipment_change,
    dataframes_dict,
    datatype="PWC",
    x_col="mean",
    y_col="std",
    title="IGSrepro3 - ERA5",
    xlabel="Mean[ΔPWC] [kg/m²]",
    ylabel="STD[ΔPWC] [kg/m²]",
    xlim=(pd.to_datetime("2000-01-01"), pd.to_datetime("2019-12-31")),
    ylim=[-20, 20],
    figsize=(10, 6),
    annotate_worst=True,
    n_worst=5,
    annotate_threshold_worst=False,
    threshold_mean=(2, 2),
    threshold_std=(2),
    show_grid=True,
    show_stats=True,
    save=False,
    save_path=None,
    dpi=600,
):
    """

    Parameters:
    -----------
    station_stats : pandas.DataFrame
        DataFrame containing station statistics with columns for mean, std, etc.
    datatype : str, default='PWC'
        Type of data to plot ('PWC' or 'ZTD')
    x_col : str, default='mean'
        Column name to use for x-axis
    y_col : str, default='std'
        Column name to use for y-axis
    title : str, default='IGSrepro3 - ERA5'
        Plot title
    xlabel : str, default='Mean[ΔPWC] [kg/m²]'
        Label for x-axis
    ylabel : str, default='STD[ΔPWC] [kg/m²]'
        Label for y-axis
    xlim : tuple, default=(pd.to_datetime("2000-01-01"), pd.to_datetime("2019-12-31"))
        Time range limits for time series plots
    ylim : list, default=[-20, 20]
        Y-axis limits for time series plots
    figsize : tuple, default=(10, 6)
        Figure size
    annotate_worst : bool, default=True
        Whether to annotate the worst stations
    n_worst : int, default=5
        Number of worst stations to annotate
    annotate_threshold_worst : bool, default=False
        Whether to annotate stations exceeding threshold values
    threshold_mean : tuple, default=(2, 2)
        Thresholds for mean values (lower, upper)
    threshold_std : float, default=2
        Threshold for standard deviation
    show_grid : bool, default=True
        Whether to show grid on the plot
    show_stats : bool, default=True
        Whether to show statistics text
    save : bool, default=False
        Whether to save the figure
    save_path : str, default=None
        If provided, save the figure to this path
    dpi : int, default=600
        DPI for saved figure

    Returns:
    --------
    g : seaborn.JointGrid
        The joint grid object containing the plot

    Notes:
    -----
    This function creates a scatter plot with marginal histograms for the given station statistics.
    It also annotates the worst stations based on the specified statistical criteria (thresholds on mean and std or worst station on mean and std).
    This function automatically plots the time series for the worst stations.
    """
    fig = plt.figure(figsize=figsize)

    g = sns.JointGrid(
        data=station_stats, x=x_col, y=y_col, height=9, ratio=4, marginal_ticks=True
    )

    g.plot_joint(sns.scatterplot)
    g.plot_marginals(sns.histplot, multiple="stack", kde=False, bins=100)
    g.ax_marg_x.grid(True, which="both", linestyle="--", alpha=0.5)
    g.ax_marg_y.grid(True, which="both", linestyle="--", alpha=0.5)
    g.ax_marg_x.set_xticks(g.ax_joint.get_xticks())
    g.ax_marg_y.set_yticks(g.ax_joint.get_yticks())

    if annotate_worst:
        worst_stats_by_std = station_stats.nlargest(n_worst, y_col)
        for i, row in worst_stats_by_std.iterrows():
            g.ax_joint.text(
                row[x_col], row[y_col], row["Station"], fontsize=9, ha="right"
            )

        worst_stats_by_mean = station_stats.iloc[
            station_stats[x_col].abs().nlargest(n_worst).index
        ]
        for i, row in worst_stats_by_mean.iterrows():
            g.ax_joint.text(
                row[x_col], row[y_col], row["Station"], fontsize=9, ha="right"
            )

    if annotate_threshold_worst:
        worst_stats_by_threshold_mean = station_stats[
            (
                (station_stats[x_col] < threshold_mean[0])
                | (station_stats[x_col] > threshold_mean[1])
            )
        ]
        for i, row in worst_stats_by_threshold_mean.iterrows():
            g.ax_joint.text(
                row[x_col], row[y_col], row["Station"], fontsize=9, ha="right"
            )

        worst_stats_by_threshold_std = station_stats[
            (station_stats[y_col] > threshold_std)
        ]
        for i, row in worst_stats_by_threshold_std.iterrows():
            g.ax_joint.text(
                row[x_col], row[y_col], row["Station"], fontsize=9, ha="right"
            )

    g.ax_joint.axvline(x=0, color="red", linestyle="-", linewidth=2)

    for q in np.quantile(station_stats[x_col], [0.5]):
        for ax in (g.ax_joint, g.ax_marg_x):
            ax.axvline(q, color="black", linestyle="--", label="median")

    for q in np.quantile(station_stats[y_col], [0.5]):
        for ax in (g.ax_joint, g.ax_marg_y):
            ax.axhline(q, color="black", linestyle="--")

    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles, labels, loc="lower left")

    if show_stats:
        std_stats = {
            "mean": station_stats[y_col].mean(),
            "std": station_stats[y_col].std(),
            "median": station_stats[y_col].median(),
            "IQR/1.35": (
                station_stats[y_col].quantile(0.75) - station_stats[y_col].quantile(0.25))/1.35,
        }

        mean_stats = {
            "mean": station_stats[x_col].mean(),
            "std": station_stats[x_col].std(),
            "median": station_stats[x_col].median(),
            "IQR/1.35": (
                station_stats[x_col].quantile(0.75) - station_stats[x_col].quantile(0.25))/1.35,
        }

        stats_text = (
            f"{xlabel} : mean = {mean_stats['mean']:.2f}, std = {mean_stats['std']:.2f}, "
            f"median = {mean_stats['median']:.2f}, IQR/1.35 = {mean_stats['IQR/1.35']:.2f}\n"
            f"{ylabel} : mean = {std_stats['mean']:.2f}, std = {std_stats['std']:.2f}, "
            f"median = {std_stats['median']:.2f}, IQR/1.35 = {std_stats['IQR/1.35']:.2f}\n"
        )

        g.fig.text(
            0.11,
            0.78,
            stats_text,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="left",
        )

    g.ax_joint.set_ylim(bottom=-0)
    g.fig.suptitle(title, y=1.03)
    g.set_axis_labels(xlabel, ylabel, fontsize=18)

    xticks = g.ax_joint.get_xticks()
    yticks = g.ax_joint.get_yticks()

    g.ax_joint.set_xticks(xticks)
    g.ax_joint.set_yticks(yticks)

    g.ax_joint.set_xticklabels([f"{x:.1f}" for x in xticks])
    g.ax_joint.set_yticklabels([f"{y:.1f}" for y in yticks])

    g.ax_marg_x.set_xticklabels([f"{x:.1f}" for x in g.ax_marg_x.get_xticks()])
    g.ax_marg_y.set_yticklabels([f"{y:.1f}" for y in g.ax_marg_y.get_yticks()])

    if show_grid:
        g.ax_joint.grid(True)

    if annotate_worst == True:
        if datatype == "PWC":
            label_mapping = {
                "IGSrepro3 - ERA5": "ΔPWC[IGSrepro3 - ERA5]",
                "IGSdaily - ERA5": "ΔPWC[IGSdaily - ERA5]",
                "EPNrepro2 - ERA5": "ΔPWC[EPNrepro2 - ERA5]",
                "IGSdaily - IGSrepro3": "ΔPWC[IGSdaily - IGSrepro3]",
                "IGSdaily - EPNrepro2": "ΔPWC[IGSdaily - EPNrepro2]",
                "IGSrepro3 - EPNrepro2": "ΔPWC[IGSrepro3 - EPNrepro2]",
            }
        else:
            label_mapping = {
                "IGSdaily - IGSrepro3": "ΔZTD[IGSdaily - IGSrepro3]",
                "IGSdaily - EPNrepro2": "ΔZTD[IGSdaily - EPNrepro2]",
                "IGSrepro3 - EPNrepro2": "ΔZTD[IGSrepro3 - EPNrepro2]",
            }

        time_series_label = label_mapping.get(title)

        if time_series_label:
            for i, row in worst_stats_by_std.iterrows():
                try:
                    plot_diff_time_series(
                        dataframes_dict,
                        row["Station"],
                        f"{datatype}_diff",
                        equipment_change=equipment_change,
                        labels=[time_series_label],
                        ylim=ylim,
                        xlim=xlim,
                        save=False,
                    )
                except Exception as e:
                    print(f"Error plotting {row['Station']}: {e}")

            for i, row in worst_stats_by_mean.iterrows():
                # Skip if already plotted in the std section
                if row["Station"] not in worst_stats_by_std["Station"].values:
                    try:
                        plot_diff_time_series(
                            dataframes_dict,
                            row["Station"],
                            f"{datatype}_diff",
                            equipment_change=equipment_change,
                            labels=[time_series_label],
                            ylim=ylim,
                            xlim=xlim,
                            save=False,
                        )
                    except Exception as e:
                        print(f"Error plotting {row['Station']}: {e}")
        else:
            print(f"Warning: Could not map title '{title}' to a time series label.")

    if annotate_threshold_worst == True:
        if datatype == "PWC":
            label_mapping = {
                "IGSrepro3 - ERA5": "ΔPWC[IGSrepro3 - ERA5]",
                "IGSdaily - ERA5": "ΔPWC[IGSdaily - ERA5]",
                "EPNrepro2 - ERA5": "ΔPWC[EPNrepro2 - ERA5]",
                "IGSdaily - IGSrepro3": "ΔPWC[IGSdaily - IGSrepro3]",
                "IGSdaily - EPNrepro2": "ΔPWC[IGSdaily - EPNrepro2]",
                "IGSrepro3 - EPNrepro2": "ΔPWC[IGSrepro3 - EPNrepro2]",
            }
        else:
            label_mapping = {
                "IGSdaily - IGSrepro3": "ΔZTD[IGSdaily - IGSrepro3]",
                "IGSdaily - EPNrepro2": "ΔZTD[IGSdaily - EPNrepro2]",
                "IGSrepro3 - EPNrepro2": "ΔZTD[IGSrepro3 - EPNrepro2]",
            }
        time_series_label = label_mapping.get(title)

        if time_series_label:
            for i, row in worst_stats_by_threshold_mean.iterrows():
                try:
                    plot_diff_time_series(
                        dataframes_dict,
                        row["Station"],
                        f"{datatype}_diff",
                        equipment_change=equipment_change,
                        labels=[time_series_label],
                        ylim=ylim,
                        xlim=xlim,
                        save=False,
                    )
                except Exception as e:
                    print(f"Error plotting {row['Station']}: {e}")

            for i, row in worst_stats_by_threshold_std.iterrows():
                # Skip if already plotted in the mean section
                if (
                    row["Station"]
                    not in worst_stats_by_threshold_mean["Station"].values
                ):
                    try:
                        plot_diff_time_series(
                            dataframes_dict,
                            row["Station"],
                            f"{datatype}_diff",
                            equipment_change=equipment_change,
                            labels=[time_series_label],
                            ylim=ylim,
                            xlim=xlim,
                            save=False,
                        )
                    except Exception as e:
                        print(f"Error plotting {row['Station']}: {e}")
        else:
            print(f"Warning: Could not map title '{title}' to a time series label.")

    if save == True:
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

    return g