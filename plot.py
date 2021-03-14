import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


def plot_anomalies(
    df: pd.DataFrame,
    anomaly_idx: pd.DataFrame,
    plottable_cols,
    title="",
    threshold=0.7,
    show=True,
):
    """
    df: The data frame
    anomaly_idx: a float (range [0,1]) data frame indicating where anomalies are located
    """
    fig = make_subplots(
        rows=len(plottable_cols),
        cols=1,
        subplot_titles=plottable_cols,
    )
    # FIXME: Clean up this loop
    for i, col in enumerate(plottable_cols):
        anomaly_idx.loc[:, col] = anomaly_idx[col].astype(float)
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[col],
                mode="lines",
                showlegend=False,
                line={"color": "#0099C6"},
            ),
            row=i + 1,
            col=1,
        )

        anomalies = df.loc[anomaly_idx[col] >= threshold][["timestamp", col]]
        scores = anomaly_idx[col].loc[anomaly_idx[col] >= threshold]
        if scores.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies[col],
                mode="markers",
                showlegend=False,
                marker={"color": ["rgba(255,0,0," + str(x) + ")" for x in scores]},
            ),
            row=i + 1,
            col=1,
        )

    yaxis = {
        "yaxis" + str(i) if i > 0 else "yaxis": {"range": (0, 1)}
        for i in range(len(plottable_cols) + 1)
    }
    fig.update_layout(
        title=title,
        height=200 * len(plottable_cols),
        xaxis={"range": (df["timestamp"].min(), df["timestamp"].max())},
        **yaxis
    )
    if show:
        fig.show()
    return fig


def plot_anomalies_by_cell(
    df: pd.DataFrame, anomaly_idx: pd.DataFrame, plottable_cols, cell_names
):
    """
    WARNING: Slow if you have many cells
    df: The full data frame
    anomaly_idx: a boolean data frame indicating where anomalies are located
    """
    groups = df.groupby("cell_name").groups
    for cell, cell_idx in tqdm(groups.items()):
        if cell in cell_names:
            cell_df = df.loc[cell_idx]

            cell_anomalies = anomaly_idx.loc[cell_idx]

            plot_anomalies(cell_df, cell_anomalies, plottable_cols, cell)
