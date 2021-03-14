import pandas as pd


def evaluate(pred_anomaly: pd.DataFrame, actual_anomaly: pd.DataFrame):
    """
    Evaluating model by comparing output anomaly dataframe with actual anomaly dataframe.
    """
    output = set(zip(list(pd.to_datetime(pred_anomaly.timestamp)), list(pred_anomaly.cell_name)))
    anomaly = set(zip(list(pd.to_datetime(actual_anomaly.timestamp)), list(actual_anomaly.cell_name)))

    total = len(actual_anomaly)
    correct = 0

    for o in output:
        if o in anomaly:
            correct += 1

    return correct / total


def true_results(path, threshold):
    if '.pkl' in path:
        df = pd.read_pickle(path)
        return df.loc[df.drop(columns=['timestamp', 'cell_name']).max(axis=1) >= threshold]
    else:
        df = pd.read_csv(path).set_index('Unnamed: 0')
        return df.loc[df.drop(columns=['timestamp', 'cell_name']).max(axis=1) >= threshold]


# %%

gt_df = pd.read_csv('hackathon_kpis_anonymised/anomaly_dataset.csv')
gt_df.set_index('Unnamed: 0', inplace=True)


# %%
pred_df = pd.concat((
    true_results('preds/iforest_notime.pkl', 0.55),
    true_results('preds/deepant_all.pkl', 0.8),
    true_results('preds/std_baseline_anomalies.pkl', 0.5),
))

evaluate(pred_df, gt_df)

