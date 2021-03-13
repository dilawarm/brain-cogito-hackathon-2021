def evaluate(expected_anomaly: pd.Dataframe, actual_anomaly: pd.Dataframe):
  """
  Evaluating model by comparing output anomaly dataframe with actual anomaly dataframe.
  """
  output = list(zip(list(expected_anomaly.timestamp), list(expected_anomaly.cell_name)))
  anomaly = list(zip(list(actual_anomaly.timestamp), list(actual_anomaly.cell_name)))

  total = len(actual_anomaly)
  correct = 0

  for o in output:
    if o in anomaly:
      correct += 1
  
  return correct / total