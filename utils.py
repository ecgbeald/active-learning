import pandas as pd


def process_csv(input_csv_path):
    """
    Process raw alert CSV files into standardised format.

    Args:
        input_csv_path (str): Path to the input CSV file in alerts_csv directory

    Returns:
        pd.DataFrame: The processed dataframe
    """

    # Read the input CSV
    print(f"Reading CSV from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    df = df.head(500_000)
    # Standardize column names
    # df.rename(columns={
    #     'time': 'timestamp',
    #     'name': 'event',
    #     'host': 'machine',
    # }, inplace=True)

    # unique_events = df['event'].unique()
    # event_map = {evt: idx + 1 for idx, evt in enumerate(unique_events)}
    # df['event'] = df['event'].map(event_map)
    # unique_machines = df['machine'].unique()
    # machines_map = {mac: idx + 1 for idx, mac in enumerate(unique_machines)}
    # df['machine'] = df['machine'].map(machines_map)

    # # Create binary label column (0=normal, 1=anomalous)
    # df['label'] = df['event_label'].apply(lambda x: 0 if x == '-' else 1)

    # # Drop unnecessary columns
    # df.drop(columns=['event_label', 'time_label', 'short', 'ip'], inplace=True)

    # # Create output directory if it doesn't exist

    # df.sort_values(by=['timestamp'], inplace=True)
    # min_timestamp = df['timestamp'].min()
    # df['timestamp'] = df['timestamp'] - min_timestamp
    df["combined"] = (
        df["timestamp"].astype(str)
        + " "
        + df["event"].astype(str)
        + " "
        + df["machine"].astype(str)
    )
    print(f"Final shape: {df.shape}")
    print(f"Unique events: {len(df['event'].unique())}")
    print(f"Unique machines: {len(df['machine'].unique())}")
    print(
        f"Normal/Anomalous distribution: {sum(df['label'] == 0)}/{sum(df['label'] == 1)}"
    )
    return df
