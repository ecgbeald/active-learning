import pandas as pd


def slice_dataframe(df, start, end):
    return df.iloc[start:end].reset_index(drop=True)


def build_event_windows(event_sequences, window_size=10, stride=1):
    windows = []

    for machine_id, (events, timestamps, labels) in event_sequences.items():
        if len(events) >= window_size:
            for i in range(0, len(events) - window_size + 1, stride):
                window = events[i : i + window_size]
                window_timestamps = timestamps[i : i + window_size]

                # if timestamp diff > 30s, ignore
                if window_timestamps[-1] - window_timestamps[0] > 30:
                    continue

                windows.append(
                    {
                        "machine_id": machine_id,
                        "events": window,
                        "timestamp": window_timestamps,
                        "label": labels[i : i + window_size],
                    }
                )
        else:
            # if timestamp diff > 30s, ignore
            if timestamps[-1] - timestamps[0] <= 30:
                windows.append(
                    {
                        "machine_id": machine_id,
                        "events": events,
                        "timestamp": timestamps,
                        "label": labels,
                    }
                )
    return windows


def process_seq(df, window_size=5, slice_start=0, slice_end=None):
    df = slice_dataframe(df, slice_start, slice_end)
    event_sequences = (
        df.groupby("machine")
        .agg({"event": list, "timestamp": list, "label": list})
        .reset_index()
    )
    event_sequences["event_count"] = event_sequences["event"].apply(len)
    event_sequences = event_sequences[event_sequences["event_count"] >= window_size]
    event_sequences_dict = {}
    for _, row in event_sequences.iterrows():
        machine_id = row["machine"]
        events = row["event"]
        timestamps = row["timestamp"]
        labels = row["label"]
        if len(events) >= window_size:
            event_sequences_dict[machine_id] = (events, timestamps, labels)
    windows = build_event_windows(
        event_sequences_dict, window_size=window_size, stride=1
    )

    text = [
        str(window["timestamp"][0]) + " " + " ".join(map(str, window["events"]))
        for window in windows
    ]
    labels = [int(any(window["label"])) for window in windows]
    raw_dataset = {
        "combined": text,
        "label": labels,
    }
    return pd.DataFrame(raw_dataset)
