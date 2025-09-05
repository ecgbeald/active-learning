import pandas as pd


def build_event_windows(event_sequences, window_size=5, min_window_size=3):
    windows = []
    diff_timestamp = 300

    for machine_id, (events, timestamps, labels) in event_sequences.items():
        n = len(events)
        if n >= window_size:
            for i in range(n - window_size + 1):
                window = events[i : i + window_size]
                window_timestamps = timestamps[i : i + window_size]

                if window_timestamps[-1] - window_timestamps[0] <= diff_timestamp:
                    windows.append(
                        {
                            "machine_id": machine_id,
                            "events": window,
                            "timestamp": window_timestamps,
                            "label": labels[i : i + window_size],
                        }
                    )
                else:
                    j = i
                    while j < i + window_size:
                        k = j + min_window_size
                        while (
                            k <= i + window_size
                            and timestamps[k - 1] - timestamps[j] <= diff_timestamp
                        ):
                            windows.append(
                                {
                                    "machine_id": machine_id,
                                    "events": events[j:k],
                                    "timestamp": timestamps[j:k],
                                    "label": labels[j:k],
                                }
                            )
                            k += 1
                        j += 1

        else:
            if n > 0 and timestamps[-1] - timestamps[0] <= diff_timestamp:
                windows.append(
                    {
                        "machine_id": machine_id,
                        "events": events,
                        "timestamp": timestamps,
                        "label": labels,
                    }
                )
    return windows


def process_seq(df, window_size=5):
    event_sequences = (
        df.groupby("machine")
        .agg({"event": list, "timestamp": list, "label": list})
        .reset_index()
    )
    event_sequences_dict = {}
    for _, row in event_sequences.iterrows():
        machine_id = row["machine"]
        events = row["event"]
        timestamps = row["timestamp"]
        labels = row["label"]
        event_sequences_dict[machine_id] = (events, timestamps, labels)
    windows = build_event_windows(event_sequences_dict, window_size=window_size)

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
