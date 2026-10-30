"""Small per-session rolling buffer for live microphone chunks."""
import librosa
import numpy as np
import pandas as pd
from src.predict import predict_window, rms_dbfs

COLUMNS = ["seconds", "chord", "score"]

def fresh_state():
    return {"audio": np.empty(0, dtype=np.float32), "samples": 0,
            "scores": None, "history": []}

def stream_chunk(chunk, state, bundle):
    if state is None:
        state = fresh_state()
    if chunk is None:
        return state, "Waiting for microphone", {}, pd.DataFrame(state["history"], columns=COLUMNS)
    sr, audio = chunk
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or sr <= 0:
        return fresh_state(), "Unknown: invalid audio", {}, pd.DataFrame(columns=COLUMNS)
    target_sr = bundle["preprocessing"]["sr"]
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    state["samples"] += len(audio)
    seconds = state["samples"] / target_sr
    # Gate the newest chunk so silence does not keep displaying an old chord.
    if rms_dbfs(audio) < bundle["min_rms_dbfs"]:
        state["audio"] = np.empty(0, dtype=np.float32)
        state["scores"] = None
        prediction = "No chord — audio too quiet"
        scores = {}
    else:
        window_size = round(bundle["preprocessing"]["duration"] * target_sr)
        state["audio"] = np.concatenate([state["audio"], audio])[-window_size:]
        minimum = round(bundle["preprocessing"]["min_duration"] * target_sr)
        if len(state["audio"]) < minimum:
            prediction, scores = "Listening…", {}
        else:
            result = predict_window(state["audio"], target_sr, bundle, require_full=False)
            scores = result["scores"]
            if scores:
                previous = state["scores"]
                if previous is not None:
                    scores = {chord: 0.6 * value + 0.4 * previous[chord]
                              for chord, value in scores.items()}
                state["scores"] = scores
                prediction = max(scores, key=scores.get)
            else:
                prediction = result["prediction"]
                state["scores"] = None
    for chord in bundle["model"].classes_:
        state["history"].append({"seconds": seconds, "chord": chord,
                                 "score": scores.get(chord, 0.0)})
    state["history"] = [row for row in state["history"] if row["seconds"] >= seconds - 30]
    top_three = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3])
    return state, prediction, top_three, pd.DataFrame(state["history"], columns=COLUMNS)
