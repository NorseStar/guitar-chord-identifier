"""Run the local guitar test page with: python app.py."""

import logging
from pathlib import Path

import gradio as gr
import pandas as pd
from src.streaming import stream_chunk, COLUMNS

from src.predict import load_model, predict_file


MODEL_PATH = Path(__file__).resolve().parent / "models" / "my_guitar_rf.joblib"


def classify_recording(filepath, bundle):
    """Return the chord, top three scores, and recording feedback for the page."""
    if not filepath:
        return "No recording yet", {}, "Record a chord or upload an audio file first."
    try:
        result = predict_file(filepath, bundle)
    except (OSError, ValueError, RuntimeError):
        logging.exception("Could not process the recording")
        return "Unable to read audio", {}, "Try recording again, or upload a WAV file."

    if result["prediction"] == "No chord":
        return "No chord — audio too quiet", {}, "Try moving closer to the microphone or playing a little louder."
    if result["prediction"] == "Unknown":
        return "Not enough usable audio", {}, "Play one chord and let it ring for about 3–4 seconds, then try again."

    # Show the three highest scores and how much audio was used.
    crop = result["crop"]
    seconds = crop["stop_s"] - crop["start_s"]
    scores = dict(list(result["scores"].items())[:3])
    return result["prediction"], scores, f"Used {seconds:.2f} seconds after trimming the quiet start."


def build_app():
    """Build the recording page and connect its buttons to prediction."""
    # Load the model once when the app starts.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Save my_guitar_rf.joblib from your guitar training notebook first.")
    bundle = load_model(MODEL_PATH)
    supported_chords = ", ".join(bundle["model"].classes_)

    with gr.Blocks(title="Guitar Chord Identifier", analytics_enabled=False) as demo:
        gr.Markdown("# Guitar Chord Identifier\nPlay a chord. See what the model hears.")
        gr.Markdown(
            f"Supported chords: **{supported_chords}**. "
            "Use the live microphone below, or upload a single recording further down."
        )
        with gr.Accordion("Live microphone -- start recording to follow chord changes", open=True):
            gr.Markdown("Keep playing while recording. Updates are requested every half second. "
                        "The last three seconds are analysed, so changes can take a few seconds to settle. "
                        "Silence clears the current prediction; other noise may still look like a chord.")
            live_state = gr.State(None)
            microphone = gr.Audio(sources=["microphone"], type="numpy", streaming=True,
                                  label="Live microphone")
            live_chord = gr.Textbox(label="Current chord", interactive=False)
            live_scores = gr.Label(label="Top 3 smoothed scores", num_top_classes=3)
            timeline = gr.LinePlot(x="seconds", y="score", color="chord",
                                   y_lim=[0, 1], title="Last 30 seconds", height=250)
            def reset_live():
                return None, "", {}, pd.DataFrame(columns=COLUMNS)
            live_outputs = [live_state, live_chord, live_scores, timeline]
            microphone.start_recording(reset_live, outputs=live_outputs, queue=True)
            microphone.stream(lambda chunk, state: stream_chunk(chunk, state, bundle),
                              inputs=[microphone, live_state], outputs=live_outputs,
                              stream_every=0.5, concurrency_limit=1, time_limit=3600,
                              api_visibility="private")
            microphone.stop_recording(reset_live, outputs=live_outputs, queue=True)
            microphone.clear(reset_live, outputs=live_outputs, queue=True)
        gr.Markdown("## Single recording or upload\nRecord one chord for 3–4 seconds, stop, then click Predict.")
        with gr.Row():
            with gr.Column():
                recording = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                     format="wav", label="Your guitar recording")
                with gr.Row():
                    predict = gr.Button("Predict chord", variant="primary")
                    clear = gr.ClearButton(value="Clear")
            with gr.Column():
                chord = gr.Textbox(label="Predicted chord", interactive=False)
                scores = gr.Label(label="Top 3 model scores", num_top_classes=3,
                                  show_heading=False)
                message = gr.Textbox(label="Recording feedback", interactive=False)
        gr.Markdown(
            "Scores are the model's estimates, **not guaranteed confidence**. "
            f"It recognises the {len(bundle['model'].classes_)} chords listed above and may be wrong on unfamiliar sounds. "
            "Live scores are smoothed model estimates, not calibrated confidence."
        )
        # Connect Predict to the saved model and the three output fields.
        predict.click(lambda audio: classify_recording(audio, bundle),
                      inputs=recording, outputs=[chord, scores, message],
                      api_name="predict_chord", concurrency_limit=1)
        # Clear the old prediction when a new recording is selected.
        recording.change(lambda: ("", {}, ""), outputs=[chord, scores, message],
                         queue=False, api_name=False)
        clear.add([recording, chord, scores, message])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, max_file_size="25mb")
