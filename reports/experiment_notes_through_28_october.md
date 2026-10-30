# Guitar Chord Identifier — DAT158

A small guitar-chord recognition project. The Gradio page now predicts isolated chords from
uploads or microphone recordings. Live feedback can come later. These are working notes for the report.

## Experiment notes

### 22 October — Getting audio into the pipeline

- Loaded recordings and plotted waveforms. There is silence before the playing,
  so used energy-based trimming instead of assuming a fixed start time.
- Dataset: 24 chords, 630 training files and 144 test files, plus noise recordings.
  Some Am/E files are extras with GarageBand in their names.
- Started with two-second crops, mono audio at 22,050 Hz and 12 mean chroma values.
  Chroma keeps pitch classes but loses octave and timing information.

### 24 October — First model

- Trained a Random Forest: 300 trees, balanced class weights, seed 42.
- Split the usable training recordings 80/20. Kept the supplied Test set aside.
- Two-second baseline: 92.9% validation accuracy, macro F1 0.928. One recording
  was too short. Results are kept in `reports/baseline_2s/`.
- Added static prediction and a quiet-audio gate. Model scores aren't certainty;
  loud noise can still get a chord prediction.

### 25 October — Listening and changing the crop

- Tried using recording numbers to identify playing techniques. Had to handle
  spaces in filenames and `Hm` filenames inside the Bm folder. Kept the actual
  files unchanged and recovered all 600 standard Fender recordings.
- Listening showed the protocol isn't reliable per file: A recording 3 had
  repeated strums, while recording 5 had a single downstrum. Stopped treating
  the inferred technique/fingering columns as verified labels.
- The two-second crop cut off a note in A recording 7. Increased the standard
  crop to **three seconds** to leave more room for the playing and decay.
- Playback sounded less bright after resampling. Comparing at the same sample
  rate showed that cropping itself wasn't causing that difference.
- Initially rejected 11 recordings because their trimmed audio was shorter than
  three seconds. Fixed that: keep **up to three seconds**, using shorter clips
  as they are. Averaging chroma still gives 12 features, so padding isn't needed.
- All 630 recordings now work: 504 training / 126 validation, **95.2% accuracy**,
  macro F1 **0.951**. Kept the original validation recordings; the one file
  excluded even by the old two-second baseline was added to training.
  This is still one small validation split, not proof of generalisation.

### 25 October — A page for trying my guitar

- Added `app.py` using Gradio: record through the microphone or upload audio,
  then click Predict to see the chord and top three scores.
- Uses the saved model and existing preprocessing. Checked uploads and silence
  through the running app. Later guitar tests are noted below.

### 25 October — Testing my own recordings

- Recorded 12 each of A, Am, C, D, E, Em and G: four downstrums, four repeated
  strumming patterns, then four slow-picking recordings within each chord folder.
- Kept the original M4A files and made separate WAV copies for analysis. Labels
  come from the folders; numbers only establish order inside each folder.
  Thumb/nail and G-voicing variations weren't logged per recording, so they're
  left unknown in the metadata.
- The existing model got **59/84 correct (70.2%)**, without retraining. A was
  predicted as Am in 10/12 cases; Em became E in 5/12. Downstrums scored 82.1%,
  repeated strumming 53.6%, and slow picking 75.0%. This confirms the supplied
  dataset's validation score doesn't describe performance on my setup.
- These recordings are now a diagnostic collection. If used to tune or train
  the model, a fresh recording session should be reserved for later evaluation.

### 28 October — Training on my guitar

- Worked through training in notebook 05 so the steps are easier to follow.
  Used the 84 phone recordings and the same 12 mean chroma features.
- Split into 63 training and 21 validation recordings, keeping every chord and
  playing style represented. A 300-tree Random Forest got **19/21 right (90.5%)**:
  one D became A, and one E became Em. This is a small, same-session check.
- Then trained a separate model on all 84 recordings and connected it to Gradio.
  Fresh recordings are needed to evaluate this model; the previous 21 now belong
  to its training data. Kept the original dataset model separately.
- First headset trial: **six of seven correct before manual trimming**. Em became
  correct after trimming. Encouraging, but only one attempt per chord.
- Keeping the planned scope to **14 chords: A–G and their minors**. The current
  model recognises seven; the remaining chords still need recordings.

### 28 October — Shuffling can start the crop too early

- Saved six headset recordings of E/Em. Five were correct; a fast E strum was
  predicted as Em. These are now debugging examples, not a final test set.
- In E_2, the crop started at 0.89 s, well before the louder playing around
  2.74 s. Much of the three-second window was spent on the lead-in.
- Notebook 06 compares the current rule, a stricter volume threshold, requiring
  0.15 s of sustained sound, and both changes together. E_2's score rose from
  about 40% to 86%, but all rules still got **5/6 correct**. Higher scores did
  not mean higher accuracy, and the fast-strum mistake remained.
- The stricter threshold also moved one slow-picked E crop forward by 0.45 s.
  The sustained-sound rule kept its original start. Need to listen for lost
  notes and check more examples before changing the app. A separate guitar/noise
  detector can wait; the app's preprocessing has not changed.

A longer window may
respond more slowly during live chord changes. Speech/singing negatives and
mixture augmentation stay on the backlog. None of these validation results
establish performance on a new guitar or microphone.

## Running it

The website uses `models/my_guitar_rf.joblib`, saved by notebook 05. With that
file and the environment ready, start it from the project root:

```powershell
.venv/Scripts/python.exe app.py
```

Open the local URL printed in the terminal (normally `http://127.0.0.1:7860`)
in Chrome or Edge. Allow microphone access, record one chord for 3–4 seconds,
stop, and click **Predict chord**. You can also upload a WAV recording. If no
microphone appears in an embedded browser, open the URL in your normal browser.
Leave the terminal running while testing; Ctrl+C stops the app. This is a local
page, not a public deployment, and this first version does not stream predictions.

Use Python 3.13 and the project virtual environment. From the project root:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
# Only if the data is not already downloaded:
.venv/Scripts/python.exe download_dataset.py
.venv/Scripts/python.exe -m src.train
.venv/Scripts/python.exe -m src.predict "path/to/chord.wav"
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The file pipeline keeps up to three seconds after trimming. Shorter clips are
accepted down to one second, a basic minimum-length guard that has not been tuned.
Training and prediction use the same preprocessing. The original-dataset training command above saves
`models/chord_rf.joblib`; it does not rebuild the website's personal model. That
training command reuses the split in
`reports/training_split.csv`. Silence, invalid audio and clips below the minimum
are reported; valid short recordings are not padded.
The original Test set is still reserved for final evaluation.

`03_preprocessing_experiments.ipynb` contains the listening experiment and notes.
`02_feature_extraction.ipynb` keeps the original two-second exploration for reference.
Reusable code is in `src/`, current model results in `reports/baseline/`.

To prepare and evaluate the personal collection (no model training):

```powershell
.venv/Scripts/python.exe -m src.personal_dataset
.venv/Scripts/python.exe -m src.evaluate_personal
```

Originals stay in `data/raw/data/marcus_recordings/<chord>/`; decoded WAVs go
to `data/processed/marcus_recordings/`. Both audio folders are ignored by Git.
Metadata and results are in `reports/marcus_recordings/`, with a walkthrough in
`notebooks/04_personal_recordings.ipynb`. Reproducing this particular experiment
requires the personal audio files, which are not included in Git.

To rebuild the website's model, prepare the personal collection as above, then
run `notebooks/05_training_on_my_guitar.ipynb` from top to bottom with its working
directory set to `notebooks/`. It extracts features, checks the validation split,
trains on all 84 recordings and saves `models/my_guitar_rf.joblib`. Restart the
app after saving. `notebooks/06_crop_start_experiment.ipynb` investigates the
saved headset checks separately; it does not retrain or change the website.

## Sources

- [Guitar chord dataset](https://huggingface.co/datasets/severyn-k/isolated-guitar-chords), CC BY 4.0.
- [librosa chroma](https://librosa.org/doc/latest/generated/librosa.feature.chroma_cqt.html),
  [scikit-learn score calibration](https://scikit-learn.org/stable/modules/calibration.html),
  [Gradio streaming](https://www.gradio.app/guides/streaming-inputs).
- OpenAI Codex helped with code, tests and drafting these notes. Listening
  observations came from my checks; model results came from running the code.

For submission: public GitHub repository with code/report and either a deployed
website or a screencast, following `task.md`.
