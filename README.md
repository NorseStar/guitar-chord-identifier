# Guitar Chord Identifier - DAT158
Marcus · Last updated: 30th of October 2026

[GitHub repository](https://github.com/NorseStar/guitar-chord-identifier)

## 1. Describe the problem

I wanted to make a basic guitar chord recognition software. The current scope 
is 11 major/minor chords: A, Am, B, Bm, C, D, Dm, E, Em, F and G, however
I started off wanting to download many different guitar chord data sets with
different types of guitars, playstyles and voicings while adding them all as labels,
but I quickly realised that the scope of the project would be too big after first
testing around with my first guitar chord audio data set. More of that is explained later.

I measure accuracy, per-chord precision, recall, F1 and a confusion matrix.
For live use, stability and the delay after a chord change also matter. Model
scores are not calibrated confidence, and the system cannot identify chords
outside its supported classes reliably.

## 2. Data

I started off with the isolated-guitar-chords dataset by severyn-k from hugging-face.
It included 630 training chord recordings and 144 test chord recordings, plus noise files.
The original Test partition remains untouched. The first issue I encountered was that an
algorithm I wrote wasn't properly picking up Bm chords. After a while, I found out that 'Bm'
is commonly written as 'Hm' in many parts of Europe, and therefore had to write new code in 
order to handle both spaces and Hm filenames in the Bm folder. My second problem arose after
manually listening to the recordings from the sample set. According to the huggingface page for
the audio samples, the original creator claimed he followed a protocol for how he recorded his audio
samples. After trusting his word, I realised that some of the audio files weren't lining up how I would 
expect them to, so I decided to manually look into and listen to the audio samples myself. I heard that 
the documented playing protocol did not always match individual files, so I stopped treating those 
inferred techniques as facts. With the audio samples being difficult to work with and not necessarily
being great at predicting chords from my own, personal guitar, I decided to record my own samples.

I then recorded my own guitar using my phone. There are now 132 recordings,
12 per chord. Folder names provide the chord labels. Files are sorted by their
recording number within each folder, not across folders. The 11 chords have four 
downstrums, four repeated-strumming takes and four slow-picking
takes. Thumb/nail and voicing details were not logged per file. And while my sample
size is small (I am just one person though, so I forgive myself), it was surprisingly 
accurate for how little recordings I had for each chord.

My original M4A files are preserved while analysis uses decoded WAV copies. Audio is
converted to mono at 22,050 Hz in order to standardise a sampling rate. Quiet parts 
are trimmed, then up to three seconds are kept; usable clips between one and three 
seconds are retained without padding. Each recording becomes 12 mean chroma values, 
representing pitch classes rather than octave or note order. A two-second crop previously
cut off a picked note, hence I upped the crop amount to 3 seconds.

Decoded WAV copies are generated locally. Microphone audio is processed by the local Gradio server.

## 3. Modeling

I use a Random Forest with 300 trees and seed 42. The original dataset baseline
used balanced class weights and the personal model uses the classifier defaults.
KNN and SVM remain possible comparisons, not completed experiments.

| Experiment | Evaluation recordings | Accuracy |
| --- | ---: | ---: |
| Original dataset, two-second crop | 126 validation | 92.9% |
| Original dataset, up to three seconds | 126 validation | 95.2% |
| Original model on my first seven chords | 84 personal recordings | 70.2% |
| Personal seven-chord model | 21 validation | 90.5% |
| Personal eleven-chord model | 33 validation | 81.8% |

The current split has 99 training and 33 validation recordings: one of each four
chord/technique takes is held aside. Macro precision is **0.848**, recall **0.818**
and F1 **0.825**. Detailed results and the confusion matrix are in
[reports/personal_11](reports/personal_11/). D and Dm account for most mistakes.
These are small within-collection splits, and the seven- and eleven-class
results do not use an identical validation set. They are not a controlled
measurement of the effect of adding classes.

After validation, a separate model is fitted on all 132 recordings for the app.
It needs fresh recordings for independent evaluation. The phone/headset change
also introduces a recording-condition difference.

An informal seven-chord headset check got six of seven right before manual
trimming. Six saved E/Em checks then got five right. In one repeated E recording,
shuffling started the crop at 0.89 s, before the main playing near 2.74 s.
Notebook 06 tried stricter and sustained-volume start rules: E's score rose from
40% to about 86%, but all rules still got five of six right. A fast E strum remained
mistaken for Em. A stricter rule also skipped more of a slow-picked recording.
The app's file-trimming settings therefore remain unchanged.

## 4. Deployment

The Gradio website supports uploads, single microphone recordings and a live
microphone section. Live audio uses a rolling three-second buffer with updates
requested every 0.5 seconds. It starts predicting after one second of usable
audio, without repeatedly trimming each window. Scores are smoothed using 60%
new and 40% previous scores, and a graph shows the last 30 seconds. Each browser
session has its own buffer. Quiet chunks clear the buffer and prediction.

This is a first live version: a window can contain two chords during a change,
and smoothing adds delay. Silence rejection does not distinguish all noise from
guitar. Automated buffer and audio tests pass, but live microphone accuracy and
latency still need hands-on testing. Streaming sessions currently have a one-hour
limit; stop and restart recording for another session.

### Run the website

Use Python 3.13. From the project root on Windows:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe app.py
```

Open the printed local address, normally http://127.0.0.1:7860, in Chrome or Edge.
Allow microphone access. Start recording in the live section and keep playing;
stop recording to end feedback. Use the separate single-recording section for
uploads. Turn headset noise cancellation off and use flat EQ for initial tests.
Restart the app after replacing its model. The app loads
`models/my_guitar_rf.joblib`; the original dataset model is kept separately.

### Reproduce training

The included personal M4A recordings are in
`data/raw/data/marcus_recordings/<chord>/Nytt opptak <number>.m4a`, 12 per chord
in the technique order described above. Then run:

```powershell
.venv/Scripts/python.exe -m src.personal_dataset
.venv/Scripts/python.exe -m src.train_personal
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The first command creates WAV copies and current metadata. The second saves the
split, metrics, predictions and confusion matrix to `reports/personal_11/`, then
rebuilds the app model. Notebook 07 walks through these results. Notebook 05
preserves the earlier seven-chord learning exercise; rerunning its save cell
would replace the app model with that older scope.

For the public dataset baseline, run `download_dataset.py`, then
`python -m src.train`. That command trains the separate `models/chord_rf.joblib`.
Earlier experiments are in notebooks 01–06 and the
[dated notes](reports/experiment_notes_through_28_october.md).

The website currently runs locally. A screencast or public deployment link is
still needed for submission. Future checks should log fresh chord labels and
predictions, especially D/Dm, and measure how long live predictions take to
settle. More datasets, speech mixtures and a separate noise detector can wait.

## 5. References

- [severyn-k/isolated-guitar-chords](https://huggingface.co/datasets/severyn-k/isolated-guitar-chords), CC BY 4.0.
- [librosa chroma features](https://librosa.org/doc/latest/generated/librosa.feature.chroma_cqt.html).
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
- [Gradio streaming inputs](https://www.gradio.app/guides/streaming-inputs).
- HVL DAT158 assignment 2 report template supplied by the course.
- OpenAI Codex helped write and explain some code (such as tests and audio pre-processing), run checks and help keep track
of, and draft, report notes.
  Recording and listening observations are fully mine and reported metrics come from the experiments.
