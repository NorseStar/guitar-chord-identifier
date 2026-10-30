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

To check whether it works, I look at accuracy, precision, recall and F1, along
with a confusion matrix to see which chords get mixed up. For live feedback,
I also want to see how quickly it catches up when I change chords. A high score
on the website does not necessarily mean the prediction is right, especially
if I play a chord it was never trained on.

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
recording number within each folder, not across folders. I followed the same
pattern for all 11 chords: four single strums, four repeated-strumming recordings,
and four slow strums/picking recordings. Thumb/nail and voicing details were not logged 
per file. And while my sample size is small (I am just one person though, so I forgive myself), 
it was surprisingly accurate for how little recordings I had for each chord.

My original M4A files are preserved while analysis uses decoded WAV copies. Audio is
converted to mono at 22,050 Hz in order to standardise a sampling rate. Quiet parts 
are trimmed, then up to three seconds are kept; usable clips between one and three 
seconds are retained without padding. Each recording becomes 12 mean chroma values, 
representing pitch classes rather than octave or note order. A two-second crop previously
cut off a picked note, hence I upped the crop amount to 3 seconds.

I included the 132 original training recordings and the trained model in this
repository so the project can be run and retrained. The WAV copies are generated
locally. The separate headset checks and unused noise recordings are not included.
Microphone audio is processed by the local Gradio server.

## 3. Modeling

I started with a Random Forest, using 300 trees and a fixed random seed of 42
so I could repeat the experiment. The original dataset model used balanced
class weights, while the model trained on my own recordings uses the defaults.
I had also considered comparing KNN and SVM, but so far I have focused on getting
one model working with my guitar.

| Experiment | Evaluation recordings | Accuracy |
| --- | ---: | ---: |
| Original dataset, two-second crop | 126 validation | 92.9% |
| Original dataset, up to three seconds | 126 validation | 95.2% |
| Original model on my first seven chords | 84 personal recordings | 70.2% |
| Personal seven-chord model | 21 validation | 90.5% |
| Personal eleven-chord model | 33 validation | 81.8% |

The original model looked good on its validation recordings, but that did not
carry over as well to my own guitar. It got 59 of my first 84 recordings right,
with A often being mistaken for Am. This was why I moved on to training with
my own recordings instead of assuming that a good validation score was enough.

For the current 11-chord experiment, I used 99 recordings for training and kept
33 for validation. Each chord and playing style had three recordings in training
and one in validation. The model got **27 out of 33 right (81.8%)**, with D and Dm
causing most of the problems. Macro precision was **0.848**, recall **0.818** and
F1 **0.825**. These averages give each chord equal weight. The full results and
confusion matrix are in [reports/personal_11](reports/personal_11/).

The accuracy dropped after adding more chords, but the validation recordings
also changed, so I cannot say that the extra chords alone caused the drop.
There are only three validation recordings per chord, and they come from the
same collection as the training recordings. I still need fresh recordings to
get a better idea of how well it works in practice.

After checking validation, I trained a separate model on all 132 recordings for
the website. That means the 33 validation recordings are now part of this model's
training data too. Testing it on those again would not be an independent check.

### Trying my headset and investigating mistakes

With the earlier seven-chord model, I tried playing each chord through my
headset microphone. Six of seven were right on the first attempt. Em was
predicted as E, but changed to Em after I manually trimmed the recording.
This made me wonder how much the crop was affecting the prediction.

I saved six more E/Em recordings to look into it. Five were right, but a fast E
strum was mistaken for Em. I also noticed that shuffling noises could start the
automatic crop before I actually played. In one repeated E recording, the crop
started at 0.89 seconds while the main playing started around 2.74 seconds.
Almost two seconds of the three-second window were being spent on the lead-in.

Notebook 06 compares a stricter volume threshold with requiring sound to last
briefly before accepting it as the start. E's score went from about 40% to 86%,
but all the settings still got five of six recordings right. The fast-strum
mistake did not go away. The stricter threshold also skipped more of a slow-picked
recording, so I have kept the original file-trimming settings for now. Getting a
higher score on one recording is not enough reason to change the whole pipeline.

## 4. Deployment

I used Gradio because I wanted a simple way to try the model without spending
most of the project building a user interface. The website lets me upload a
recording, record one chord through the microphone, or leave the microphone
running for live feedback.

The live version keeps the latest three seconds of audio and requests a new
prediction every half second. It starts predicting once it has at least one
second of usable audio. Unlike the uploaded recordings, these windows are not
trimmed each time, since I may already be in the middle of playing a chord.

The scores are smoothed using 60% of the new scores and 40% of the previous
smoothed scores. This is intended to make the display less jumpy, although it
also adds some delay. A graph shows the last 30 seconds, and quiet audio clears
the current prediction and buffer. This still does not mean it can tell every
background noise apart from a guitar.

When I tried playing songs through my headset, the model could often follow a
chord while I stayed on it. I got more consistent predictions from uploaded
single strums. During chord changes I kept seeing Bm, even when I wasn't
playing it. The three-second window can contain both chords at once, which may
explain some of those mistakes. This was an informal playing test, so I don't
have an accuracy figure for live use.

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

The website runs locally. I recorded two videos where I play guitar and
briefly explain how the website works.

## 5. References

- [severyn-k/isolated-guitar-chords](https://huggingface.co/datasets/severyn-k/isolated-guitar-chords), CC BY 4.0.
- [librosa chroma features](https://librosa.org/doc/latest/generated/librosa.feature.chroma_cqt.html).
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
- [Gradio streaming inputs](https://www.gradio.app/guides/streaming-inputs).
- HVL DAT158 assignment 2 report template supplied by the course.
- OpenAI Codex helped write and explain some code (such as tests and audio pre-processing), run checks and help keep track
of, and draft, report notes.
  Recording and listening observations are fully mine and reported metrics come from the experiments.
