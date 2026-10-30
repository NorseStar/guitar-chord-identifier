# Personal recording collection

84 M4A recordings provided by me, stored unchanged under
`data/raw/data/marcus_recordings/`. WAV copies under `data/processed/` are decoded
at the native 48 kHz mono format; ordinary model preprocessing then converts
them to 22,050 Hz and selects up to three seconds. Decoding does not restore
information already lost in M4A compression.

Chord labels use the folder names, not a presumed global order.
Within each folder, numeric filename order gives positions 1–12:
1–4 downstrums, 5–8 repeated up/down strumming, 9–12 slow picking.
The patterns were not performed with exact timing.

Thumb and fingernail playing were used for variety but not identified per file.
Half the G recordings used the third fret on the B string; the label stays G.
Individual picking material and voicing are unknown. No instrument model,
microphone identity or session identifier has been inferred.

`metadata.csv` links original and decoded paths, records original SHA-256 hashes,
and contains labels and header measurements. Paths are relative to the project root.
`predictions.csv` includes all class scores and selected crop boundaries.
`metrics.json` identifies the evaluated model by its hash. All 24 model outputs
are allowed, even though this collection contains only seven chord classes.

The existing model scored 59/84 (70.2%); the correct chord was in the top three
for 79/84 (94.0%). There were no rejected files. Top-three accuracy does not mean
the first prediction was correct, and the scores have not been calibrated.

These files were not used for training. They can serve as a development collection
for investigating the transfer gap; do not claim them as an untouched final test
if future choices are based on these results. Personal audio is ignored by Git.
