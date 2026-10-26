# Parroting speech model training kit

Train a speech model that knows **your** voice, accent and words, then use it in [Parroting](https://sona-orbit.github.io/parroting/).

Parroting runs speech recognition on your Mac with SenseVoice Small. This kit fine-tunes that model on your own recordings with LoRA (about 2% of the weights), tests it on clips it never saw, and turns it into a file Parroting can import.

Training does **not** happen inside Parroting. You run this kit on your own computer with an NVIDIA GPU.

## What you get

Our own test, 2 hours of everyday dictation, test set of 263 new clips:

| | Error rate on the test set |
|---|---|
| Original model | 6.7% |
| Trained with this kit | 4.7–5.5% (two runs) |

- That is 19–30% fewer errors. Runs differ because the best epoch is picked on a validation set whose text is itself the model's guess. The biggest gains are on words you say often: names, product names, mixed-in English terms.
- These recordings only had the model's own transcripts as text. With read-aloud clips (exact text) you can expect more.
- Trade-off: the model gets closer to you and slightly worse on other people's speech, about 1 point higher error rate on a general test set.

## You need

- A computer with an **NVIDIA GPU, 8 GB or more**. Linux, or Windows with WSL2.
  - CPU-only also works, but it is slow: about 50 minutes per epoch for 90 minutes of audio on 16 cores.
  - Training on a Mac has not been tested.
- Python 3.10 or newer, git, cmake and a C++ compiler (for the one-time conversion tool build).
- About 5 GB of free disk space.
- 20–30 minutes or more of your recordings with correct text (see step 1).

Time on an RTX 4090 with 90 minutes of audio: about 8 minutes to train, plus a few minutes to build the conversion tool the first time.

## Steps

### 1. Collect recordings in Parroting

The model learns from **audio + correct text**, so the quality of the text matters most.

- **Read aloud** (best, on the Personalization page): Parroting shows sentences, you read them, and it marks the words it misheard. The text is exact. You can do it a few minutes at a time.
- **Confirmed fixes**: when you fix a misheard word after dictating, Parroting asks you to confirm it. Confirmed fixes become training text.
- **Recognized only**: ordinary dictation where you changed nothing. It works, but the text may contain the model's own mistakes, so gains are limited.

Parroting keeps only your latest 300 recordings by default. To build up more, raise **Settings > Advanced > More settings > History > Keep recordings**. Read-aloud recordings are always kept.

### 2. Export the training data

In Parroting, open **Words & history > My data**, then choose **Export training data…**. Unzip the file on the training computer. The folder looks like this:

```
parroting-asr-dataset-20261027-1200/
  dataset.json
  manifest.jsonl
  audio/
```

About 1 in 5 clips is marked as **test**. Those clips are never trained on. They only measure before and after.

### 3. Set up (once)

```bash
git clone https://github.com/sona-orbit/parroting-asr-training
cd parroting-asr-training
python3 -m venv .venv          # on Ubuntu you may need: sudo apt install python3-venv
source .venv/bin/activate
pip install torch              # pick the build for your CUDA version at https://pytorch.org
pip install -r requirements.txt
```

### 4. Check the data

```bash
python ptrain.py check ~/parroting-asr-dataset-20261027-1200
```

This shows how many minutes you have from each source, warns you if there is too little, and shows whether your GPU was found.

### 5. Train

```bash
python ptrain.py train ~/parroting-asr-dataset-20261027-1200 --out run1
```

The first run downloads the original model (about 900 MB). The kit then does four things:

1. Tests the original model on the test clips.
2. Trains for 4 epochs.
3. Keeps the best epoch, chosen on a separate validation set.
4. Tests again.

The result is in `run1/report.md`, for example:

```
| | Test error rate | Doubled English words |
| Original model | 6.73% | 0 |
| Trained | 4.68% | 1 |
- 30% fewer errors; of 263 test clips, 85 got better and 38 got worse
- **Use it**
```

If the report says **Don't use it**, the model didn't get clearly better. Collect more read-aloud clips and try again.

### 6. Make the file for Parroting

```bash
bash export_gguf.sh run1
```

The first time, this downloads and builds [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) v0.2.4, the same engine Parroting uses. It then writes `run1/SenseVoiceSmall-mine-Q8_0.gguf` (about 250 MB) and transcribes one test clip so you can see the file works.

Chinese output from this step is in Simplified characters. That is expected: Parroting converts it to your script.

### 7. Import it into Parroting

Copy the `.gguf` file to your Mac. In Parroting, open **Talk > Manage models > Import model…** and choose it. Parroting first test-transcribes a short clip and only switches when that works. To go back at any time, choose **Back to the standard model**.

## Options

| Option | Default | When to change it |
|---|---|---|
| `--epochs` | 4 | More data can take 5–6 |
| `--lr` | 3e-4 | Lower it (1e-4) if the validation error goes up from the first epoch |
| `--rank` | 8 | Rarely needed |
| `--sources` | all | `--sources read_aloud,confirmed` trains on exact text only |
| `--max-seconds` | auto, from free GPU memory | Lower it if you get "CUDA out of memory" |
| `--device` | auto | `cpu` to force CPU |

## Good to know

- **Training text is converted to Simplified Chinese.** In our tests, training on Traditional characters made the model worse, because the original model was trained mostly on Simplified.
- **Clips over 30 seconds are skipped.** The model reads up to 30 seconds at a time.
- **"Doubled English words"** counts outputs like `GoogleGoogle`. An earlier training of ours had this side effect, so the kit watches for it.
- **More hours of the same kind of data helps less than better text.** Going from 30 minutes to 2 hours of "recognized only" data only moved the error rate from 3.6% to 3.4% in our test. Read-aloud clips are the better investment.

## Dataset format (parroting-asr-dataset v1)

`dataset.json` describes the whole export. `manifest.jsonl` has one clip per line:

```json
{"id": "...", "audio": "audio/....wav", "seconds": 4.8,
 "text": "the correct text, verbatim",
 "asr_text": "what the model heard at the time",
 "source": "read_aloud | confirmed | recognized",
 "split": "train | test", "recorded_at": "...", "model": "...", "lang": "zh"}
```

Audio is 16 kHz mono 16-bit WAV. A clip's `split` never moves from test back to train in later exports, so you can compare runs.

## Privacy

The export contains your voice and what you said. Everything in this kit runs on your own computer. The only downloads are the original model from Hugging Face and transcribe.cpp from GitHub. Think before sharing your dataset or your trained model with anyone.

## License

- The code in this repository is under the MIT License.
- SenseVoice Small is released under the [FunASR model license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE). Models you train from it are bound by that license too.
