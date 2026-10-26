#!/usr/bin/env bash
# Turn a ptrain.py run into a .gguf that Parroting can import (Q8_0, about 250 MB).
# Usage: bash export_gguf.sh run1 [output name]
set -euo pipefail
RUN=${1:?usage: bash export_gguf.sh run1}
NAME=${2:-SenseVoiceSmall-mine-Q8_0.gguf}
HERE=$(cd "$(dirname "$0")" && pwd)
TC=${TRANSCRIBE_CPP:-$HERE/transcribe.cpp}
[ -f "$RUN/merged_model.pt" ] || { echo "No $RUN/merged_model.pt. Run: python ptrain.py train ..."; exit 1; }
if [ ! -x "$TC/build/bin/transcribe-quantize" ]; then
  echo "First run: downloading and building transcribe.cpp (a few minutes)..."
  # Same version as the engine inside Parroting (transcribe-cpp 0.2.4), so the app can read the file.
  [ -d "$TC" ] || git clone -q -c advice.detachedHead=false --depth 1 --branch v0.2.4 https://github.com/handy-computer/transcribe.cpp "$TC"
  cmake -S "$TC" -B "$TC/build" -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TOOLS=ON -DTRANSCRIBE_BUILD_TESTS=OFF > "$TC/build.log" 2>&1
  cmake --build "$TC/build" -j --target transcribe-quantize transcribe-cli >> "$TC/build.log" 2>&1 \
    || { echo "Build failed, see $TC/build.log"; exit 1; }
fi
SNAP=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('FunAudioLLM/SenseVoiceSmall'))")
CK=$(mktemp -d)
trap 'rm -rf "$CK"' EXIT
for f in "$SNAP"/*; do [ -f "$f" ] && cp -L "$f" "$CK/"; done
cp "$RUN/merged_model.pt" "$CK/model.pt"
echo "Converting..."
python3 "$TC/scripts/convert-sensevoice.py" "$CK" "$CK/f32.gguf" --repo-id FunAudioLLM/SenseVoiceSmall > "$RUN/convert.log" 2>&1 \
  || { echo "Conversion failed, see $RUN/convert.log"; exit 1; }
"$TC/build/bin/transcribe-quantize" "$CK/f32.gguf" "$RUN/$NAME" --quant Q8_0 > /dev/null
echo "Done: $RUN/$NAME ($(du -h "$RUN/$NAME" | cut -f1))"
# Transcribe the first test clip so you can see the file works
CLIP=$(python3 -c "import json,os;r=json.load(open('$RUN/result.json'));t=r['test_detail'][0];print(os.path.join(r['data'],t.get('audio','audio/'+t['id']+'.wav')));print(t['text'])" 2>/dev/null || true)
if [ -n "$CLIP" ]; then
  echo "Check  expected: $(echo "$CLIP" | sed -n 2p)"
  echo "            got: $("$TC/build/bin/transcribe-cli" -q -m "$RUN/$NAME" "$(echo "$CLIP" | head -1)" 2>/dev/null | sed -n 's/^text: //p')"
fi
echo "Copy it to your Mac, then in Parroting: Talk > Manage models > Import model..."
