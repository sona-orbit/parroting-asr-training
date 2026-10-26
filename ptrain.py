#!/usr/bin/env python3
"""Train your own Parroting speech model (SenseVoice Small + LoRA).

  python ptrain.py check  <dataset folder>              # is there enough data, is the format right
  python ptrain.py train  <dataset folder> --out run1   # train, then test; results in run1/report.md
  bash export_gguf.sh run1                              # make a .gguf Parroting can import

The dataset folder is what Parroting's "Export training data" gives you, unzipped
(dataset.json, manifest.jsonl, audio/).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, collections, json, math, os, random, re, sys, time

DUP = re.compile(r"([A-Za-z]{2,})\1")


def load(folder, sources=None):
    meta = json.load(open(os.path.join(folder, "dataset.json"), encoding="utf-8"))
    if meta.get("format") != "parroting-asr-dataset" or meta.get("version") != 1:
        sys.exit("Not a Parroting v1 training dataset (dataset.json format/version)")
    rows, bad = [], 0
    for line in open(os.path.join(folder, "manifest.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r["path"] = os.path.join(folder, r["audio"])
        if not os.path.exists(r["path"]) or not r.get("text", "").strip():
            bad += 1
            continue
        if sources and r["source"] not in sources:
            continue
        rows.append(r)
    return meta, rows, bad


def mins(rows):
    return sum(r.get("seconds", 0) for r in rows) / 60


def cmd_check(a):
    meta, rows, bad = load(a.data)
    print(f"Dataset: {a.data}\nParroting {meta.get('app_version')}, model at export: {meta.get('model') or 'unknown'}, language {meta.get('language')}")
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["source"], r["split"])].append(r)
    names = {"read_aloud": "read aloud", "confirmed": "confirmed fixes", "recognized": "recognized only"}
    for s in ("read_aloud", "confirmed", "recognized"):
        tr, te = by[(s, "train")], by[(s, "test")]
        if tr or te:
            print(f"  {names[s]:<16} train {len(tr):>5} clips {mins(tr):6.1f} min   test {len(te):>4} clips {mins(te):5.1f} min")
    if bad:
        print(f"  {bad} clips have no audio or no text and will be skipped")
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    good = [r for r in tr if r["source"] in ("read_aloud", "confirmed")]
    print()
    if len(te) < 30:
        print("! Fewer than 30 test clips: the before/after comparison will be unreliable. Record more read-aloud rounds, then export again.")
    if mins(tr) < 15:
        print("! Less than 15 minutes of training audio: you will probably see no gain. Aim for 20-30 minutes or more.")
    elif not good:
        print("i All training text is 'recognized only': it works, but gains are limited. Read-aloud clips or confirmed fixes help more.")
    else:
        print("OK: ready to train.")
    try:
        import torch
        dev = "CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU (slow)"
        print("Will use:", dev)
    except ImportError:
        print("! PyTorch is not installed. See Setup in the README.")


def cmd_train(a):
    import numpy as np, opencc, torch, torch.nn as nn
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

    torch.manual_seed(0); random.seed(0)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.max_seconds is None:
        # About 5 padded seconds of audio per free GB (measured: 8 GB free fits 40 s)
        free = torch.cuda.mem_get_info()[0] / 2**30 if dev == "cuda" else 16
        a.max_seconds = max(30.0, min(240.0, 5 * (free - 1)))
    os.makedirs(a.out, exist_ok=True)
    srcs = set(a.sources.split(",")) if a.sources else None
    meta, rows, _ = load(a.data, srcs)
    test = [r for r in rows if r["split"] == "test"]
    train = [r for r in rows if r["split"] == "train"]
    random.Random(42).shuffle(train)
    nd = max(20, int(len(train) * 0.08))
    dv, tr = train[:nd], train[nd:]
    if len(tr) < 50 or len(test) < 10:
        sys.exit("Too little data (need 50+ training clips and 10+ test clips). Run check first.")
    TW, CN = opencc.OpenCC("s2twp"), opencc.OpenCC("t2s")
    norm = lambda t: re.sub(r"[\s\W_]+", "", TW.convert(t or "").lower())

    def cer(r, h):
        r, h = norm(r), norm(h)
        if not r:
            return 0.0 if not h else 1.0
        d = list(range(len(h) + 1))
        for i, x in enumerate(r, 1):
            p, d[0] = d[0], i
            for j, y in enumerate(h, 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
        return d[len(h)] / len(r)

    log = open(os.path.join(a.out, "train.log"), "a", encoding="utf-8")

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True); log.write(s + "\n"); log.flush()

    say(f"train {len(tr)} clips ({mins(tr):.1f} min), validation {len(dv)}, test {len(test)}; device {dev}")
    am = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device=dev, disable_update=True)
    M, tok, fe = am.model, am.kwargs["tokenizer"], am.kwargs["frontend"]
    enc = lambda s: tok.encode(s, allowed_special="all") if "allowed_special" in tok.encode.__code__.co_varnames else tok.encode(s)

    def ids_of(r):
        # Train on Simplified Chinese text: Traditional targets made the model worse in our tests.
        # Parroting converts the output back to your script.
        lang = "<|en|>" if r.get("lang") == "en" else "<|zh|>"
        return enc(lang) + enc("<|NEUTRAL|>") + enc("<|Speech|>") + enc("<|withitn|>") + enc(CN.convert(r["text"]))

    def decode(rs):
        M.eval(); out = []
        with torch.no_grad():
            for r in rs:
                out.append(rich_transcription_postprocess(am.generate(input=r["path"], language="auto", use_itn=True, disable_pbar=True)[0]["text"]))
        return out

    def score(rs):
        h = decode(rs)
        return float(np.mean([cer(r["text"], x) for r, x in zip(rs, h)])), sum(bool(DUP.search(x)) for x in h), h

    class LoRA(nn.Module):
        def __init__(s, base, r, alpha):
            super().__init__(); s.base = base; s.scale = alpha / r
            s.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
            s.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device)); s.drop = nn.Dropout(0.05)

        def forward(s, x):
            return s.base(x) + (s.drop(x) @ s.A.T @ s.B.T) * s.scale

    t0 = time.time()
    b_test = score(test)
    say(f"before: test error rate {100*b_test[0]:.2f}%, doubled English words {b_test[1]} ({time.time()-t0:.0f} s)")
    names = ("linear_q_k_v", "linear_out", "w_1", "w_2")
    reps = [(p, cn, ch) for pn, p in M.named_modules() if pn.startswith("encoder") for cn, ch in p.named_children() if isinstance(ch, nn.Linear) and cn in names]
    for p, cn, ch in reps:
        setattr(p, cn, LoRA(ch, a.rank, 2 * a.rank))
    for p in M.parameters():
        p.requires_grad = False
    lp = [p for n, p in M.named_parameters() if n.endswith((".A", ".B"))]
    for p in lp:
        p.requires_grad = True
    say(f"LoRA: {len(reps)} layers, {sum(p.numel() for p in lp)/1e6:.2f}M parameters")
    feats = {}
    for r in tr:
        au = load_audio_text_image_video(r["path"], fs=16000)
        sp, sl = extract_fbank(au, data_type="sound", frontend=fe, is_final=True)
        feats[r["id"]] = (sp[0], int(sl[0]))
    opt = torch.optim.AdamW(lp, lr=a.lr, weight_decay=0.01)
    best, best_ep, hist = 9.0, 0, []
    ad = os.path.join(a.out, "adapter.pt")
    for ep in range(a.epochs):
        M.train(); random.shuffle(tr); tot = 0.0
        for i in range(0, len(tr), a.batch):
            b = tr[i:i + a.batch]
            # Split the batch by audio length and accumulate gradients, so small GPUs work too.
            # Padded size = longest clip x clip count, so sort by length before splitting.
            parts, cur = [], []
            for r in sorted(b, key=lambda r: r.get("seconds", 10)):
                longest = max([x.get("seconds", 10) for x in cur + [r]])
                if cur and longest * (len(cur) + 1) > a.max_seconds:
                    parts.append(cur); cur = []
                cur.append(r)
            parts.append(cur)
            opt.zero_grad()
            for part in parts:
                X = nn.utils.rnn.pad_sequence([feats[r["id"]][0] for r in part], batch_first=True).to(dev)
                L = torch.tensor([feats[r["id"]][1] for r in part], device=dev)
                Y = [torch.tensor(ids_of(r)) for r in part]
                YL = torch.tensor([len(y) for y in Y], device=dev)
                Yp = nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=-1).to(dev)
                loss, _, _ = M(X, L, Yp, YL)
                (loss * len(part) / len(b)).backward(); tot += float(loss) * len(part) / len(b)
            opt.step()
        dc, dd, _ = score(dv)
        obj = dc + 0.005 * dd
        hist.append({"epoch": ep + 1, "loss": round(tot, 1), "val_error_%": round(100 * dc, 2), "doubled_en": dd, "sec": round(time.time() - t0)})
        say(hist[-1])
        if obj < best:
            best, best_ep = obj, ep + 1
            torch.save({k: v for k, v in M.state_dict().items() if k.endswith((".A", ".B"))}, ad)
    M.load_state_dict(torch.load(ad), strict=False)
    a_test = score(test)
    for pn, p in list(M.named_modules()):
        for cn, ch in list(p.named_children()):
            if isinstance(ch, LoRA):
                with torch.no_grad():
                    ch.base.weight += (ch.B @ ch.A) * ch.scale
                setattr(p, cn, ch.base)
    torch.save({k: v.cpu() for k, v in M.state_dict().items()}, os.path.join(a.out, "merged_model.pt"))
    better = sum(cer(r["text"], h2) < cer(r["text"], h1) for r, h1, h2 in zip(test, b_test[2], a_test[2]))
    worse = sum(cer(r["text"], h2) > cer(r["text"], h1) for r, h1, h2 in zip(test, b_test[2], a_test[2]))
    gain = (b_test[0] - a_test[0]) / b_test[0] * 100 if b_test[0] else 0
    verdict = "Use it" if a_test[0] < b_test[0] * 0.97 and a_test[1] <= b_test[1] + 2 else "Don't use it (no clear gain)"
    res = {"data": os.path.abspath(a.data), "params": {"rank": a.rank, "lr": a.lr, "epochs": a.epochs, "batch": a.batch, "best_epoch": best_ep},
           "n": {"train": len(tr), "pick": len(dv), "test": len(test)}, "history": hist,
           "before": {"cer": b_test[0], "dup": b_test[1]}, "after": {"cer": a_test[0], "dup": a_test[1]},
           "test_detail": [{"id": r["id"], "audio": r["audio"], "text": r["text"], "before": h1, "after": h2} for r, h1, h2 in zip(test, b_test[2], a_test[2])],
           "minutes": round((time.time() - t0) / 60, 1), "verdict": verdict}
    json.dump(res, open(os.path.join(a.out, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rep = f"""# Training result

- Data: {len(tr)} training clips ({mins(tr):.1f} min), {len(test)} test clips (never trained on)
- Time: {res['minutes']} min; best epoch {best_ep}

| | Test error rate | Doubled English words |
|---|---|---|
| Original model | {100*b_test[0]:.2f}% | {b_test[1]} |
| Trained | {100*a_test[0]:.2f}% | {a_test[1]} |

- {abs(gain):.0f}% {'fewer' if gain >= 0 else 'more'} errors; of {len(test)} test clips, {better} got better and {worse} got worse
- **{verdict}**

Next: `bash export_gguf.sh {a.out}` makes the file to import into Parroting.
Clip-by-clip comparison: test_detail in result.json.
"""
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write(rep)
    say(rep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check a dataset"); c.add_argument("data")
    t = sub.add_parser("train", help="train and test")
    t.add_argument("data"); t.add_argument("--out", default="run1")
    t.add_argument("--epochs", type=int, default=4); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--rank", type=int, default=8); t.add_argument("--batch", type=int, default=8)
    t.add_argument("--max-seconds", dest="max_seconds", type=float, default=float(os.environ["MAX_SECONDS"]) if "MAX_SECONDS" in os.environ else None,
                   help="padded seconds of audio per GPU step; lower it if you run out of memory (8 GB: ~40, 24 GB: ~240)")
    t.add_argument("--sources", help="use only these sources, e.g. read_aloud,confirmed")
    t.add_argument("--device", help="cuda or cpu (default: auto)")
    a = ap.parse_args()
    {"check": cmd_check, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
