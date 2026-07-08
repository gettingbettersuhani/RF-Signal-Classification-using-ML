"""
======================================================================
    RF SIGNAL CLASSIFICATION SYSTEM
    DRDO Project Showcase
======================================================================
    Author  : Suhani Pareek
    Input   : data/samples/synthetic_dataset_V1.pkl
    Models  : Random Forest | SVM | Residual 1D-CNN
    Classes : 9 Modulation Types
    Run     : python demo/showcase.py
======================================================================
"""

import os
import sys
import time
import json
import random
import pickle
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

# ── project root ──
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── third-party (all in requirements.txt) ──
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import torch
from sklearn.preprocessing import LabelEncoder

# ── project modules ──
from src.features.extract_features import extract_features, estimate_snr
from src.data.preprocessing import calculate_fft_shifted
from src.deep_learning.cnn_model import CNNModel

# ── paths ──
SAMPLE_PKL    = PROJECT_ROOT / "data" / "samples" / "synthetic_dataset_V1.pkl"
MODEL_DIR     = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR    = PROJECT_ROOT / "results" / "showcase"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.75

# ====================================================================
#  Console Helpers
# ====================================================================
W = 70

def line(ch="="):
    print(ch * W)

def header(title):
    print()
    line()
    print(f"  {title}")
    line()
    print()

def row(label, value):
    print(f"    {label:<24} {value}")

# ====================================================================
#  1. Load Dataset
# ====================================================================

def load_dataset():
    header("LOADING DATASET")
    if not SAMPLE_PKL.exists():
        print(f"  ERROR: {SAMPLE_PKL} not found"); sys.exit(1)
    row("File", SAMPLE_PKL.name)
    t0 = time.perf_counter()
    with open(SAMPLE_PKL, "rb") as f:
        try:    data = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0); data = pickle.load(f, encoding="latin1")
    row("Loaded in", f"{time.perf_counter()-t0:.1f}s")

    keys = list(data.keys())
    if isinstance(keys[0], tuple):
        mods = sorted({k[0] for k in keys})
        snrs = sorted({k[1] for k in keys})
        total = sum(len(v) for v in data.values())
    else:
        mods = sorted(str(k) for k in keys); snrs = []
        total = sum((v.shape[0] if hasattr(v,"shape") else len(v)) for v in data.values())

    row("Total signals", f"{total:,}")
    row("Modulation classes", len(mods))
    if snrs:
        row("SNR range", f"{min(snrs)} dB to {max(snrs)} dB")
    print()
    print("    Classes:", "  ".join(mods))
    print()
    return data, mods, snrs

# ====================================================================
# 2. Load All Three Models
# ====================================================================
def load_models():

    header("LOADING TRAINED MODELS")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ----------------------------------------------------------
    # Label Encoder
    # ----------------------------------------------------------

    label_encoder = _load_label_encoder()

    row(
        "Classes",
        ", ".join(label_encoder.classes_)
    )

    # ----------------------------------------------------------
    # Standard Scaler
    # ----------------------------------------------------------

    scaler_path = PROCESSED_DIR / "standard_scaler.pkl"

    if not scaler_path.exists():

        raise FileNotFoundError(
            f"{scaler_path} not found."
        )

    scaler = joblib.load(
        scaler_path
    )

    row(
        "Standard Scaler",
        "Loaded"
    )

    # ----------------------------------------------------------
    # Selected Features
    # ----------------------------------------------------------

    feature_path = PROCESSED_DIR / "selected_features.csv"

    if not feature_path.exists():

        raise FileNotFoundError(
            f"{feature_path} not found."
        )

    selected_features = pd.read_csv(
        feature_path
    )["Feature"].tolist()

    row(
        "Selected Features",
        len(selected_features)
    )

    # ----------------------------------------------------------
    # Random Forest
    # ----------------------------------------------------------

    rf_path = MODEL_DIR / "random_forest.pkl"

    if not rf_path.exists():

        raise FileNotFoundError(rf_path)

    rf = joblib.load(
        rf_path
    )

    row(
        "Random Forest",
        "Loaded"
    )

    # ----------------------------------------------------------
    # SVM
    # ----------------------------------------------------------

    svm_path = MODEL_DIR / "svm.pkl"

    if not svm_path.exists():

        raise FileNotFoundError(svm_path)

    svm = joblib.load(
        svm_path
    )

    row(
        "SVM",
        "Loaded"
    )

    # ----------------------------------------------------------
    # CNN
    # ----------------------------------------------------------

    cnn_path = MODEL_DIR / "cnn_best.pth"

    if not cnn_path.exists():

        raise FileNotFoundError(cnn_path)

    checkpoint = torch.load(

        cnn_path,

        map_location=device,

        weights_only=False

    )

    state_dict = (

        checkpoint["model_state_dict"]

        if isinstance(checkpoint, dict)

        and "model_state_dict" in checkpoint

        else checkpoint

    )

    cnn = CNNModel(

        num_classes=len(

            label_encoder.classes_

        )

    )

    cnn.load_state_dict(

        state_dict

    )

    cnn.to(device)

    cnn.eval()

    row(
        "Residual CNN",
        f"Loaded ({device})"
    )

    print()

    return (

        rf,

        svm,

        cnn,

        selected_features,

        device,

        label_encoder,

        scaler

    )

# ====================================================================
#  Helpers
# ====================================================================
def _iq2row(iq):
    iq = np.array(iq)
    if iq.ndim == 1:
        h = len(iq)//2; iq = np.array([iq[:h], iq[h:]])
    if iq.shape[0] != 2 and iq.shape[1] == 2:
        iq = iq.T
    return iq

def _feat_row(iq, selected_features, snr_hint=None):

    iq = _iq2row(iq)

    features = extract_features(iq)

    if "OriginalSNR" not in features:

        if snr_hint is not None:

            features["OriginalSNR"] = float(snr_hint)

        else:

            features["OriginalSNR"] = float(

                estimate_snr(

                    iq[0] + 1j * iq[1]

                )

            )

    dataframe = pd.DataFrame(

        [features]

    )

    # ---------------------------------------------------------
    # Validate Selected Features
    # ---------------------------------------------------------

    missing = [

        feature

        for feature in selected_features

        if feature not in dataframe.columns

    ]

    if len(missing):

        raise ValueError(

            "Missing Features:\n"

            + "\n".join(missing)

        )

    dataframe = dataframe[

        selected_features

    ].copy()

    dataframe.replace(

        [np.inf, -np.inf],

        np.nan,

        inplace=True

    )

    dataframe.fillna(

        0.0,

        inplace=True

    )

    dataframe = dataframe.astype(

        np.float64

    )

    return dataframe

def _ml_pred(model, feature_scaled, label_encoder):

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    prediction = model.predict(feature_scaled)[0]

    prediction = _decode_label(

        prediction,

        label_encoder

    )

    # ----------------------------------------------------------
    # Confidence
    # ----------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(

            feature_scaled

        )[0]

        confidence = float(

            np.max(probabilities)

        )

    else:

        confidence = 1.0

    return prediction, confidence

def _cnn_pred(cnn, iq, device, label_encoder):

    iq = _iq2row(iq)

    tensor = torch.tensor(
        iq,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = cnn(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    prediction = int(
        probabilities.argmax().item()
    )

    confidence = float(
        probabilities.max().item()
    )

    prediction = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return prediction, confidence

# The 9 modulation classes the models were trained on (SNR >= 0)
TRAINED_CLASSES = {"8PSK","AM-DSB","AM-SSB","BPSK","CPFSK","GFSK","PAM4","QAM64","QPSK"}
TRAINED_CLASS_ORDER = sorted(TRAINED_CLASSES)
MIN_SNR = 0


def _load_label_encoder():
    for path in [
        PROCESSED_DIR / "label_encoder.pkl",
        PROCESSED_DIR / "cnn_label_encoder.pkl",
    ]:
        if path.exists():
            encoder = joblib.load(path)
            if hasattr(encoder, "classes_"):
                return encoder

    encoder = LabelEncoder()
    encoder.fit(TRAINED_CLASS_ORDER)
    return encoder


def _decode_label(label, label_encoder):
    if isinstance(label, str):
        return label
    if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
        try:
            return label_encoder.inverse_transform([int(label)])[0]
        except Exception:
            pass
    return str(label)

def _pick_random(data, mods, snrs, n=5):
    """Pick n random signals from trained classes at SNR >= 0."""
    # filter keys to trained classes / reasonable SNR
    valid_keys = []
    for key in data.keys():
        if isinstance(key, tuple):
            mod, snr = key
            if mod in TRAINED_CLASSES and snr >= MIN_SNR:
                valid_keys.append(key)
        else:
            if str(key) in TRAINED_CLASSES:
                valid_keys.append(key)
    if not valid_keys:
        valid_keys = list(data.keys())   # fallback

    # try to pick one signal per unique modulation
    samples = []
    random.shuffle(valid_keys)
    used = set()
    for key in valid_keys:
        if len(samples) >= n:
            break
        mod = key[0] if isinstance(key, tuple) else str(key)
        if mod in used:
            continue
        used.add(mod)
        sigs = data[key]
        sig = sigs[random.randint(0, len(sigs)-1)]
        snr = key[1] if isinstance(key, tuple) else None
        samples.append((np.array(sig), mod, snr))
    # fill remaining with random picks
    while len(samples) < n:
        key = random.choice(valid_keys)
        sigs = data[key]
        sig = sigs[random.randint(0, len(sigs)-1)]
        mod = key[0] if isinstance(key, tuple) else str(key)
        snr = key[1] if isinstance(key, tuple) else None
        samples.append((np.array(sig), mod, snr))
    return samples[:n]

# ====================================================================
#  3. Display the 5 Selected Signals
# ====================================================================
def display_signals(samples):
    header("5 RANDOMLY SELECTED TEST SIGNALS")
    print(f"    {'#':<4} {'Modulation':<14} {'SNR':>6} {'Shape':>12} {'dtype':>10}")
    print("    " + "-" * 50)
    for i, (iq, mod, snr) in enumerate(samples, 1):
        iq2 = _iq2row(iq)
        snr_s = f"{snr} dB" if snr is not None else "N/A"
        print(f"    {i:<4} {mod:<14} {snr_s:>6} {str(iq2.shape):>12} {str(iq2.dtype):>10}")
    print()

# ====================================================================
#  4. Run Inference on All 5 Signals with All 3 Models
# ====================================================================
def run_inference(samples, rf, svm, cnn, sel, device, label_encoder, scaler):
    header("MODEL PREDICTIONS")
    results = []
    for i, (iq, true_mod, true_snr) in enumerate(samples, 1):
        iq2 = _iq2row(iq)
        fr  = _feat_row(iq2, sel, true_snr)
        
        fr_scaled = pd.DataFrame(

            scaler.transform(fr),

            columns=fr.columns

        ) if scaler else fr

        t0 = time.perf_counter()
        rf_prediction, rf_confidence = _ml_pred(rf,fr_scaled,label_encoder)
        rf_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        svm_prediction, svm_confidence = _ml_pred(svm,fr_scaled,label_encoder)
        svm_time = (time.perf_counter() - t0) * 1000

        cp, cc, ct = "N/A", 0.0, 0.0
        if cnn:
            t0 = time.perf_counter()
            cp, cc = _cnn_pred(cnn, iq2, device, label_encoder)
            ct = (time.perf_counter()-t0)*1000

        results.append(

    {

        "num": i,

        "iq": iq2,

        "true_mod": true_mod,

        "true_snr": true_snr,

        "rf_pred": rf_prediction,

        "rf_conf": rf_confidence,

        "rf_ms": rf_time,

        "svm_pred": svm_prediction,

        "svm_conf": svm_confidence,

        "svm_ms": svm_time,

        "cnn_pred": cp,

        "cnn_conf": cc,

        "cnn_ms": ct

    }

)

    # Print prediction table
    print(f"    {'#':<4} {'True':<12} {'RF Pred':<12} {'RF %':>6}  "
          f"{'SVM Pred':<12} {'SVM %':>6}  "
          f"{'CNN Pred':<12} {'CNN %':>6}")
    print("    " + "-" * 78)
    for r in results:
        rf_mark  = " *" if r["rf_pred"]  == r["true_mod"] else "  "
        svm_mark = " *" if r["svm_pred"] == r["true_mod"] else "  "
        cnn_mark = " *" if r["cnn_pred"] == r["true_mod"] else "  "
        print(f"    {r['num']:<4} {r['true_mod']:<12} "
              f"{r['rf_pred']:<12} {r['rf_conf']*100:>5.1f}%{rf_mark}  "
              f"{r['svm_pred']:<12} {r['svm_conf']*100:>5.1f}%{svm_mark}  "
              f"{r['cnn_pred']:<12} {r['cnn_conf']*100:>5.1f}%{cnn_mark}")
    print()
    print("    (* = correct prediction)")
    print()
    return results

# ====================================================================
#  5. Comparison Summary
# ====================================================================
def comparison_summary(results):
    header("MODEL COMPARISON")
    n = len(results)
    rf_correct  = sum(1 for r in results if r["rf_pred"]  == r["true_mod"])
    svm_correct = sum(1 for r in results if r["svm_pred"] == r["true_mod"])
    cnn_correct = sum(1 for r in results if r["cnn_pred"] == r["true_mod"])
    rf_avg_ms   = sum(r["rf_ms"]  for r in results) / n
    svm_avg_ms  = sum(r["svm_ms"] for r in results) / n
    cnn_avg_ms  = sum(r["cnn_ms"] for r in results) / n
    rf_avg_conf = sum(r["rf_conf"]  for r in results) / n
    svm_avg_conf= sum(r["svm_conf"] for r in results) / n
    cnn_avg_conf= sum(r["cnn_conf"] for r in results) / n

    print(f"    {'Metric':<24} {'Random Forest':>14} {'SVM':>14} {'CNN':>14}")
    print("    " + "-" * 66)
    print(f"    {'Correct / Total':<24} {'%d/%d' % (rf_correct,n):>14} {'%d/%d' % (svm_correct,n):>14} {'%d/%d' % (cnn_correct,n):>14}")
    print(f"    {'Accuracy on batch':<24} {rf_correct/n*100:>13.1f}% {svm_correct/n*100:>13.1f}% {cnn_correct/n*100:>13.1f}%")
    print(f"    {'Avg Confidence':<24} {rf_avg_conf*100:>13.1f}% {svm_avg_conf*100:>13.1f}% {cnn_avg_conf*100:>13.1f}%")
    print(f"    {'Avg Latency':<24} {rf_avg_ms:>11.1f} ms {svm_avg_ms:>11.1f} ms {cnn_avg_ms:>11.1f} ms")

    # trained accuracy from metadata
    print()
    print("    Trained Model Performance (full test set):")
    print("    " + "-" * 50)
    rf_meta  = MODEL_DIR / "random_forest_metadata.json"
    svm_meta = MODEL_DIR / "svm_metadata.json"
    cnn_hist = MODEL_DIR / "cnn_history.json"
    if rf_meta.exists():
        m = json.loads(rf_meta.read_text())
        print(f"    Random Forest  : {m['accuracy']*100:.2f}% accuracy  |  F1: {m['f1']*100:.2f}%")
    if svm_meta.exists():
        m = json.loads(svm_meta.read_text())
        print(f"    SVM            : {m['accuracy']*100:.2f}% accuracy  |  F1: {m['f1']*100:.2f}%")
    if cnn_hist.exists():
        h = json.loads(cnn_hist.read_text())
        print(f"    Residual CNN   : {max(h['val_accuracy'])*100:.2f}% best val accuracy")
    print()
    return rf_correct, svm_correct, cnn_correct

# ====================================================================
#  6. Final Verdict
# ====================================================================
def final_verdict(results, rf_ok, svm_ok, cnn_ok):
    header("FINAL VERDICT")
    n = len(results)

    # Determine best model
    scores = {"Random Forest": rf_ok, "SVM": svm_ok, "Residual CNN": cnn_ok}
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]

    # Majority vote per signal
    print("    Signal-wise Majority Vote:")
    print("    " + "-" * 50)
    majority_correct = 0
    for r in results:
        votes = [r["rf_pred"], r["svm_pred"], r["cnn_pred"]]
        vote_counts = Counter(votes)
        majority = vote_counts.most_common(1)[0][0]
        status = "CORRECT" if majority == r["true_mod"] else "WRONG"
        if majority == r["true_mod"]:
            majority_correct += 1
        print(f"    Signal {r['num']}  True: {r['true_mod']:<10}  "
              f"Majority: {majority:<10}  --> {status}")
    print()
    print(f"    Majority Vote Accuracy : {majority_correct}/{n} ({majority_correct/n*100:.0f}%)")
    print()
    print("    Best individual model on this batch:")
    print(f"    --> {best_name} ({best_score}/{n} correct)")
    print()

    # Unknown detection note
    low_conf = []
    for r in results:
        for mname, ck in [("RF", "rf_conf"), ("SVM", "svm_conf"), ("CNN", "cnn_conf")]:
            if r[ck] < CONFIDENCE_THRESHOLD:
                low_conf.append((r["num"], mname, r[ck]))
    if low_conf:
        print("    Low-confidence detections (threshold = %.0f%%):" % (CONFIDENCE_THRESHOLD*100))
        for num, mn, c in low_conf:
            print(f"      Signal {num} by {mn}: {c*100:.1f}% --> flagged as Unknown")
    else:
        print(f"    All predictions above confidence threshold ({CONFIDENCE_THRESHOLD*100:.0f}%)")
        print("    No signals flagged as Unknown")
    print()

# ====================================================================
#  7. Save Visualisation
# ====================================================================
def save_visualisation(samples, results):
    header("SAVING VISUALISATION")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("RF Signal Classification -- DRDO Showcase",
                 fontsize=16, fontweight="bold", y=0.98)

    n = len(samples)

    # Top row: IQ waveforms of all 5 signals
    for i, (iq, mod, snr) in enumerate(samples):
        ax = fig.add_subplot(4, n, i + 1)
        iq2 = _iq2row(iq)
        t = np.arange(iq2.shape[1])
        ax.plot(t, iq2[0], lw=0.7, color="royalblue", label="I")
        ax.plot(t, iq2[1], lw=0.7, color="tomato", alpha=0.8, label="Q")
        snr_s = f" ({snr}dB)" if snr is not None else ""
        ax.set_title(f"#{i+1} {mod}{snr_s}", fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # Second row: IQ constellation
    for i, (iq, mod, snr) in enumerate(samples):
        ax = fig.add_subplot(4, n, n + i + 1)
        iq2 = _iq2row(iq)
        ax.scatter(iq2[0], iq2[1], s=2, alpha=0.4, color="seagreen")
        ax.set_title("Constellation", fontsize=8)
        if i == 0:
            ax.set_ylabel("Q", fontsize=8)
        ax.set_xlabel("I", fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # Third row: confidence comparison bars for each signal
    for i, r in enumerate(results):
        ax = fig.add_subplot(4, n, 2*n + i + 1)
        models = ["RF", "SVM", "CNN"]
        confs  = [r["rf_conf"]*100, r["svm_conf"]*100, r["cnn_conf"]*100]
        preds  = [r["rf_pred"], r["svm_pred"], r["cnn_pred"]]
        colors = []
        for p in preds:
            colors.append("mediumseagreen" if p == r["true_mod"] else "indianred")
        bars = ax.bar(models, confs, color=colors, edgecolor="black", lw=0.5)
        ax.axhline(y=CONFIDENCE_THRESHOLD*100, color="navy", ls="--", lw=0.8)
        ax.set_ylim(0, 110)
        ax.set_title(f"True: {r['true_mod']}", fontsize=8, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Confidence %", fontsize=8)
        for bar, pred in zip(bars, preds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    pred, ha="center", fontsize=5.5, fontweight="bold")
        ax.tick_params(labelsize=6)

    # Fourth row: single summary bar chart
    ax = fig.add_subplot(4, 1, 4)
    n_res = len(results)
    rf_acc  = sum(1 for r in results if r["rf_pred"]  == r["true_mod"]) / n_res * 100
    svm_acc = sum(1 for r in results if r["svm_pred"] == r["true_mod"]) / n_res * 100
    cnn_acc = sum(1 for r in results if r["cnn_pred"] == r["true_mod"]) / n_res * 100
    names = ["Random Forest", "SVM", "Residual CNN"]
    accs  = [rf_acc, svm_acc, cnn_acc]
    cols  = ["steelblue", "salmon", "mediumseagreen"]
    bars  = ax.barh(names, accs, color=cols, edgecolor="black", lw=0.6, height=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{acc:.0f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy on 5-signal batch (%)", fontsize=10)
    ax.set_title("Model Accuracy Comparison", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "showcase_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved --> {out}")
    print()

# ====================================================================
#  Main
# ====================================================================
def main():
    os.system("cls" if os.name == "nt" else "clear")
    print()
    line("=")
    print("    RF SIGNAL CLASSIFICATION SYSTEM")
    print("    DRDO Project Showcase")
    line("=")
    print()

    t_start = time.perf_counter()

    # 1. Load
    data, mods, snrs = load_dataset()

    # 2. Models
    rf, svm, cnn, sel, device, label_encoder, scaler = load_models()

    # 3. Pick 5 random signals
    samples = _pick_random(data, mods, snrs, n=5)

    # 4. Show the signals
    display_signals(samples)

    # 5. Predict
    results = run_inference(samples,rf,svm,cnn,sel,device,label_encoder,scaler)

    # 6. Compare
    rf_ok, svm_ok, cnn_ok = comparison_summary(results)

    # 7. Verdict
    final_verdict(results, rf_ok, svm_ok, cnn_ok)

    # 8. Save plot
    save_visualisation(samples, results)

    # Done
    header("DONE")
    row("Total runtime", f"{time.perf_counter()-t_start:.1f}s")
    row("Output", str(OUTPUT_DIR / "showcase_results.png"))
    print()
    line("=")
    print()


if __name__ == "__main__":
    main()
