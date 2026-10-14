"""Recover saved figures and rounded training metrics without training or a checkpoint."""
import base64
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "voxceleb_training_VGG16_lite.ipynb"

def source(cell):
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else value

def text(value):
    return "".join(value) if isinstance(value, list) else value

def chart(rows):
    # SVG uses only recorded numbers, with epochs numbered 1 through 50.
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="470" viewBox="0 0 1200 470">',
             '<rect width="1200" height="470" fill="white"/>',
             '<g font-family="sans-serif" fill="#172033">',
             '<text x="600" y="28" text-anchor="middle" font-size="21">VoxCeleb speaker classification: recorded training</text>',
             '<text x="600" y="451" text-anchor="middle" font-size="13">Reconstructed from rounded notebook logs; best validation accuracy: 86.60% at epoch 45</text>']
    for panel, keys, title, maximum, unit in [
        (0, ("train_loss", "validation_loss"), "Cross-entropy loss", 7, ""),
        (1, ("train_accuracy_percent", "validation_accuracy_percent"), "Accuracy (%)", 100, "%")
    ]:
        left, top, width, height = 70 + panel * 590, 85, 490, 290
        parts.append(f'<text x="{left+width/2}" y="62" text-anchor="middle" font-size="17">{title}</text>')
        for n in range(6):
            value = maximum * n / 5
            y = top + height * (1 - n / 5)
            parts.append(f'<path d="M {left} {y} H {left+width}" stroke="#dce1e8"/>')
            parts.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-size="12">{value:g}{unit}</text>')
        for epoch in [1, 10, 20, 30, 40, 50]:
            x = left + (epoch - 1) * width / 49
            parts.append(f'<text x="{x}" y="{top+height+21}" text-anchor="middle" font-size="12">{epoch}</text>')
        parts.append(f'<text x="{left+width/2}" y="{top+height+43}" text-anchor="middle" font-size="13">Epoch</text>')
        for key, color, label, offset in zip(keys, ["#2563eb", "#ea580c"], ["Training", "Validation"], [0, 120]):
            points = " ".join(f"{left+(r['epoch']-1)*width/49:.2f},{top+height*(1-r[key]/maximum):.2f}" for r in rows)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            parts.append(f'<path d="M {left+offset} 76 h 22" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="{left+offset+28}" y="80" font-size="12">{label}</text>')
        if panel == 1:
            x, y = left + 44 * width / 49, top + height * (1 - 86.60 / 100)
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#15803d"/>')
    parts.append('</g></svg>')
    return "\n".join(parts)

def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assets, results = ROOT / "assets", ROOT / "results"
    assets.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    saved = []
    for cell in notebook["cells"]:
        code = source(cell)
        for name in ["sample_spectrograms.png", "inference_example.png"]:
            if f"savefig('{name}'" in code:
                image = next((o.get("data", {}).get("image/png") for o in cell.get("outputs", []) if o.get("data", {}).get("image/png")), None)
                if image:
                    (assets / name).write_bytes(base64.b64decode(text(image)))
                    saved.append(name)
    logs = "\n".join(text(o.get("text", "")) for c in notebook["cells"] for o in c.get("outputs", []) if o.get("output_type") == "stream")
    pattern = r"Epoch (\d+)/50\s*-+\s*Train Loss:\s*([\d.]+) \| Train Acc:\s*([\d.]+)%\s*Val Loss:\s*([\d.]+) \| Val Acc:\s*([\d.]+)%\s*Epoch Time:\s*([\d.]+) min"
    keys = ["epoch", "train_loss", "train_accuracy_percent", "validation_loss", "validation_accuracy_percent", "epoch_time_minutes_rounded"]
    rows = [dict(zip(keys, [int(m[0])] + [float(v) for v in m[1:]])) for m in re.findall(pattern, logs)]
    if [r["epoch"] for r in rows] != list(range(1, 51)):
        raise ValueError("Expected complete saved epoch logs for epochs 1 through 50.")
    payload = {"provenance": "Reconstructed from rounded saved stdout; epoch times are rounded minutes.", "epochs": rows}
    (results / "epoch_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (results / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (results / "training_log.txt").write_text(logs + "\n", encoding="utf-8")
    (assets / "training_history_reconstructed.svg").write_text(chart(rows), encoding="utf-8")
    print(f"Recovered {len(rows)} epochs and figures: {', '.join(saved)}")
    print("The empty training plot and failed confusion matrix were not exported.")

if __name__ == "__main__":
    main()
