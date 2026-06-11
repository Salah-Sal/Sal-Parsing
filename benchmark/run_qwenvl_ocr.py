"""Generic Qwen-VL OCR runner (works for QARI / Qwen2-VL and Qwen2.5-VL fine-tunes).

Usage:
  python run_qwenvl_ocr.py --model <hf_id> --image <png> --out <md> \
      --prompt "<ocr prompt>" [--max-pixels 1500000] [--max-new-tokens 4096]

Records wall time and peak MPS memory to <out>.meta.json. Designed for a
16 GB Apple-silicon machine: caps image pixels to bound vision-token memory.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-pixels", type=int, default=1_500_000)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if dev == "mps":
        torch.mps.empty_cache()

    t0 = time.time()
    # cap pixels at the processor level so the vision tower emits fewer tokens
    processor = AutoProcessor.from_pretrained(
        a.model, max_pixels=a.max_pixels, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    t_load = time.time() - t0

    img = Image.open(a.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": a.prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(dev)
    n_vis_tok = int(inputs["input_ids"].shape[1])

    t1 = time.time()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
    out_ids = gen[0][inputs["input_ids"].shape[1]:]
    md = processor.decode(out_ids, skip_special_tokens=True)
    t_infer = time.time() - t1

    peak = (torch.mps.current_allocated_memory() / 1e9) if dev == "mps" else 0.0
    driver_peak = (torch.mps.driver_allocated_memory() / 1e9) if dev == "mps" else 0.0

    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(md, encoding="utf-8")
    meta = {
        "model": a.model, "device": dev, "image": a.image,
        "input_tokens": n_vis_tok, "output_chars": len(md),
        "load_s": round(t_load, 1), "infer_s": round(t_infer, 1),
        "mps_alloc_gb": round(peak, 2), "mps_driver_gb": round(driver_peak, 2),
        "max_pixels": a.max_pixels, "max_new_tokens": a.max_new_tokens,
    }
    outp.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
