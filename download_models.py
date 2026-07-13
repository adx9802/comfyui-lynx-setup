"""Download Kijai Wan 2.1 + Lynx model files for ComfyUI Desktop.

Shows a live overall progress bar (total GB across all model files) plus a
per-file status line. Run with --watch to only monitor an in-progress download
started elsewhere (no downloading of its own).
"""

from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from tqdm import tqdm


def build_downloads(models: str) -> list[dict]:
    """Return the list of files to fetch. `approx` is a fallback size (bytes)
    used only if the real size can't be read from Hugging Face."""
    return [
        {
            "repo": "Kijai/WanVideo_comfy_fp8_scaled",
            "filename": "T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
            "dest": os.path.join(models, "diffusion_models", "WanVideo", "fp8_scaled_kj", "T2V"),
            "label": "T2V base model (fp8 scaled)",
            "approx": 14_500_000_000,
        },
        {
            "repo": "Kijai/WanVideo_comfy",
            "filename": "Lynx/Wan2_1-T2V-14B-Lynx_lite_ip_layers_fp16.safetensors",
            # Lynx example workflow references the name without "-14B".
            "save_as": "Wan2_1-T2V-Lynx_lite_ip_layers_fp16.safetensors",
            "dest": os.path.join(models, "diffusion_models", "WanVideo", "lynx"),
            "label": "Lynx lite IP layers",
            "approx": 840_000_000,
        },
        {
            "repo": "Kijai/WanVideo_comfy",
            "filename": "Lynx/Wan2_1-T2V-14B-Lynx_full_ref_layers_fp16.safetensors",
            # Lynx example workflow references the name without "-14B".
            "save_as": "Wan2_1-T2V-Lynx_full_ref_layers_fp16.safetensors",
            "dest": os.path.join(models, "diffusion_models", "WanVideo", "lynx"),
            "label": "Lynx full ref layers",
            "approx": 4_200_000_000,
        },
        {
            "repo": "Kijai/WanVideo_comfy",
            "filename": "Lynx/lynx_lite_resampler_fp32.safetensors",
            "dest": os.path.join(models, "diffusion_models", "WanVideo", "lynx"),
            "label": "Lynx lite resampler",
            "approx": 330_000_000,
        },
        {
            "repo": "Kijai/WanVideo_comfy",
            "filename": "umt5-xxl-enc-bf16.safetensors",
            "dest": os.path.join(models, "text_encoders"),
            "label": "UMT5-XXL text encoder",
            "approx": 11_400_000_000,
        },
        {
            "repo": "Kijai/WanVideo_comfy",
            "filename": "Wan2_1_VAE_bf16.safetensors",
            "dest": os.path.join(models, "vae", "wanvideo"),
            "label": "Wan 2.1 VAE",
            "approx": 253_807_438,
        },
        {
            "repo": "Kijai/WanVideo_comfy",
            "filename": "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
            # Workflow references the name with a trailing underscore; save it as that.
            "save_as": "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16_.safetensors",
            "dest": os.path.join(models, "loras", "WanVideo", "Lightx2v"),
            "label": "LoRA Lightx2v (step distill)",
            "approx": 600_000_000,
        },
        {
            # Kijai's "_comfy" copy was removed; pull the original from alibaba-pai
            # and save it under the filename the Lynx workflow expects.
            "repo": "alibaba-pai/Wan2.1-Fun-Reward-LoRAs",
            "filename": "Wan2.1-Fun-14B-InP-HPS2.1.safetensors",
            "save_as": "Wan2.1-Fun-14B-InP-HPS2.1_reward_lora_comfy.safetensors",
            "dest": os.path.join(models, "loras", "WanVid", "funreward"),
            "label": "LoRA Fun reward (HPS2.1)",
            "approx": 300_000_000,
        },
    ]


def final_path(entry: dict) -> Path:
    name = entry.get("save_as") or os.path.basename(entry["filename"])
    return Path(entry["dest"]) / name


def is_complete(entry: dict) -> bool:
    fp = final_path(entry)
    return fp.exists() and fp.stat().st_size > 1000


def local_bytes(entry: dict) -> int:
    """Bytes present for this file: the finished file if it exists, otherwise
    the largest matching .incomplete part in the HF download cache."""
    fp = final_path(entry)
    if fp.exists() and fp.stat().st_size > 1000:
        return fp.stat().st_size
    cache = Path(entry["dest"]) / ".cache" / "huggingface" / "download"
    if cache.exists():
        sizes = [p.stat().st_size for p in cache.rglob("*.incomplete")]
        if sizes:
            return max(sizes)
    return 0


def remote_size(entry: dict) -> int:
    try:
        url = hf_hub_url(repo_id=entry["repo"], filename=entry["filename"])
        meta = get_hf_file_metadata(url)
        if meta.size:
            return int(meta.size)
    except Exception:
        pass
    return int(entry.get("approx", 0))


def current_total(entries: list[dict]) -> int:
    return sum(min(local_bytes(e), e["size"]) for e in entries)


def run_progress_bar(entries: list[dict], total: int, stop: threading.Event) -> None:
    bar = tqdm(
        total=total or None,
        initial=current_total(entries),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Overall models",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    try:
        while not stop.is_set():
            bar.n = min(current_total(entries), total) if total else current_total(entries)
            bar.refresh()
            time.sleep(1)
        bar.n = min(current_total(entries), total) if total else current_total(entries)
        bar.refresh()
    finally:
        bar.close()


def ensure_file(entry: dict) -> None:
    dest_dir = entry["dest"]
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = final_path(entry)
    local_path = Path(hf_hub_download(repo_id=entry["repo"], filename=entry["filename"], local_dir=dest_dir))
    # hf_hub_download may place the file in a nested subfolder (from the repo path)
    # and/or under a different name; normalize it to the workflow-expected name.
    if local_path.resolve() != dest_file.resolve():
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if dest_file.exists():
            dest_file.unlink()
        shutil.move(str(local_path), str(dest_file))
        # remove now-empty nested dir left behind (e.g. dest/Lightx2v/)
        try:
            local_path.parent.rmdir()
        except OSError:
            pass


def fmt_gb(n: int) -> str:
    return f"{n / 1_000_000_000:.2f} GB"


def print_summary(entries: list[dict]) -> None:
    print("\nFile status:")
    for e in entries:
        mark = "OK  " if is_complete(e) else "... "
        print(f"  [{mark}] {e['label']:<32} {fmt_gb(e['size'])}")


def watch_loop(entries: list[dict], total: int) -> None:
    print("Watching model download progress (Ctrl+C to stop)...")
    bar = tqdm(
        total=total or None,
        initial=current_total(entries),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Overall models",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    try:
        while True:
            bar.n = min(current_total(entries), total) if total else current_total(entries)
            bar.refresh()
            if all(is_complete(e) for e in entries):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bar.close()
    print_summary(entries)


def download(entries: list[dict], total: int) -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # our overall bar replaces per-file bars
    stop = threading.Event()
    monitor = threading.Thread(target=run_progress_bar, args=(entries, total, stop), daemon=True)
    monitor.start()
    n = len(entries)
    try:
        for i, e in enumerate(entries, 1):
            if is_complete(e):
                tqdm.write(f"[{i}/{n}] SKIP  {e['label']} (already downloaded)")
                continue
            tqdm.write(f"[{i}/{n}] GET   {e['label']} ({fmt_gb(e['size'])})")
            ensure_file(e)
            tqdm.write(f"[{i}/{n}] DONE  {e['label']}")
    finally:
        stop.set()
        monitor.join()
    print_summary(entries)
    print("\nALL DOWNLOADS COMPLETE")


def main() -> None:
    default_models = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Comfy-Desktop",
        "ComfyUI-Shared",
        "models",
    )
    parser = argparse.ArgumentParser(description="Download Lynx + Wan 2.1 models with a progress bar")
    parser.add_argument("--models-root", default=default_models)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Only display a live progress bar for an in-progress download; do not download.",
    )
    args = parser.parse_args()

    entries = build_downloads(args.models_root)
    print("Reading file sizes from Hugging Face...")
    for e in entries:
        e["size"] = remote_size(e)
    total = sum(e["size"] for e in entries)
    print(f"Total model set: {fmt_gb(total)} across {len(entries)} files\n")

    if args.watch:
        watch_loop(entries, total)
    else:
        download(entries, total)


if __name__ == "__main__":
    main()
