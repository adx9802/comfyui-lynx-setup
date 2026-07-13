"""Download Kijai Wan 2.1 + Lynx model files for ComfyUI Desktop."""

from __future__ import annotations

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download


def ensure_file(repo: str, filename: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    basename = os.path.basename(filename)
    dest_file = os.path.join(dest_dir, basename)
    if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1000:
        print(f"SKIP: {dest_file}")
        return dest_file

    print(f"DOWNLOAD: {repo}/{filename}")
    local_path = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)
    nested = os.path.join(dest_dir, filename.replace("/", os.sep))
    if os.path.exists(nested) and os.path.normpath(nested) != os.path.normpath(dest_file):
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.move(nested, dest_file)
    elif os.path.basename(local_path) == basename:
        if os.path.normpath(local_path) != os.path.normpath(dest_file):
            shutil.move(local_path, dest_file)
    print(f"DONE: {dest_file}")
    return dest_file


def main() -> None:
    default_models = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Comfy-Desktop",
        "ComfyUI-Shared",
        "models",
    )
    parser = argparse.ArgumentParser(description="Download Lynx + Wan 2.1 models")
    parser.add_argument("--models-root", default=default_models)
    args = parser.parse_args()
    models = args.models_root

    downloads = [
        (
            "Kijai/WanVideo_comfy_fp8_scaled",
            "T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
            os.path.join(models, "diffusion_models", "WanVideo", "fp8_scaled_kj", "T2V"),
        ),
        (
            "Kijai/WanVideo_comfy",
            "Lynx/Wan2_1-T2V-14B-Lynx_lite_ip_layers_fp16.safetensors",
            os.path.join(models, "diffusion_models", "WanVideo", "lynx"),
        ),
        (
            "Kijai/WanVideo_comfy",
            "Lynx/Wan2_1-T2V-14B-Lynx_full_ref_layers_fp16.safetensors",
            os.path.join(models, "diffusion_models", "WanVideo", "lynx"),
        ),
        (
            "Kijai/WanVideo_comfy",
            "Lynx/lynx_lite_resampler_fp32.safetensors",
            os.path.join(models, "diffusion_models", "WanVideo", "lynx"),
        ),
        (
            "Kijai/WanVideo_comfy",
            "umt5-xxl-enc-bf16.safetensors",
            os.path.join(models, "text_encoders"),
        ),
        (
            "Kijai/WanVideo_comfy",
            "Wan2_1_VAE_bf16.safetensors",
            os.path.join(models, "vae", "wanvideo"),
        ),
        (
            "Kijai/WanVideo_comfy",
            "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16_.safetensors",
            os.path.join(models, "loras", "WanVideo", "Lightx2v"),
        ),
        (
            "Kijai/WanVideo_comfy",
            "Fun/Wan2.1-Fun-14B-InP-HPS2.1_reward_lora_comfy.safetensors",
            os.path.join(models, "loras", "WanVid", "funreward"),
        ),
    ]

    for repo, filename, dest in downloads:
        ensure_file(repo, filename, dest)

    print("ALL DOWNLOADS COMPLETE")


if __name__ == "__main__":
    main()
