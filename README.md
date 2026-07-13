# comfyui-lynx-setup

Automated setup for **ByteDance Lynx** human identity-preserving video generation on **ComfyUI Desktop** (Windows, NVIDIA GPU).

Installs custom nodes, optional Visual Studio C++ Build Tools, Python dependencies, and downloads ~35GB of Wan 2.1 + Lynx models from [Kijai's Hugging Face repos](https://huggingface.co/Kijai/WanVideo_comfy).

## Requirements

- Windows 10/11
- ComfyUI Desktop with a local GPU install (tested with NVIDIA RTX 4090, 24GB VRAM)
- Git
- ~80GB free disk space
- Internet connection for model downloads

## Quick start

1. Close Comfy Desktop.
2. Clone this repo.
3. Double-click `RUN_LYNX_SETUP.bat` **or** run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_lynx_comfyui.ps1
```

4. Restart Comfy Desktop.
5. Load `lynx_workflow.json` from your Desktop (copied automatically).

## Default paths (ComfyUI Desktop)

| Setting | Default |
|---|---|
| ComfyUI root | `D:\LocalAI\LocalAI\ComfyUI` |
| Python | `<ComfyUI>\.venv\Scripts\python.exe` |
| Models | `%LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Shared\models` |
| Log | `%USERPROFILE%\lynx_setup_log.txt` |

Override with environment variables:

```powershell
$env:LYNX_COMFY_ROOT = "D:\path\to\ComfyUI"
$env:LYNX_MODELS_ROOT = "C:\path\to\models"
powershell -File .\setup_lynx_comfyui.ps1
```

Or copy `config.example.ps1` to `config.ps1`, edit paths, and dot-source before running.

## Script flags

```powershell
.\setup_lynx_comfyui.ps1 -SkipBuildTools   # skip VS C++ Build Tools install
.\setup_lynx_comfyui.ps1 -SkipModels       # nodes/deps only, no HF downloads
```

## What gets installed

**Custom nodes**
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

**Acceleration deps**
- [triton-windows](https://github.com/woct0rdho/triton-windows) (`torch.compile` backend; pinned `<3.7` for torch 2.10/2.11)
- [SageAttention](https://github.com/woct0rdho/SageAttention) (fast attention; the workflow's model loader uses `attention_mode=sageattn`)

**Models (~35GB)**
- Wan 2.1 T2V 14B (FP8 scaled)
- Lynx lite IP + full ref layers + resampler
- umt5 text encoder, Wan VAE, LightX2V + Fun LoRAs

## RTX 4090 settings

The bundled workflow uses:
- 832×480, 121 frames (~5 sec @ 24fps)
- Block swap: 35
- IP scale: 0.7, Ref scale: 0.6
- 6 steps with LightX2V distilled LoRA

## Troubleshooting

**`TritonMissing` / `No module named 'triton'` / `No module named 'sageattention'`**
- The Lynx workflow uses `torch.compile` (needs Triton) and SageAttention. Install both prebuilt Windows wheels into the ComfyUI venv:
```powershell
& "D:\LocalAI\LocalAI\ComfyUI\.venv\Scripts\python.exe" -m pip install "triton-windows<3.7"
& "D:\LocalAI\LocalAI\ComfyUI\.venv\Scripts\python.exe" -m pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post5/sageattention-2.2.0+cu130torch2.10.0andhigher.post5-cp310-abi3-win_amd64.whl"
```
- Match the wheel to your CUDA: use `cu130` for torch `+cu130`, `cu128` for torch `+cu12x`.
- Alternative (no install): in the workflow, remove/bypass the Torch Compile settings node and set the model loader `attention_mode` to `sdpa`. Slower, but avoids Triton/SageAttention entirely.

**numpy / compiler errors on Python 3.13**
- The script installs VS 2022 C++ Build Tools automatically.
- Reboot after Build Tools install, then re-run.
- `facenet-pytorch` is intentionally skipped (breaks numpy on 3.13).

**HF rate limits**
```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

**insightface optional**
- Only needed for auto face-crop node.
- Lynx generation works with manual face crop if insightface fails.

## License

Setup scripts: MIT. Model weights follow their respective upstream licenses (Kijai repos, ByteDance Lynx, Alibaba Wan).
