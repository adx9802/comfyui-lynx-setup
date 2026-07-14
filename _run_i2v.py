"""Build + run a Wan2.1 Fun-InP image-to-video workflow that PRESERVES the input
picture (background, face, styling) as the first frame and animates from it.

Usage:
  python _run_i2v.py [image] [width] [height] [frames] [steps] [start_latent_strength] [noise_aug]

The prompt graph is also written to i2v_preserve_workflow.json (API format).
"""
import json, time, urllib.request, urllib.error, uuid, sys, functools

print = functools.partial(print, flush=True)
SERVER = "http://127.0.0.1:8188"

IMAGE   = sys.argv[1] if len(sys.argv) > 1 else "IMG_1698.jpg"
W       = int(sys.argv[2]) if len(sys.argv) > 2 else 512
H       = int(sys.argv[3]) if len(sys.argv) > 3 else 512
FRAMES  = int(sys.argv[4]) if len(sys.argv) > 4 else 81
STEPS   = int(sys.argv[5]) if len(sys.argv) > 5 else 25
START_LATENT_STR = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0   # 1.0 = strongest adherence to the photo
NOISE_AUG        = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0   # small >0 adds a bit of motion

POS = "the subject stays in the same place with the same background, subtle natural motion, gentle head and hair movement, cinematic, high quality"
NEG = "new scene, different background, camera cut, teleport, morphing, distorted face, extra limbs, low quality, blurry, jpeg artifacts, watermark, text"

prompt = {
    "1": {"class_type": "WanVideoModelLoader", "inputs": {
        "model": "wan2.1_fun_inp_1.3B_bf16.safetensors",
        "base_precision": "bf16", "quantization": "disabled",
        "load_device": "offload_device", "attention_mode": "sdpa", "rms_norm_function": "default"}},
    "2": {"class_type": "WanVideoVAELoader", "inputs": {
        "model_name": "wanvideo\\Wan2_1_VAE_bf16.safetensors", "precision": "bf16"}},
    "4": {"class_type": "LoadImage", "inputs": {"image": IMAGE}},
    "6": {"class_type": "WanVideoImageToVideoEncode", "inputs": {
        "width": W, "height": H, "num_frames": FRAMES,
        "noise_aug_strength": NOISE_AUG, "start_latent_strength": START_LATENT_STR,
        "end_latent_strength": 1.0, "force_offload": True,
        "fun_or_fl2v_model": True,
        "vae": ["2", 0], "start_image": ["4", 0]}},
    "7": {"class_type": "WanVideoTextEncodeCached", "inputs": {
        "model_name": "umt5-xxl-enc-bf16.safetensors", "precision": "bf16",
        "positive_prompt": POS, "negative_prompt": NEG,
        "quantization": "disabled", "use_disk_cache": True, "device": "gpu"}},
    "8": {"class_type": "WanVideoSampler", "inputs": {
        "model": ["1", 0], "image_embeds": ["6", 0], "text_embeds": ["7", 0],
        "steps": STEPS, "cfg": 6.0, "shift": 5.0, "seed": 42,
        "force_offload": True, "scheduler": "unipc", "riflex_freq_index": 0,
        "rope_function": "comfy"}},
    "9": {"class_type": "WanVideoDecode", "inputs": {
        "vae": ["2", 0], "samples": ["8", 0], "enable_vae_tiling": False,
        "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
    "10": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["9", 0], "frame_rate": 16, "loop_count": 0,
        "filename_prefix": "i2v_preserve", "format": "video/h264-mp4",
        "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True,
        "pingpong": False, "save_output": True}},
}

with open("i2v_preserve_workflow.json", "w", encoding="utf-8") as f:
    json.dump(prompt, f, indent=2)
print("wrote i2v_preserve_workflow.json")

def post(path, payload):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())

def get(path):
    with urllib.request.urlopen(SERVER + path, timeout=15) as r:
        return json.loads(r.read().decode())

print(f"Submitting I2V ({IMAGE}, {W}x{H}x{FRAMES}, steps={STEPS}, start_latent_str={START_LATENT_STR}, noise_aug={NOISE_AUG})...")
try:
    resp = post("/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})
except urllib.error.HTTPError as e:
    print("SUBMIT FAILED:", e.read().decode("utf-8", "replace")); sys.exit(1)
pid = resp.get("prompt_id")
print("prompt_id:", pid)
if resp.get("node_errors"):
    print("NODE ERRORS:", json.dumps(resp["node_errors"], indent=2))

start = time.time()
while True:
    time.sleep(5)
    el = int(time.time() - start)
    hist = get(f"/history/{pid}")
    if pid in hist:
        st = hist[pid].get("status", {})
        ss = st.get("status_str"); done = st.get("completed")
        print(f"[{el}s] status: {ss} completed={done}")
        if done or ss in ("success", "error"):
            for nid, o in hist[pid].get("outputs", {}).items():
                for k, v in o.items():
                    print(f"  node {nid} {k}: {v}")
            if ss == "error":
                for m in st.get("messages", []):
                    if m[0] in ("execution_error",):
                        print("ERROR:", json.dumps(m)[:600])
            print("RESULT:", ss)
            break
    else:
        q = get("/queue")
        if len(q.get("queue_running", [])) == 0 and len(q.get("queue_pending", [])) == 0 and el > 15:
            print("Job left queue with no history -> worker crash (check log)."); break
    if el > 1200:
        print("Timed out."); break
