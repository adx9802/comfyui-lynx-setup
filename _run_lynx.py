import json, time, urllib.request, urllib.error, uuid, sys, functools

print = functools.partial(print, flush=True)

SERVER = "http://127.0.0.1:8188"
DUMP = r"C:\Users\aho\.cursor\projects\c-Users-aho-comfyui-lynx-setup\agent-tools\8aca6235-a1f8-4a1d-a9b9-4cc86701cb64.txt"

# CLI: width height frames [blocks_to_swap] [quantization]
W = int(sys.argv[1]) if len(sys.argv) > 1 else 480
H = int(sys.argv[2]) if len(sys.argv) > 2 else 272
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 49
BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
QUANT = sys.argv[5] if len(sys.argv) > 5 else "disabled"

def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))

def get(path):
    with urllib.request.urlopen(SERVER + path, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))

# --- Build the API prompt from the known-good dump ---
with open(DUMP, encoding="utf-8") as f:
    prompt = json.load(f)["queue_running"][0][2]

# Edit 1: dimensions
prompt["24"]["inputs"].update({"width": W, "height": H, "num_frames": FRAMES})
# Edit 2: aggressive block swap; synchronous offload so each block's VRAM is
# freed before the next loads (non_blocking=True causes accumulation -> OOM).
prompt["29"]["inputs"].update({"blocks_to_swap": BLOCKS, "offload_img_emb": True,
                               "offload_txt_emb": True, "prefetch_blocks": 0,
                               "use_non_blocking": False})
# Edit 2b: optional fp8 quantization to halve weight memory
prompt["12"]["inputs"]["quantization"] = QUANT
# Edit 3: bypass full-ref adapter -> use lite IP only
prompt["12"]["inputs"]["extra_model"] = ["69", 0]
prompt.pop("13", None)
# Edit 3b: drop only ref_image so ref_latent=None (skips ref-extraction/OOM);
# keep ref_text_embed so the sampler's dict_to_device() doesn't hit None.
prompt["55"]["inputs"].pop("ref_image", None)

client_id = str(uuid.uuid4())
print(f"Submitting (lite-IP only, {W}x{H}x{FRAMES}, blocks_to_swap={BLOCKS}, quant={QUANT})...")
try:
    resp = post("/prompt", {"prompt": prompt, "client_id": client_id})
except urllib.error.HTTPError as e:
    print("SUBMIT FAILED:", e.read().decode("utf-8", "replace"))
    sys.exit(1)

pid = resp.get("prompt_id")
print("prompt_id:", pid, "| queue number:", resp.get("number"))
if resp.get("node_errors"):
    print("NODE ERRORS:", json.dumps(resp["node_errors"], indent=2))

# --- Poll ---
start = time.time()
last = ""
while True:
    time.sleep(5)
    el = int(time.time() - start)
    hist = get(f"/history/{pid}")
    if pid in hist:
        st = hist[pid].get("status", {})
        status_str = st.get("status_str")
        completed = st.get("completed")
        print(f"[{el}s] history status: {status_str} completed={completed}")
        if completed or status_str in ("success", "error"):
            outs = hist[pid].get("outputs", {})
            print("OUTPUTS:")
            for nid, o in outs.items():
                for k, v in o.items():
                    print(f"  node {nid} {k}: {v}")
            if status_str == "error":
                for m in st.get("messages", []):
                    print("MSG:", json.dumps(m)[:500])
            print("RESULT:", "SUCCESS" if status_str == "success" or completed else status_str)
            break
    else:
        q = get("/queue")
        run = len(q.get("queue_running", []))
        pend = len(q.get("queue_pending", []))
        msg = f"[{el}s] not in history yet | running={run} pending={pend}"
        if msg[7:] != last:
            print(msg); last = msg[7:]
        if run == 0 and pend == 0 and el > 15:
            print("Job left the queue but produced no history -> likely worker crash (OOM).")
            break
    if el > 900:
        print("Timed out after 15 min.")
        break
