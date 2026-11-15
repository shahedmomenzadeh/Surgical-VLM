import os
import json
import base64
import subprocess
import urllib.request
import tempfile
import shutil

API_URL = "http://localhost:20128/v1/chat/completions"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_NAME = "ag/gemini-3.8-flash"

def extract_frames(video_path, num_frames=6):
    """Extracts num_frames uniformly distributed across the video duration."""
    tmp_dir = tempfile.mkdtemp(prefix="vlm_eval_frames_")
    try:
        # Get duration
        cmd_dur = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        res = subprocess.run(cmd_dur, capture_output=True, text=True)
        dur = float(res.stdout.strip()) if res.stdout.strip() else 10.0
        
        # Calculate sample timestamps
        step = max(dur / (num_frames + 1), 0.5)
        timestamps = [step * (i + 1) for i in range(num_frames)]
        
        frame_paths = []
        for idx, ts in enumerate(timestamps):
            out_img = os.path.join(tmp_dir, f"frame_{idx:02d}.jpg")
            cmd = [
                'ffmpeg', '-y', '-ss', str(ts), '-i', video_path,
                '-vframes', '1', '-q:v', '2', out_img
            ]
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(out_img) and os.path.getsize(out_img) > 500:
                frame_paths.append(out_img)
        return tmp_dir, frame_paths
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return tmp_dir, []

def query_gemini(prompt_text, frame_paths):
    content = [{"type": "text", "text": prompt_text}]
    for fp in frame_paths:
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 350,
        "stream": False
    }
    
    req = urllib.request.Request(
        API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8")
    )
    
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        if hasattr(e, 'read'):
            return f"API ERROR: {e.code} - {e.read().decode('utf-8')}"
        return f"ERROR: {e}"

test_cases = [
    {
        "case_id": "Case 1: Routine Step (Internal Dataset)",
        "path": os.path.join(REPO_ROOT, "dataset_sft", "Validation", "PH_0024_0009_S1", "clip_05.mp4"),
        "ground_truth_anomaly": False,
        "ground_truth_class": "G) Normal / Routine (No Adverse Event)",
        "task3_question": "Based on the surgical step shown, what is the surgeon's immediate goal and action?\nA) Perform anterior vitrectomy\nB) Inject and orient a foldable IOL into the capsular bag\nC) Perform relaxing cuts with micro-scissors for a capsule tear\nD) Cauterize iris vessels to stop active bleeding",
        "task3_gt": "B"
    },
    {
        "case_id": "Case 2: Argentinian Flag Sign (Internal Dataset)",
        "path": os.path.join(REPO_ROOT, "dataset_sft", "Test", "2PnbLBFKiyQ", "clip_07.mp4"),
        "ground_truth_anomaly": True,
        "ground_truth_class": "B) Anterior Capsule Radial Tear / Argentinian Flag Sign",
        "task3_question": "When an uncontrolled radial tear / Argentinian flag sign occurs during capsulotomy, what is the correct immediate surgical action?\nA) Quickly withdraw all instruments and depressurize the eye\nB) Immediately increase phaco vacuum to maximum\nC) Maintain anterior chamber pressurization with high-viscosity OVD and make relaxing cuts with micro-scissors to redirect vector forces\nD) Proceed directly with standard high-fluidic hydrodissection",
        "task3_gt": "C"
    },
    {
        "case_id": "Case 3: Intraocular Hemorrhage (Internal Dataset)",
        "path": os.path.join(REPO_ROOT, "dataset_sft", "Test", "_QXWa7QaEgk", "clip_02.mp4"),
        "ground_truth_anomaly": True,
        "ground_truth_class": "F) Intraocular Hemorrhage / Hyphema",
        "task3_question": "Given the active intraocular bleeding and red thrombus forming in the anterior chamber, what is the indicated management?\nA) Elevate IOP / infusion pressure to tamponade bleeding, followed by bimanual anterior chamber washout and clot aspiration\nB) Immediately implant an in-the-bag single piece IOL\nC) Perform a rapid anterior capsulorhexis without viscoelastic\nD) Rapidly decompress the anterior chamber to hypotony",
        "task3_gt": "A"
    },
    {
        "case_id": "Case 4: Posterior Capsule Rupture (New Downloaded Video)",
        "path": os.path.join(REPO_ROOT, "dataset_sft", "Candidate New Videos", "KXDOVl0tLu8.mp4"),
        "ground_truth_anomaly": True,
        "ground_truth_class": "A) Posterior Capsule Rupture (PCR)",
        "task3_question": "When the phaco probe breaks the posterior capsule as shown in this case, what is the primary critical rule to follow?\nA) Immediately yank the phaco probe out of the eye\nB) Do NOT abruptly withdraw the phaco probe; maintain infusion, inject dispersive viscoelastic through paracentesis to tamponade vitreous, then withdraw\nC) Increase aspiration flow rate to fish for remaining fragments\nD) Rapidly insert a single-piece acrylic IOL directly into the torn bag",
        "task3_gt": "B"
    }
]

print("="*70)
print(f"EVALUATING MODEL: {MODEL_NAME} on {API_URL}")
print("="*70)

all_results = []

for case in test_cases:
    print(f"\nEvaluating: {case['case_id']}")
    print(f"File: {os.path.basename(case['path'])}")
    
    tmp_dir, frames = extract_frames(case['path'], num_frames=6)
    print(f"Extracted {len(frames)} frames for visual temporal inspection.")
    
    # Task 1: Binary Detection
    p1 = """You are an expert ophthalmic surgeon evaluating a cataract surgery video clip presented as sequential frames.
TASK: Analyze these sequential frames. Is there any intraoperative complication, adverse event, or tissue anomaly present?
Respond strictly in JSON:
{
  "explanation": "<1-2 sentences on what you observe in the eye and instruments>",
  "complication_present": <true or false>
}"""
    res1 = query_gemini(p1, frames)
    
    # Task 2: Closed-set Multi-Class Identification
    p2 = """You are an expert ophthalmic surgeon analyzing sequential frames from a cataract surgery video clip.
TASK: Identify the specific intraoperative situation shown:
A) Posterior Capsule Rupture (PCR)
B) Anterior Capsule Radial Tear / Argentinian Flag Sign
C) Zonular Dialysis / Dehiscence / Phacodonesis
D) Vitreous Prolapse / Anterior Vitrectomy
E) Iris Prolapse / Intraoperative Floppy Iris (IFIS)
F) Intraocular Hemorrhage / Hyphema
G) Normal / Routine (No Adverse Event)

Respond strictly in JSON:
{
  "explanation": "<brief clinical justification based on anatomy and tools>",
  "answer": "<Single Letter A-G>"
}"""
    res2 = query_gemini(p2, frames)
    
    # Task 3: Clinical Action & Rescue Reasoning
    p3 = f"""You are an expert ophthalmic surgeon analyzing sequential frames from a cataract surgery video clip.
TASK: {case['task3_question']}

Respond strictly in JSON:
{{
  "explanation": "<1-2 sentences clinical justification>",
  "answer": "<Single Letter A-D>"
}}"""
    res3 = query_gemini(p3, frames)
    
    # Clean up frames
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    result_entry = {
        "case": case['case_id'],
        "file": os.path.basename(case['path']),
        "gt_anomaly": case['ground_truth_anomaly'],
        "gt_class": case['ground_truth_class'],
        "task1_response": res1,
        "task2_response": res2,
        "task3_response": res3
    }
    all_results.append(result_entry)
    
    print("\n--- RESULTS ---")
    print("TASK 1 (Binary Complication):", res1)
    print("TASK 2 (Multi-Class ID):     ", res2)
    print("TASK 3 (Clinical Reasoning): ", res3)

# Save evaluation output
out_file = os.path.join(CURRENT_DIR, "gemini_38_flash_evaluation_results.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\nAll results saved to: {out_file}")
