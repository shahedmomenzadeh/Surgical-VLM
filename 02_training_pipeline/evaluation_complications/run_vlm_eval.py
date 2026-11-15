import os
import json
import base64
import subprocess
import tempfile
import shutil
import openai

# Setup OpenAI client pointing to the local proxy
client = openai.OpenAI(
    base_url="http://localhost:20128/v1",
    api_key="none"
)

MODEL_NAME = "ag/gemini-3.8-flash"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)


def extract_uniform_frames(video_path, num_frames=6):
    """Uniformly extracts frames across video duration."""
    tmp_dir = tempfile.mkdtemp(prefix="gemini_eval_")
    frame_paths = []
    try:
        # Probe duration
        cmd_dur = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        res = subprocess.run(cmd_dur, capture_output=True, text=True)
        dur = float(res.stdout.strip()) if res.stdout.strip() else 10.0
        
        step = max(dur / (num_frames + 1), 0.3)
        timestamps = [step * (i + 1) for i in range(num_frames)]
        
        for idx, ts in enumerate(timestamps):
            out_img = os.path.join(tmp_dir, f"frame_{idx:02d}.jpg")
            cmd = [
                'ffmpeg', '-y', '-ss', str(ts), '-i', video_path,
                '-vframes', '1', '-q:v', '2', out_img
            ]
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(out_img) and os.path.getsize(out_img) > 500:
                frame_paths.append(out_img)
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
    return tmp_dir, frame_paths

def call_vlm(prompt_text, frame_paths):
    content = [{"type": "text", "text": prompt_text}]
    for fp in frame_paths:
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=300,
            stream=False
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

test_cases = [
    {
        "id": "Case 1 (Routine Step, Internal Dataset)",
        "video": os.path.join(REPO_ROOT, "dataset_sft", "Validation", "PH_0024_0009_S1", "clip_05.mp4"),
        "gt_task1": False,
        "gt_task2": "G",
        "task3_prompt": "Based on the surgical step shown in these sequential frames, what is the surgeon's immediate goal and action?\nA) Perform anterior vitrectomy\nB) Inject and position a foldable intraocular lens (IOL) into the capsular bag\nC) Perform relaxing cuts with micro-scissors for a capsule tear\nD) Cauterize iris vessels to stop active bleeding\n\nRespond strictly in JSON:\n{\n  \"explanation\": \"<1-2 sentences>\",\n  \"answer\": \"<Single Letter A-D>\"\n}",
        "gt_task3": "B"
    },
    {
        "id": "Case 2 (Argentinian Flag Sign / Radial Tear, Internal Dataset)",
        "video": os.path.join(REPO_ROOT, "dataset_sft", "Test", "2PnbLBFKiyQ", "clip_07.mp4"),
        "gt_task1": True,
        "gt_task2": "B",
        "task3_prompt": "When an uncontrolled radial tear / Argentinian flag sign occurs during capsulotomy, what is the correct immediate surgical action?\nA) Quickly withdraw all instruments and depressurize the eye\nB) Immediately increase phaco vacuum to maximum\nC) Maintain anterior chamber pressurization with high-viscosity OVD and make relaxing cuts with micro-scissors to redirect vector forces\nD) Proceed directly with standard high-pressure hydrodissection\n\nRespond strictly in JSON:\n{\n  \"explanation\": \"<1-2 sentences>\",\n  \"answer\": \"<Single Letter A-D>\"\n}",
        "gt_task3": "C"
    },
    {
        "id": "Case 3 (Intraocular Hemorrhage / Hyphema, Internal Dataset)",
        "video": os.path.join(REPO_ROOT, "dataset_sft", "Test", "_QXWa7QaEgk", "clip_02.mp4"),
        "gt_task1": True,
        "gt_task2": "F",
        "task3_prompt": "Given the active intraocular bleeding and red thrombus forming in the anterior chamber, what is the indicated management?\nA) Elevate infusion pressure to tamponade bleeding, followed by bimanual anterior chamber washout and clot aspiration\nB) Immediately implant an in-the-bag single piece IOL\nC) Perform a rapid anterior capsulorhexis without viscoelastic\nD) Rapidly decompress the anterior chamber to complete hypotony\n\nRespond strictly in JSON:\n{\n  \"explanation\": \"<1-2 sentences>\",\n  \"answer\": \"<Single Letter A-D>\"\n}",
        "gt_task3": "A"
    },
    {
        "id": "Case 4 (Posterior Capsule Rupture, New Candidate Video)",
        "video": os.path.join(REPO_ROOT, "dataset_sft", "Candidate New Videos", "KXDOVl0tLu8.mp4"),
        "gt_task1": True,
        "gt_task2": "A",
        "task3_prompt": "When the phaco probe breaks the posterior capsule as shown in this case, what is the primary critical rule to follow?\nA) Immediately pull the phaco probe out of the eye\nB) Do NOT abruptly withdraw the phaco probe; maintain infusion, inject dispersive viscoelastic through paracentesis to tamponade vitreous, then withdraw\nC) Increase aspiration flow rate to fish for remaining fragments\nD) Rapidly insert a single-piece acrylic IOL directly into the torn bag\n\nRespond strictly in JSON:\n{\n  \"explanation\": \"<1-2 sentences>\",\n  \"answer\": \"<Single Letter A-D>\"\n}",
        "gt_task3": "B"
    }
]

print("="*70)
print(f"EVALUATING MODEL: {MODEL_NAME} using OpenAI library on {client.base_url}")
print("="*70)

results = []

for case in test_cases:
    print(f"\n>>> Running Evaluation on: {case['id']}")
    print(f"    File: {os.path.basename(case['video'])}")
    
    tmp_dir, frames = extract_uniform_frames(case['video'], num_frames=6)
    print(f"    Extracted {len(frames)} frames for visual inspection.")
    
    # Task 1: Binary Detection
    p1 = """You are an expert ophthalmic surgeon evaluating a cataract surgery video clip presented as sequential frames.
TASK: Analyze these sequential frames. Is there any intraoperative complication, adverse event, or tissue anomaly present?
Respond strictly in JSON:
{
  "explanation": "<1-2 sentences on what you observe>",
  "complication_present": <true or false>
}"""
    t1_resp = call_vlm(p1, frames)
    
    # Task 2: Multi-class Identification
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
  "explanation": "<brief clinical justification>",
  "answer": "<Single Letter A-G>"
}"""
    t2_resp = call_vlm(p2, frames)
    
    # Task 3: Clinical Action & Rescue Reasoning
    p3 = f"""You are an expert ophthalmic surgeon analyzing sequential frames from a cataract surgery video clip.
TASK: {case['task3_prompt']}"""
    t3_resp = call_vlm(p3, frames)
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    print("\n  [Task 1 - Binary Complication Detection]:")
    print(f"    Model: {t1_resp}")
    print(f"    Ground Truth: complication_present = {case['gt_task1']}")
    
    print("\n  [Task 2 - Multi-Class Identification]:")
    print(f"    Model: {t2_resp}")
    print(f"    Ground Truth: {case['gt_task2']}")
    
    print("\n  [Task 3 - Clinical Action & Reasoning]:")
    print(f"    Model: {t3_resp}")
    print(f"    Ground Truth: {case['gt_task3']}")
    
    results.append({
        "case": case['id'],
        "file": os.path.basename(case['video']),
        "gt_task1": case['gt_task1'],
        "gt_task2": case['gt_task2'],
        "gt_task3": case['gt_task3'],
        "t1_resp": t1_resp,
        "t2_resp": t2_resp,
        "t3_resp": t3_resp
    })

# Save results to evaluation directory
out_path = os.path.join(CURRENT_DIR, "gemini_eval_report.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print(f"Evaluation finished successfully. Report saved to:\n  {out_path}")
print("="*70)
