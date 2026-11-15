import os
import glob
import csv
import json
import subprocess

val_dir = r'D:\programming\Qwen3-VL-Training\qwen3vl_2b_finetune\dataset_grpo\Validation'
test_dir = r'D:\programming\Qwen3-VL-Training\qwen3vl_2b_finetune\dataset_grpo\Test'
cand_dir = r'D:\programming\Qwen3-VL-Training\qwen3vl_2b_finetune\dataset_sft\Candidate New Videos'

# 1. Read SFT anomaly CSV
sft_anom_csv = r'D:\programming\Qwen3-VL-Training\qwen3vl_2b_finetune\dataset_sft\anomalous_youtube_full_videos.csv'
anom_map = {}
if os.path.exists(sft_anom_csv):
    with open(sft_anom_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            anom_map[r['youtube_video_id']] = r

# 2. Read Candidate evaluation CSV
eval_cand_csv = os.path.join(CURRENT_DIR, 'complication_evaluation_dataset.csv')
cand_map = {}
if os.path.exists(eval_cand_csv):
    with open(eval_cand_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            cand_map[r['youtube_id']] = r

def get_video_info(video_path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration:format=duration,size',
        '-of', 'json', video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        stream = data['streams'][0] if 'streams' in data and data['streams'] else {}
        format_info = data.get('format', {})
        w = stream.get('width', 0)
        h = stream.get('height', 0)
        dur = float(stream.get('duration') or format_info.get('duration') or 0.0)
        sz = int(format_info.get('size') or os.path.getsize(video_path))
        return w, h, dur, sz
    except Exception as e:
        return 0, 0, 0.0, os.path.getsize(video_path) if os.path.exists(video_path) else 0

all_cases = []

# Process Validation videos
for p in sorted(os.listdir(val_dir)):
    vid_file = os.path.join(val_dir, p, 'full_video.mp4')
    if os.path.exists(vid_file):
        w, h, dur, sz = get_video_info(vid_file)
        is_anom = p in anom_map
        anom_info = anom_map.get(p, {})
        all_cases.append({
            'id': p,
            'split': 'Validation',
            'video_path': vid_file,
            'width': w, 'height': h, 'duration': dur, 'size_mb': sz / (1024*1024),
            'is_anomalous': is_anom,
            'categories': anom_info.get('anomaly_categories', 'Routine / Normal'),
            'title': anom_info.get('title', f'YouTube {p}')
        })

# Process Test videos
for p in sorted(os.listdir(test_dir)):
    vid_file = os.path.join(test_dir, p, 'full_video.mp4')
    if os.path.exists(vid_file):
        w, h, dur, sz = get_video_info(vid_file)
        is_anom = p in anom_map
        anom_info = anom_map.get(p, {})
        all_cases.append({
            'id': p,
            'split': 'Test',
            'video_path': vid_file,
            'width': w, 'height': h, 'duration': dur, 'size_mb': sz / (1024*1024),
            'is_anomalous': is_anom,
            'categories': anom_info.get('anomaly_categories', 'Routine / Normal'),
            'title': anom_info.get('title', f'YouTube {p}')
        })

# Process Candidate New Videos
for f in sorted(glob.glob(os.path.join(cand_dir, '*.mp4'))):
    vid_id = os.path.splitext(os.path.basename(f))[0]
    w, h, dur, sz = get_video_info(f)
    c_info = cand_map.get(vid_id, {})
    all_cases.append({
        'id': vid_id,
        'split': 'New_Candidate',
        'video_path': f,
        'width': w, 'height': h, 'duration': dur, 'size_mb': sz / (1024*1024),
        'is_anomalous': True,
        'categories': c_info.get('complication_category', 'Surgical Complication'),
        'title': c_info.get('title', f'CataractCoach {vid_id}')
    })

print(f"Total Video Cases cataloged: {len(all_cases)}")
val_count = sum(1 for c in all_cases if c['split'] == 'Validation')
test_count = sum(1 for c in all_cases if c['split'] == 'Test')
cand_count = sum(1 for c in all_cases if c['split'] == 'New_Candidate')
anom_total = sum(1 for c in all_cases if c['is_anomalous'])
routine_total = sum(1 for c in all_cases if not c['is_anomalous'])

print(f"  - Validation: {val_count} (Anomalous: {sum(1 for c in all_cases if c['split'] == 'Validation' and c['is_anomalous'])}, Routine: {sum(1 for c in all_cases if c['split'] == 'Validation' and not c['is_anomalous'])})")
print(f"  - Test:       {test_count} (Anomalous: {sum(1 for c in all_cases if c['split'] == 'Test' and c['is_anomalous'])}, Routine: {sum(1 for c in all_cases if c['split'] == 'Test' and not c['is_anomalous'])})")
print(f"  - Candidate:  {cand_count} (All Complication Cases)")
print(f"Total Anomalous: {anom_total}, Total Routine: {routine_total}")

# Save full audit
out_audit = os.path.join(CURRENT_DIR, 'dataset_full_videos_audit.json')
with open(out_audit, 'w', encoding='utf-8') as f:
    json.dump(all_cases, f, indent=2, ensure_ascii=False)
print(f"Audit saved to {out_audit}")
