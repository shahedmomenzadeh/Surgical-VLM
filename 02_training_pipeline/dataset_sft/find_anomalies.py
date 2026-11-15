import os
import glob
import json
import re

base = r'D:\programming\Qwen3-VL-Training\qwen3vl_2b_finetune\dataset_sft'

meta_files = [f for f in glob.glob(os.path.join(base, '**', '*.jsonl'), recursive=True) if not f.endswith('_sft.jsonl')]
sft_files = glob.glob(os.path.join(base, '**', '*_sft.jsonl'), recursive=True)

all_clips = {}
for mf in meta_files:
    clip_key = mf.replace('\\', '/')
    all_clips[clip_key] = {'meta_path': mf, 'sft_path': mf[:-6] + '_sft.jsonl'}

for sf in sft_files:
    meta_p = sf[:-10] + '.jsonl'
    clip_key = meta_p.replace('\\', '/')
    if clip_key not in all_clips:
        all_clips[clip_key] = {'meta_path': meta_p, 'sft_path': sf}

print(f"Total unique clips found: {len(all_clips)}")

rules = [
    ('Posterior Capsule Rupture / Tear', re.compile(r'\b(posterior\s+capsul\w*\s+ruptur\w*|pc\s+tear|posterior\s+tear|capsular\s+rupture|tear\s+in\s+the\s+posterior\s+capsule)\b', re.I)),
    ('Radial / Anterior Capsular Tear / Argentinian Flag Sign', re.compile(r'\b(radial\s+tear|tear\s+run[\s-]out|capsular\s+runout|argentin\w*|cut\w*\s+the\s+edges\s+of\s+the\s+torn\s+anterior\s+capsule)\b', re.I)),
    ('Vitreous Prolapse / Anterior Vitrectomy', re.compile(r'\b(vitrectom\w*|vitreous\s+loss|vitreous\s+prolapse|prolapsed\s+vitreous)\b', re.I)),
    ('Zonular Dialysis / Dehiscence / CTR / CTS Insertion', re.compile(r'\b(zonul\w*\s+(dialysis|dehiscence|laxity|weakness|defect|break)|capsular\s+tension\s+(ring|segment)|\bctr\b|\bcts\b)\b', re.I)),
    ('Lens Subluxation / Dislocation / Dropped Nucleus', re.compile(r'\b(dislocated\s+(crystalline\s+)?lens|subluxat\w*|lens\s+dislocation|drop\w*\s+nucleus|nucleus\s+drop\w*|dropped\s+fragment)\b', re.I)),
    ('Iris Prolapse / Trauma / Floppy Iris (IFIS)', re.compile(r'\b(iris\s+prolaps\w*|iris\s+damage|iris\s+trauma|floppy\s+iris|\bifis\b|persistent\s+iris\s+prolapse|iris\s+chaf\w*)\b', re.I)),
    ('Intraocular Bleeding / Hyphema / Iris Bleeding', re.compile(r'\b(hyphema|blood\s+reflux\w*|refluxes\s+from\s+the\s+Schlemm|filling?\s+the\s+anterior\s+chamber|blood\s+from\s+the\s+iris|washout\s+of\s+(fresh\s+)?blood|blood\s+clot|layered\s+blood|bleeding\s+at\s+the\s+pupillary\s+margin|bleeding\s+source\s+in\s+the\s+angle)\b', re.I)),
    ('Corneal Edema / Descemet Detachment / Wound Burn', re.compile(r'\b(descemet\w*\s+detach\w*|wound\s+burn|thermal\s+burn|corneal\s+edema|edematous\s+cornea)\b', re.I)),
    ('Subconjunctival / Limbal Hemorrhage / Incision Bleed', re.compile(r'\b(subconjunctival\s+hemorrhag\w*|conjunctival\s+hemorrhag\w*|bleeding\s+from\s+limbal\s+vessels|bleeding\s+at\s+the\s+entry\s+site|blood\s+at\s+the\s+incision\s+site|episcleral\s+bleeding|petechial\s+hemorrhages)\b', re.I)),
    ('Pupil Constriction / Iris Suturing / Sphincterotomy / Synechia', re.compile(r'\b(synechi\w*|pupil\s+constrict\w*|iris\s+suture|sphincterotom\w*)\b', re.I))
]

detected = []

for k, paths in all_clips.items():
    mf = paths['meta_path']
    sf = paths['sft_path']
    
    meta_data = {}
    if os.path.exists(mf):
        try:
            with open(mf, 'r', encoding='utf-8') as f:
                meta_data = json.loads(f.readline())
        except Exception:
            pass
        
    sft_data = {}
    if os.path.exists(sf):
        try:
            with open(sf, 'r', encoding='utf-8') as f:
                sft_data = json.loads(f.readline())
        except Exception:
            pass

    clip_id = meta_data.get('clip_id') or os.path.splitext(os.path.basename(mf))[0]
    parent_id = meta_data.get('parent_video_id', '')
    video_path = meta_data.get('video_path') or meta_data.get('clip_filename', '')
    
    split = 'Train' if '/Train/' in k else ('Validation' if '/Validation/' in k else ('Test' if '/Test/' in k else 'Unknown'))

    texts = []
    if meta_data.get('visual_description'): texts.append(meta_data['visual_description'])
    if meta_data.get('transcript_context'): texts.append(meta_data['transcript_context'])
    if meta_data.get('step_title'): texts.append(meta_data['step_title'])
    
    for m in sft_data.get('messages', []):
        if m.get('role') == 'assistant':
            texts.append(m.get('content', ''))
        elif m.get('role') == 'user':
            for c in m.get('content', []):
                if c.get('type') == 'text':
                    texts.append(c.get('text', ''))
                    
    full_text = ' '.join(texts)
    
    clip_anomalies = []
    for label, pat in rules:
        matches = pat.findall(full_text)
        if matches:
            clean_matches = []
            for match in matches:
                m_str = match if isinstance(match, str) else match[0]
                # Filter negative mentions like "without visible hemorrhage"
                start_pos = full_text.lower().find(m_str.lower())
                window = full_text[max(0, start_pos - 40): min(len(full_text), start_pos + len(m_str) + 40)]
                if re.search(r'\b(without\s+visible\s+bleeding|without\s+visible\s+hemorrhage|no\s+obvious\s+complications|prevent\s+bleeding)\b', window, re.I):
                    continue
                clean_matches.append(m_str)
            if clean_matches:
                clip_anomalies.append({
                    'category': label,
                    'keywords_matched': list(set(clean_matches))
                })
                
    if clip_anomalies:
        detected.append({
            'clip_id': clip_id,
            'parent_video_id': parent_id,
            'split': split,
            'video_path': video_path,
            'meta_file': mf,
            'sft_file': sf if os.path.exists(sf) else None,
            'anomalies': clip_anomalies,
            'visual_description': meta_data.get('visual_description', ''),
            'transcript_context': meta_data.get('transcript_context', '')
        })

output_json_path = os.path.join(base, 'cataract_anomalies_and_adverse_events.json')
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(detected, f, indent=2, ensure_ascii=False)

output_txt_path = os.path.join(base, 'cataract_anomalies_and_adverse_events.txt')
with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write(f"Total anomalous / complicated clips detected: {len(detected)}\n\n")
    for item in detected:
        f.write(f"Clip ID: {item['clip_id']} | Split: {item['split']} | Video: {item['video_path']}\n")
        anom_str = "; ".join([f"{a['category']} ({', '.join(a['keywords_matched'])})" for a in item['anomalies']])
        f.write(f"Anomalies: {anom_str}\n")
        f.write(f"Description: {item['visual_description']}\n")
        if item['transcript_context']:
            f.write(f"Transcript: {item['transcript_context']}\n")
        f.write("-" * 80 + "\n")

print(f"Analysis saved to:\n  - {output_json_path}\n  - {output_txt_path}")
print(f"Total anomalous clips: {len(detected)}")
