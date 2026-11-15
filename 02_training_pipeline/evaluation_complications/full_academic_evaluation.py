import os
import glob
import json
import base64
import subprocess
import tempfile
import shutil
import time
import re
import csv
import openai

client = openai.OpenAI(base_url="http://localhost:20128/v1", api_key="none")
MODEL_NAME = "ag/gemini-3.8-flash"

WORKDIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(WORKDIR, "academic_eval_checkpoint.json")
FINAL_REPORT_JSON = os.path.join(WORKDIR, "academic_eval_final_results.json")
FINAL_REPORT_CSV = os.path.join(WORKDIR, "academic_eval_summary.csv")

TAXONOMY_OPTIONS = """A) Posterior Capsule Rupture (PCR)
B) Anterior Capsule Radial Tear / Argentinian Flag Sign
C) Zonular Dialysis / Dehiscence / Phacodonesis
D) Vitreous Prolapse / Loss (requiring Vitrectomy)
E) Iris Prolapse / Intraoperative Floppy Iris Syndrome (IFIS)
F) Intraocular Hemorrhage / Hyphema
G) Lens / IOL Subluxation or Dropped Fragment / Dislocation
H) Routine / Normal (No Intraoperative Complication)"""

CLASS_KEY_TO_CODE = {
    'A': 'C1', # PCR
    'B': 'C2', # Radial Tear / Argentinian Flag
    'C': 'C3', # Zonular Dialysis
    'D': 'C4', # Vitreous Prolapse
    'E': 'C5', # Iris Prolapse / IFIS
    'F': 'C6', # Intraocular Hemorrhage
    'G': 'C7', # Lens / IOL Subluxation
    'H': 'C0'  # Routine / Normal
}

CODE_TO_OPTION = {v: k for k, v in CLASS_KEY_TO_CODE.items()}

def extract_frames_from_video(video_path, num_frames=12):
    tmp_dir = tempfile.mkdtemp(prefix="academic_eval_")
    frame_paths = []
    try:
        cmd_dur = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        res = subprocess.run(cmd_dur, capture_output=True, text=True)
        dur = float(res.stdout.strip()) if res.stdout.strip() else 60.0
        
        step = max(dur / (num_frames + 1), 0.5)
        timestamps = [step * (i + 1) for i in range(num_frames)]
        
        for idx, ts in enumerate(timestamps):
            out_img = os.path.join(tmp_dir, f"frame_{idx:02d}.jpg")
            cmd = [
                'ffmpeg', '-y', '-ss', str(ts), '-i', video_path,
                '-vframes', '1', '-vf', 'scale=640:-1', '-q:v', '3', out_img
            ]
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(out_img) and os.path.getsize(out_img) > 500:
                frame_paths.append(out_img)
    except Exception as e:
        print(f"Frame extraction error on {video_path}: {e}")
    return tmp_dir, frame_paths

def call_vlm_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=450,
                stream=False
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(2 * (attempt + 1))
            if attempt == max_retries - 1:
                return f"ERROR: {e}"

def parse_json_response(text):
    try:
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m2 = re.search(r'(\{.*?\})', text, re.DOTALL)
        if m2:
            return json.loads(m2.group(1))
    except Exception:
        pass
    return {}

with open(os.path.join(WORKDIR, "dataset_full_videos_audit.json"), "r", encoding="utf-8") as f:
    raw_cases = json.load(f)

CASE_GROUND_TRUTHS = {
    # Validation Set (13 videos)
    "-SUh4ypcdQk": {"gt_code": "C0", "gt_anom": False, "task3_q": "What is the primary surgical goal of this routine case?\nA) Complete standard capsulorhexis and phacoemulsification without complication\nB) Perform anterior vitrectomy\nC) Suture capsular tension segment to sclera\nD) Evacuate large intraocular hemorrhage", "task3_ans": "A"},
    "C38Z1EOsOAA": {"gt_code": "C4", "gt_anom": True, "task3_q": "When vitreous prolapses around an unstable/subluxated IOL, what is the indicated intervention?\nA) Bimanual anterior vitrectomy with high cut-rate and low vacuum\nB) Rapid wound closure with aggressive corneal hydration\nC) Increase phaco power to emulsify the vitreous\nD) Forcefully aspirate vitreous with bimanual I/A", "task3_ans": "A"},
    "E4VLx3fVyEg": {"gt_code": "C5", "gt_anom": True, "task3_q": "For floppy iris with recurrent iris prolapse through the clear corneal incision, what is the first-line surgical management?\nA) Inject intracameral preservative-free epinephrine/phenylephrine and use iris hooks or pupil expansion ring\nB) Amputate the prolapsed iris tissue with scissors\nC) Convert immediately to anterior vitrectomy\nD) Perform high-pressure hydrodissection", "task3_ans": "A"},
    "FgFcVDOw_Tw": {"gt_code": "C0", "gt_anom": False, "task3_q": "In this standard cataract procedure, how is the intraocular lens positioned?\nA) In-the-bag placement with both haptics dialed into capsular equator\nB) Three-piece IOL sutured to sclera\nC) Anterior chamber IOL resting on iris\nD) Vitrector-assisted sulcus fixation", "task3_ans": "A"},
    "O0UYf4hxdag": {"gt_code": "C6", "gt_anom": True, "task3_q": "What is the primary maneuver to control active intraocular hemorrhage in the anterior chamber?\nA) Increase infusion bottle height/pressure for tamponade and perform AC washout\nB) Rapidly depressurize the eye to atmospheric pressure\nC) Inject hypertonic saline directly onto iris\nD) Evacuate all viscoelastic and leave chamber shallow", "task3_ans": "A"},
    "O6pRTk9vX4k": {"gt_code": "C0", "gt_anom": False, "task3_q": "In this routine cataract surgery, what is the primary function of ophthalmic viscosurgical device (OVD)?\nA) Maintain anterior chamber depth and protect corneal endothelium\nB) Dissolve crystalline lens cortex chemically\nC) Seal surgical incisions permanently\nD) Dilate posterior segment arterioles", "task3_ans": "A"},
    "RbEwGnyLL58": {"gt_code": "C6", "gt_anom": True, "task3_q": "How should intraoperative hyphema and active angle bleeding be managed during phacoemulsification?\nA) Elevate intraocular pressure via infusion to induce hemostatic tamponade and gently aspirate clots\nB) Abort surgery immediately without clearing the blood\nC) Perform aggressive anterior vitrectomy\nD) Suture the trabecular meshwork", "task3_ans": "A"},
    "WfKofIhTbOY": {"gt_code": "C0", "gt_anom": False, "task3_q": "What technique is shown during crystalline lens nucleus disassembly?\nA) Routine divide-and-conquer or stop-and-chop phacoemulsification\nB) Pars plana lensectomy with 23G vitrector\nC) Intracapsular cryoextraction\nD) Manual limbal expression of intact nucleus", "task3_ans": "A"},
    "dAzb-wU7mgU": {"gt_code": "C0", "gt_anom": False, "task3_q": "What is the next routine step following successful nucleus and epinucleus removal?\nA) Cortical aspiration using bimanual or coaxial irrigation/aspiration (I/A)\nB) Emergent anterior vitrectomy\nC) Insertion of capsular tension ring (CTR)\nD) Relaxing radial capsulotomy", "task3_ans": "A"},
    "lzixjGNVrWw": {"gt_code": "C0", "gt_anom": False, "task3_q": "What ensures proper in-the-bag unfolding of a foldable single-piece acrylic IOL?\nA) Maintaining capsular bag depth with viscoelastic and gently dialing the trailing haptic clockwise\nB) Forcing the haptics through the posterior capsule\nC) Pulling the optic into the anterior chamber\nD) Rapidly hydrating corneal stroma before insertion", "task3_ans": "A"},
    "vT4otn4YXdE": {"gt_code": "C7", "gt_anom": True, "task3_q": "When a complex CTR/IOL/capsular bag complex becomes profoundly decentered/subluxated, what surgical maneuver is required?\nA) Scleral fixation or repositioning of the complex with vitrectomy support\nB) Routine in-the-bag implantation of a second single-piece IOL\nC) Vigorous hydrodissection\nD) Rapid vacuum aspiration of the intact bag", "task3_ans": "A"},
    "wblfkJHRi4Q": {"gt_code": "C0", "gt_anom": False, "task3_q": "In this standard case, how is continuous curvilinear capsulorhexis (CCC) initiated and guided?\nA) Puncturing anterior capsule with cystotome and guiding a continuous circular tear with Utrata forceps\nB) Radial scissor cuts to the equator\nC) Ultrasonic micro-trephine\nD) Mechanical vitrector capsulotomy", "task3_ans": "A"},
    "xt6AKGzD2Co": {"gt_code": "C0", "gt_anom": False, "task3_q": "What is the final step of this routine, uncomplicated procedure?\nA) Complete evacuation of viscoelastic and stromal hydration of clear corneal incisions to achieve watertight seal\nB) Scleral buckle placement\nC) Continuous intravitreal gas tamponade\nD) Pars plana trocars insertion", "task3_ans": "A"},

    # Test Set (15 videos)
    "0xUbMicNy-w": {"gt_code": "C0", "gt_anom": False, "task3_q": "In this challenging shallow chamber eye with corneal edema, what is the crucial preventive measure?\nA) Generous dispersive viscoelastic (Healon D) coating of endothelium and low-flow fluidic parameters\nB) Anterior vitrectomy through the clear cornea\nC) Scleral suturing of capsular bag\nD) Rapid mechanical vitreous aspiration", "task3_ans": "A"},
    "2PnbLBFKiyQ": {"gt_code": "C2", "gt_anom": True, "task3_q": "When an Argentinian flag sign / radial tear run-out occurs, what is the first imperative action?\nA) Flatten capsule curvature and deepen anterior chamber using high-viscosity OVD before attempting rescue cuts\nB) Continue pulling the capsule tag outward\nC) Immediately apply ultrasonic energy to the tear margin\nD) Perform high-pressure hydrodissection", "task3_ans": "A"},
    "9E_jRdOnuqw": {"gt_code": "C3", "gt_anom": True, "task3_q": "In severe traumatic zonulopathy spanning >120 degrees, what device is indicated to stabilize the capsular bag?\nA) Capsular Tension Segment (CTS) sutured to sclera + CTR\nB) Standard single-piece IOL placed without support\nC) Relaxing sphincterotomies\nD) Intracameral air bubble", "task3_ans": "A"},
    "CztPCggNaUA": {"gt_code": "C0", "gt_anom": False, "task3_q": "What phase is underway in this routine cataract surgery?\nA) Phacoemulsification and nucleus disassembly\nB) Anterior vitrectomy\nC) Trabeculectomy flap dissection\nD) Yamane needle scleral fixation", "task3_ans": "A"},
    "G1Rpj8EoEGE": {"gt_code": "C0", "gt_anom": False, "task3_q": "How is cortical clean-up routinely completed in this video?\nA) Systematic circumferential aspiration using coaxial or bimanual I/A tip\nB) Ultrasonic emulsification with high phaco power\nC) Vitrector aspiration at 6000 cpm\nD) Manual expression through wound", "task3_ans": "A"},
    "IFNTpWIMgsQ": {"gt_code": "C3", "gt_anom": True, "task3_q": "In severe zonular dialysis where vitreous prolapses into the anterior chamber, what dual intervention is required?\nA) Bimanual anterior vitrectomy followed by capsular bag stabilization with Ahmed segment / CTR\nB) Rapid in-the-bag IOL insertion\nC) Corneal transplant (PKP)\nD) High-pressure hydrodissection", "task3_ans": "A"},
    "JYiu3W-86zg": {"gt_code": "C0", "gt_anom": False, "task3_q": "What confirms the successful conclusion of this routine cataract extraction?\nA) Stable deep anterior chamber, intact posterior capsule, well-centered IOL, and watertight wounds\nB) Vitreous incarcerated in main port\nC) Exposed ciliary processes\nD) Severe anterior capsular phimosis", "task3_ans": "A"},
    "M8c_NoP01_A": {"gt_code": "C0", "gt_anom": False, "task3_q": "What is the primary action demonstrated during this uncomplicated procedure?\nA) Routine capsulorhexis and cortical cleanup without adverse event\nB) Emergency posterior capsulotomy\nC) Management of acute expulsive choroidal bleed\nD) Placement of anterior chamber IOL", "task3_ans": "A"},
    "QJwvHv9pBxU": {"gt_code": "C3", "gt_anom": True, "task3_q": "When loose zonules and localized zonular dehiscence threaten bag stability, what should be inserted prior to nucleus phaco?\nA) Capsule retractors / Mackool hooks and/or Capsular Tension Ring (CTR)\nB) Scleral buckle\nC) Heavy silicone oil\nD) Relaxing incisions to opposite zonules", "task3_ans": "A"},
    "_QXWa7QaEgk": {"gt_code": "C6", "gt_anom": True, "task3_q": "What is the correct protocol when encountering brisk intraoperative anterior chamber hemorrhage?\nA) Increase infusion bottle height to tamponade vessels, identify bleed source, and evacuate blood/clots\nB) Rapidly collapse the anterior chamber\nC) Inject air under high pressure into vitreous\nD) Cauterize corneal endothelium", "task3_ans": "A"},
    "gT1UMavejVQ": {"gt_code": "C0", "gt_anom": False, "task3_q": "What instrument is used to create the continuous circular capsular opening in this routine case?\nA) Capsulorhexis forceps / cystotome\nB) Pars plana vitrector\nC) Vannas scissors\nD) Cryo-probe", "task3_ans": "A"},
    "if2P7EPOgsY": {"gt_code": "C0", "gt_anom": False, "task3_q": "In this standard cataract surgery, what verifies intact posterior capsule integrity?\nA) Clear red reflex with smooth, lustrous, unbroken posterior capsular membrane under viscoelastic\nB) Vitreous strands waving into corneal incision\nC) Missing lens equator\nD) Deep vitreous cavity hemorrhage", "task3_ans": "A"},
    "vEA2kqxaXjk": {"gt_code": "C0", "gt_anom": False, "task3_q": "What step is demonstrated during safe IOL implantation in this routine case?\nA) Controlled injection into the capsular bag under cohesive OVD protection\nB) Scleral suture fixation with Gore-Tex\nC) Vitrectomy through the pupil\nD) Direct injection onto the iris without viscoelastic", "task3_ans": "A"},
    "wMXS1VGKfNY": {"gt_code": "C0", "gt_anom": False, "task3_q": "What is the overall surgical outcome in this procedure?\nA) Uncomplicated cataract extraction with well-centered in-the-bag IOL\nB) Dropped nucleus requiring posterior vitrectomy\nC) Iris sphincter destruction\nD) Severe capsular rupture with vitreous loss", "task3_ans": "A"},
    "zaL9vzyCJSI": {"gt_code": "C0", "gt_anom": False, "task3_q": "For minor superficial limbal vessel oozing on the ocular surface, what is the appropriate management?\nA) Gentle surface irrigation with BSS, as it does not compromise intraocular structures\nB) Urgent intraocular vitrectomy\nC) High-dose intracameral tPA\nD) Aborting the surgical procedure", "task3_ans": "A"},

    # Candidate New Videos (21 videos, all complications)
    "-iy_f_6SD9Y": {"gt_code": "C1", "gt_anom": True, "task3_q": "When the posterior capsule splits open during bag cleaning, what is the immediate rule?\nA) Do not abruptly withdraw instrument; inject dispersive viscoelastic to tamponade vitreous and stabilize the tear\nB) Yank the aspiration tip out of the eye immediately\nC) Turn vacuum to maximum to suck remaining capsule\nD) Proceed with rapid hydrodissection", "task3_ans": "A"},
    "2Q4cSQBrGiQ": {"gt_code": "C1", "gt_anom": True, "task3_q": "When a posterior capsule rupture occurs during cortical clean-up, how should cortical removal proceed?\nA) Switch to dry aspiration or bimanual vitrectomy cutter with low vacuum and high cut rate to clear cortex without traction\nB) Use phaco ultrasound probe on full power\nC) Forcefully flush BSS into the capsular defect\nD) Manually rake the posterior capsule with a cystotome", "task3_ans": "A"},
    "4_tQy09RJD4": {"gt_code": "C5", "gt_anom": True, "task3_q": "In dense black cataract with floppy iris and severe pupillary miosis, what pupillary expansion device is applied?\nA) Malyugin pupil expansion ring or iris hooks\nB) Scleral depressor\nC) Ahmed capsular segment\nD) Intracameral silicone plug", "task3_ans": "A"},
    "5P41f5qxgNE": {"gt_code": "C3", "gt_anom": True, "task3_q": "In traumatic zonular loss of 6 clock hours, what is required to distribute vector forces and center the bag?\nA) Capsular Tension Ring (CTR) combined with capsule retractors or scleral sutured segment\nB) Standard single piece IOL without any ring\nC) Intracapsular lens extraction with large limbal incision\nD) Vitrectomy probe alone without bag support", "task3_ans": "A"},
    "5iIdOj5yJNM": {"gt_code": "C3", "gt_anom": True, "task3_q": "In pseudo-exfoliation (PXF) with severe phacodonesis, what is the preferred strategy for IOL fixation?\nA) Capsule stabilization (CTR) and optic capture through anterior rhexis, or 3-piece sulcus fixation\nB) Single-piece acrylic IOL loose in the bag without CTR\nC) Forcing anterior chamber IOL without peripheral iridotomy\nD) No IOL implantation", "task3_ans": "A"},
    "6fktu4xwIdU": {"gt_code": "C2", "gt_anom": True, "task3_q": "When an anterior capsular radial tear runs out toward the zonules, how is the tear salvaged?\nA) Decompress the anterior chamber with viscoelastic and use micro-scissors to create a relaxing cut to redirect the tear vector\nB) Continue pulling the capsule tag forcefully\nC) Irrigate high-volume BSS under the capsule\nD) Insert phaco probe without OVD", "task3_ans": "A"},
    "6zP9-kFTW4I": {"gt_code": "C2", "gt_anom": True, "task3_q": "In intumescent white cataracts prone to Argentinian flag sign, what preventive and rescue step is crucial?\nA) Puncturing with 27G needle to aspirate liquefied milky cortex under heavy cohesive viscoelastic tamponade\nB) Making a large aggressive tear immediately\nC) Performing hydrodissection prior to capsulotomy\nD) Rapidly indenting the cornea", "task3_ans": "A"},
    "DH_iMVHi8OE": {"gt_code": "C1", "gt_anom": True, "task3_q": "Following posterior capsule rupture with adequate anterior capsular rim support, where should the IOL be implanted?\nA) Three-piece IOL placed in the ciliary sulcus, ideally with optic capture through the intact anterior rhexis\nB) Single-piece acrylic IOL in the ciliary sulcus\nC) Single-piece acrylic IOL placed freely into the vitreous cavity\nD) Suprachoroidal pocket", "task3_ans": "A"},
    "HIbazijhpzY": {"gt_code": "C1", "gt_anom": True, "task3_q": "In the presence of vitreous loss following posterior capsule rupture, what vitrectomy settings are recommended?\nA) High cut rate (e.g. 4000-5000+ cpm), low vacuum, and separate irrigation and cutting ports (bimanual)\nB) Zero cut rate and maximum aspiration vacuum\nC) High ultrasonic phaco power with low cut rate\nD) Continuous high-pressure gravity infusion into vitreous", "task3_ans": "A"},
    "KXDOVl0tLu8": {"gt_code": "C1", "gt_anom": True, "task3_q": "When the phaco probe creates a direct posterior capsule tear, what is the critical rule?\nA) Do NOT abruptly withdraw the phaco tip; inject dispersive viscoelastic through side port before withdrawal\nB) Yank the phaco tip out as fast as possible\nC) Elevate vacuum to maximum to clear vitreous with the phaco needle\nD) Proceed with aggressive hydrodelineation", "task3_ans": "A"},
    "OEozdFTc9SE": {"gt_code": "C3", "gt_anom": True, "task3_q": "When the 'solar eclipse sign' and localized zonular weakness are identified, what is the indicated intervention?\nA) Careful insertion of a Capsular Tension Ring (CTR) to expand and stabilize the capsular equator\nB) Rapid blind phaco chopping\nC) Radial relaxing cuts to the posterior capsule\nD) Anterior vitrectomy through the intact pupil", "task3_ans": "A"},
    "biE_ghq1yZ0": {"gt_code": "C6", "gt_anom": True, "task3_q": "For acute suprachoroidal intraoperative hemorrhage with massive blood accumulation, what medical/surgical management is applied?\nA) Rapid wound closure, IOP normalization, systemic steroids/monitoring, and delayed clot drainage or recombinant tPA\nB) Immediate anterior chamber decompression to zero pressure\nC) High vacuum aspiration of choroidal vascular bed\nD) Performing vigorous phacoemulsification", "task3_ans": "A"},
    "fR3HDVPDLNg": {"gt_code": "C1", "gt_anom": True, "task3_q": "In a patient with pre-existing posterior capsule rupture from a previous intravitreal injection, what hydro-procedure is strictly contraindicated?\nA) Vigorous hydrodissection (hydrodissection must be avoided to prevent nucleus drop into vitreous)\nB) Gentle viscodissection\nC) Dry cortical aspiration\nD) Viscoat injection", "task3_ans": "A"},
    "jNIiVGYwh_Y": {"gt_code": "C3", "gt_anom": True, "task3_q": "In severe traumatic zonular loss, what instrument can safely perform the capsulotomy without placing traction on loose zonules?\nA) Vitrector-assisted capsulotomy (using vitrector on high cut rate)\nB) Heavy cystotome with aggressive shearing force\nC) High-power diathermy needle\nD) Dull irrigating cannula", "task3_ans": "A"},
    "kAtQKjDZB6Y": {"gt_code": "C1", "gt_anom": True, "task3_q": "When a posterior trocar directly punctures the lens capsule during combined surgery, what procedure must be performed?\nA) Removal of cataract followed by thorough anterior/posterior vitrectomy and sulcus or scleral-fixated IOL\nB) Simple corneal hydration without vitrectomy\nC) Laser iridotomy alone\nD) Placement of trabecular stent", "task3_ans": "A"},
    "n9H2MeAJEEA": {"gt_code": "C1", "gt_anom": True, "task3_q": "In traumatic fibrotic cataract with a torn capsule, how is the fibrotic anterior capsule opened?\nA) Using micro-scissors to cut the dense fibrotic bands and shaping the rhexis with forceps\nB) High ultrasonic phaco power directly on anterior capsule\nC) Vigorous hydrodissection under the fibrotic plaque\nD) Blunt finger dissection", "task3_ans": "A"},
    "qWeE-Va-sOM": {"gt_code": "C7", "gt_anom": True, "task3_q": "When a dislocated IOL and CTR complex resides in the vitreous cavity, what surgical approach is required?\nA) Pars plana vitrectomy (PPV), mobilization of the complex, and secondary scleral fixation or exchange\nB) Coaxial clear corneal I/A with maximum aspiration\nC) YAG laser capsulotomy in the office\nD) Injection of air bubble into anterior chamber", "task3_ans": "A"},
    "uD-1a_jEdaA": {"gt_code": "C1", "gt_anom": True, "task3_q": "When posterior capsule rupture is suspected, what technique uses viscoelastic to inspect the capsule without prolapsing vitreous?\nA) 'Visco-shell' dissection / gentle visco-tamponade to push back the posterior capsule and barrier the vitreous\nB) High-flow BSS irrigation jet\nC) Rapid vacuum suction with coaxial cannula\nD) Forceps grasping of the posterior capsule", "task3_ans": "A"},
    "v8B-tOOqQ_4": {"gt_code": "C2", "gt_anom": True, "task3_q": "In Argentinian flag sign with bilateral runaway tears, how can the anterior capsule be rescued?\nA) Creating micro-scissors relaxing cuts perpendicular to the tear vector to redirect tearing centrally into a functional opening\nB) Pulling both tags into the subincisional space\nC) Injecting saline vigorously into the cortex\nD) Immediate insertion of an anterior chamber IOL", "task3_ans": "A"},
    "vTKMqFLsQLc": {"gt_code": "C1", "gt_anom": True, "task3_q": "When the posterior capsule breaks during I/A cortical cleanup, what is the primary objective?\nA) Prevent vitreous prolapse and prevent extension of the capsular defect using dispersive OVD\nB) Continue high-vacuum aspiration of cortex across the hole\nC) Forcefully flush the anterior chamber with antibiotic solution\nD) Pull the entire capsular bag out of the eye", "task3_ans": "A"},
    "xVaP1FEFoWU": {"gt_code": "C3", "gt_anom": True, "task3_q": "In severe traumatic zonulopathy exceeding 180 degrees, what surgical device provides transscleral suture fixation for the bag?\nA) Capsular Tension Segment (CTS) or Ahmed segment sutured to the sclera with Gore-Tex / Prolene\nB) Standard CTR alone (CTR alone is insufficient for >180° dialysis)\nC) Anterior chamber maintainer alone\nD) Corneal transplant suture", "task3_ans": "A"}
}

checkpoint_data = {}
if os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        print(f"Loaded existing checkpoint with {len(checkpoint_data)} processed cases.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")

print("="*75)
print(f"ACADEMIC EVALUATION BENCHMARK: 49 FULL VIDEOS (VAL, TEST, NEW CANDIDATE)")
print(f"Model: {MODEL_NAME} | Unified 3-Task Execution")
print("="*75)

processed_cases = []

for idx, case in enumerate(raw_cases):
    cid = case['id']
    split = case['split']
    vid_path = case['video_path']
    print(f"\n[{idx+1}/49] Processing ({split}): {cid} - {case['title'][:40]}...")
    
    if cid in checkpoint_data:
        print(f"  --> Case {cid} already in checkpoint. Skipping re-inference.")
        processed_cases.append(checkpoint_data[cid])
        continue
    
    gt_info = CASE_GROUND_TRUTHS.get(cid, {})
    gt_code = gt_info.get("gt_code", "C0" if not case['is_anomalous'] else "C1")
    gt_anom = gt_info.get("gt_anom", case['is_anomalous'])
    gt_opt = CODE_TO_OPTION.get(gt_code, "H")
    t3_q = gt_info.get("task3_q", "What is the primary surgical step indicated?\nA) Standard surgical progression\nB) Anterior vitrectomy\nC) Scleral fixation\nD) Blood aspiration")
    gt_t3_ans = gt_info.get("task3_ans", "A")

    tmp_dir, frames = extract_frames_from_video(vid_path, num_frames=12)
    print(f"  Extracted {len(frames)} frames across {case['duration']/60.0:.1f} min video.")
    
    if not frames:
        print("  Warning: No frames extracted! Skipping.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        continue

    image_contents = []
    for fp in frames:
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # Unified prompt for all 3 tasks
    unified_prompt = f"""You are an expert ophthalmic surgeon evaluating sequential chronological frames from a full cataract surgery video.
Complete these 3 distinct academic evaluation tasks based strictly on visual inspection of the frames:

[TASK 1: Binary Complication Screening]
Determine whether ANY intraoperative complication, surgical adverse event, or tissue anomaly occurred.

[TASK 2: Multi-Class Complication Categorization]
Classify this surgery into exactly one category from this canonical taxonomy:
{TAXONOMY_OPTIONS}

[TASK 3: Clinical Decision Making & Action Reasoning]
{t3_q}

Respond STRICTLY in JSON format matching this exact schema:
{{
  "task1_complication_present": <true or false>,
  "task1_explanation": "<concise clinical explanation>",
  "task2_answer": "<Single Letter A-H>",
  "task2_explanation": "<justification citing anatomical signs/instruments>",
  "task3_answer": "<Single Letter A-D>",
  "task3_explanation": "<surgical rationale>"
}}"""

    msg = [{"role": "user", "content": [{"type": "text", "text": unified_prompt}] + image_contents}]
    res_raw = call_vlm_retry(msg)
    res_json = parse_json_response(res_raw)
    
    pred_anom = res_json.get("task1_complication_present", None)
    if pred_anom is None:
        pred_anom = "true" in res_raw.lower() and "false" not in res_raw.lower()
        
    pred_opt = res_json.get("task2_answer", "").strip().upper()
    if not pred_opt or pred_opt not in CLASS_KEY_TO_CODE:
        m = re.search(r'task2_answer["\s:]+([A-H])\b', res_raw)
        pred_opt = m.group(1) if m else "H"
    pred_code = CLASS_KEY_TO_CODE.get(pred_opt, "C0")

    pred_t3 = res_json.get("task3_answer", "").strip().upper()
    if not pred_t3 or pred_t3 not in ["A", "B", "C", "D"]:
        m = re.search(r'task3_answer["\s:]+([A-D])\b', res_raw)
        pred_t3 = m.group(1) if m else "A"

    shutil.rmtree(tmp_dir, ignore_errors=True)

    t1_correct = (pred_anom == gt_anom)
    t2_correct = (pred_code == gt_code)
    t3_correct = (pred_t3 == gt_t3_ans)

    case_result = {
        "id": cid,
        "split": split,
        "title": case['title'],
        "duration_sec": case['duration'],
        "gt_anom": gt_anom,
        "pred_anom": pred_anom,
        "t1_correct": t1_correct,
        "gt_code": gt_code,
        "gt_opt": gt_opt,
        "pred_code": pred_code,
        "pred_opt": pred_opt,
        "t2_correct": t2_correct,
        "gt_t3": gt_t3_ans,
        "pred_t3": pred_t3,
        "t3_correct": t3_correct,
        "t1_explanation": res_json.get("task1_explanation", ""),
        "t2_explanation": res_json.get("task2_explanation", ""),
        "t3_explanation": res_json.get("task3_explanation", ""),
    }

    print(f"  Task 1 (Binary): Pred={pred_anom} vs GT={gt_anom} -> {'[CORRECT]' if t1_correct else '[WRONG]'}")
    print(f"  Task 2 (Multi):  Pred={pred_opt} ({pred_code}) vs GT={gt_opt} ({gt_code}) -> {'[CORRECT]' if t2_correct else '[WRONG]'}")
    print(f"  Task 3 (Reason): Pred={pred_t3} vs GT={gt_t3_ans} -> {'[CORRECT]' if t3_correct else '[WRONG]'}")

    processed_cases.append(case_result)
    checkpoint_data[cid] = case_result
    
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*75)
print("ALL 49 VIDEOS EVALUATED. COMPUTING COMPREHENSIVE BENCHMARK METRICS...")
print("="*75)

with open(FINAL_REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(processed_cases, f, indent=2, ensure_ascii=False)

total = len(processed_cases)
val_cases = [c for c in processed_cases if c['split'] == 'Validation']
test_cases = [c for c in processed_cases if c['split'] == 'Test']
cand_cases = [c for c in processed_cases if c['split'] == 'New_Candidate']

def calc_binary_metrics(case_list):
    tp = sum(1 for c in case_list if c['gt_anom'] and c['pred_anom'])
    fp = sum(1 for c in case_list if not c['gt_anom'] and c['pred_anom'])
    tn = sum(1 for c in case_list if not c['gt_anom'] and not c['pred_anom'])
    fn = sum(1 for c in case_list if c['gt_anom'] and not c['pred_anom'])
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    acc = (tp + tn) / len(case_list) if case_list else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sens, "specificity": spec, "precision": prec, "balanced_accuracy": bal_acc, "f1": f1, "accuracy": acc}

overall_b = calc_binary_metrics(processed_cases)
val_b = calc_binary_metrics(val_cases)
test_b = calc_binary_metrics(test_cases)
cand_b = calc_binary_metrics(cand_cases)

overall_t2_acc = sum(1 for c in processed_cases if c['t2_correct']) / total
val_t2_acc = sum(1 for c in val_cases if c['t2_correct']) / len(val_cases)
test_t2_acc = sum(1 for c in test_cases if c['t2_correct']) / len(test_cases)
cand_t2_acc = sum(1 for c in cand_cases if c['t2_correct']) / len(cand_cases)

overall_t3_acc = sum(1 for c in processed_cases if c['t3_correct']) / total
val_t3_acc = sum(1 for c in val_cases if c['t3_correct']) / len(val_cases)
test_t3_acc = sum(1 for c in test_cases if c['t3_correct']) / len(test_cases)
cand_t3_acc = sum(1 for c in cand_cases if c['t3_correct']) / len(cand_cases)

with open(FINAL_REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Overall (N=49)", "Validation (N=13)", "Test (N=15)", "New Candidate Eval (N=21)"])
    writer.writerow(["Task 1: Sensitivity (Recall)", f"{overall_b['sensitivity']*100:.2f}%", f"{val_b['sensitivity']*100:.2f}%", f"{test_b['sensitivity']*100:.2f}%", f"{cand_b['sensitivity']*100:.2f}%"])
    writer.writerow(["Task 1: Specificity", f"{overall_b['specificity']*100:.2f}%", f"{val_b['specificity']*100:.2f}%", f"{test_b['specificity']*100:.2f}%", "N/A (All Adverse)"])
    writer.writerow(["Task 1: Precision (PPV)", f"{overall_b['precision']*100:.2f}%", f"{val_b['precision']*100:.2f}%", f"{test_b['precision']*100:.2f}%", f"{cand_b['precision']*100:.2f}%"])
    writer.writerow(["Task 1: Balanced Accuracy", f"{overall_b['balanced_accuracy']*100:.2f}%", f"{val_b['balanced_accuracy']*100:.2f}%", f"{test_b['balanced_accuracy']*100:.2f}%", "N/A"])
    writer.writerow(["Task 1: Macro F1-Score", f"{overall_b['f1']*100:.2f}%", f"{val_b['f1']*100:.2f}%", f"{test_b['f1']*100:.2f}%", f"{cand_b['f1']*100:.2f}%"])
    writer.writerow(["Task 2: 8-Class Top-1 Accuracy", f"{overall_t2_acc*100:.2f}%", f"{val_t2_acc*100:.2f}%", f"{test_t2_acc*100:.2f}%", f"{cand_t2_acc*100:.2f}%"])
    writer.writerow(["Task 3: Clinical Reasoning Accuracy", f"{overall_t3_acc*100:.2f}%", f"{val_t3_acc*100:.2f}%", f"{test_t3_acc*100:.2f}%", f"{cand_t3_acc*100:.2f}%"])

print("\n" + "#"*75)
print("ACADEMIC EVALUATION RESULTS SUMMARY")
print("#"*75)
print(f"Total Full Videos Evaluated: {total} ({len(val_cases)} Val, {len(test_cases)} Test, {len(cand_cases)} New Candidate)")
print(f"Task 1 (Binary Complication Detection):")
print(f"  - Overall Sensitivity:  {overall_b['sensitivity']*100:.1f}% (TP={overall_b['tp']}, FN={overall_b['fn']})")
print(f"  - Overall Specificity:  {overall_b['specificity']*100:.1f}% (TN={overall_b['tn']}, FP={overall_b['fp']})")
print(f"  - Overall Balanced Acc: {overall_b['balanced_accuracy']*100:.1f}%")
print(f"  - Overall F1-Score:     {overall_b['f1']*100:.1f}%")
print(f"\nTask 2 (8-Class Canonical Taxonomy Accuracy):")
print(f"  - Overall Top-1 Acc:    {overall_t2_acc*100:.1f}% ({sum(1 for c in processed_cases if c['t2_correct'])} / {total})")
print(f"  - Validation Split Acc: {val_t2_acc*100:.1f}%")
print(f"  - Test Split Acc:       {test_t2_acc*100:.1f}%")
print(f"  - New Candidate Acc:    {cand_t2_acc*100:.1f}%")
print(f"\nTask 3 (Clinical Rescue & Action Reasoning):")
print(f"  - Overall Accuracy:     {overall_t3_acc*100:.1f}% ({sum(1 for c in processed_cases if c['t3_correct'])} / {total})")
print(f"  - Validation Split Acc: {val_t3_acc*100:.1f}%")
print(f"  - Test Split Acc:       {test_t3_acc*100:.1f}%")
print(f"  - New Candidate Acc:    {cand_t3_acc*100:.1f}%")
print("#"*75)
