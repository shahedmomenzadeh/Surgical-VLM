import os
import json
import csv

base_eval = os.path.dirname(os.path.abspath(__file__))
os.makedirs(base_eval, exist_ok=True)

# Full data dictionary for the new candidate complication videos
eval_records = [
    {
        "youtube_id": "n9H2MeAJEEA",
        "title": "CataractCoach™ 2415: traumatic cataract and fibrotic capsule",
        "url": "https://www.youtube.com/watch?v=n9H2MeAJEEA",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2024-12-16",
        "complication_category": "Traumatic Capsular Rupture & Fibrotic Capsule",
        "surgical_stage": "Pre-existing Trauma / Capsulorhexis / Lens Placement",
        "instruments": ["Trypan blue", "Capsulorhexis forceps", "Sinskey hook", "Micro-scissors", "Vitrector", "Phaco probe", "Sulcus 3-piece IOL"],
        "anatomy": ["Cornea", "Iris", "Anterior capsule", "Lens cortex", "Posterior capsule", "Ciliary sulcus"],
        "summary": "Traumatic cataract with wrinkled, fibrotic lens capsule and pre-existing capsule rupture. Trypan blue used cautiously to avoid vitreous contamination. Micro-scissors used to cut fibrotic anterior capsule. Phaco probe quickly clears dense shell; three-piece IOL placed in sulcus with optic capture.",
        "transcript": """[0:00] cataract coach.com traumatic cataract in a fibrotic capsule. What happened in this case and how are you going to proceed? Now this is a tough case. Look carefully here: you can see there's a very wrinkled and fibrotic lens capsule. You see a lot of white cataract lens material, but something looks a little off here right? So put a little trypan blue dye. Now be careful: if you put in too much trypan blue dye when there's zonular weakness or a deficiency, you may get trypan blue dye in the vitreous cavity and that could mean a hot mess; it'll ruin your red reflex at least temporarily.
Here's some viscoelastic. Here's the pre-op look: there is definitely a white cataract there, small pupil, and look at the gap between the iris and the lens. That gap is very important.
Now back to our case here: here comes the main incision, and how will you even do a rhexis in this case? You can see the surgeon grabbing in here with forceps and trying to get a rhexis done, but the capsule is so wrinkled and fibrotic it's just really not easy. So you may have to use something like sharp scissors, or looks like a Sinskey hook here trying to just separate out some of the lens material.
So clearly this patient had a rupture of the lens capsule from the injury, and you see leakage of lens material that became liquefied later and then probably absorbed. This patient probably had a lot of inflammation in the eye for a long period of time. So while the capsular bag appears mostly empty, it's because most of this lens material has been absorbed basically, liquefied and then absorbed.
Now using micro-scissors here to try to cut this material, or maybe just forceps to pull it out of the eye. This is very unusual. Try to get some sort of round opening here in the anterior lens capsule because all this lens material has to be cleared out.
Now it looks like going in with a small gauge vitrector trying to loosen up this shell basically of lens material here, and this can become very hard to remove. And there you go, look at that: it's almost like a Soemmering's ring, but this is the lens peripheral cortex that has become very opacified.
Now going to the phaco probe: the phaco probe is going to be a lot easier when you have this kind of lens material. Shoving it down a 23-gauge vitrector is just not easy. Look at that, with a phaco probe you make short work of it, very very easy and much more efficient. And there is a good posterior capsule.
Viscoelastic going inside here, taking out some of that lens cortex. You can again use the vitrector. Here's a pro-tip on the machine settings: go to cortex removal I/A mode, and then even though you're using the vitrector it won't engage the cutter.
Here comes a three-piece lens: remember the 7L rule. Open up the optic, get it turned correctly. Haptics positioned in the sulcus nicely placed. Now do optic capture if you can: optic capture puts the optic in a better position for IOL calculations and holds the optic back away from the posterior surface of the iris so you're less likely to get the optic rubbing against the iris.
Sealing up the incisions; look at that, a beautiful case and great restoration of vision."""
    },
    {
        "youtube_id": "kAtQKjDZB6Y",
        "title": "CataractCoach™ 2866: trocar placement punctures the lens",
        "url": "https://www.youtube.com/watch?v=kAtQKjDZB6Y",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2026-03-12",
        "complication_category": "Posterior Capsule Rupture (Trocar-Induced) & Vitrectomy",
        "surgical_stage": "Combined Phaco-PPV / Trocar Placement",
        "instruments": ["23G/25G Trocar cannula", "Dispersive OVD", "Phaco probe", "Pars plana vitrector", "3-piece IOL"],
        "anatomy": ["Sclera", "Pars plana", "Posterior lens capsule", "Vitreous cavity", "Macula"],
        "summary": "Inadvertent trocar puncture of posterior lens capsule during combined Phaco-PPV. Management mandates: (1) no traditional hydrodissection (hydrodelineation only), (2) viscoelastic tamponade behind lens nucleus, (3) low-flow phaco, (4) early vitrector takeover to prevent nucleus drop into macular hole.",
        "transcript": """CataractCoach 2866: Trocar placement punctures the lens.
We always aim for a perfect case, but as every seasoned surgeon knows, it’s not about the complication, rather it’s about how you recover from it.
Today we’re discussing a challenging scenario: a combined Phaco-PPV for a macular hole, and while inserting a trocar, the sharp tip inadvertently punctures the posterior lens capsule. This is a 'stop and think' moment. If you proceed with standard phacoemulsification, that small tear will quickly become a massive hole, and you’ll find the crystalline lens sitting on the macula.
Here is how to handle a Trocar-Induced Posterior Capsule Rupture successfully:
- Step 1: Don’t Panic and Don’t Hydrodissect. If you haven't started the cataract portion yet, do not perform traditional hydrodissection. The fluid pressure will propagate the tear. Instead, perform hydrodelineation to stay away from the capsule and focus on a slow-motion phaco technique.
- Step 2: Scaffolding with Viscoelastic. Before you even touch the lens with the phaco probe, fill the area of the tear with a cohesive OVD. This creates a physical barrier that keeps the vitreous back and prevents lens fragments from falling into the vitreous cavity.
- Step 3: Management of the Nucleus. Use a low-flow setting and avoid any sudden changes in anterior chamber pressure that would allow the tear to expand.
- Step 4: The Vitrectomy Advantage. Since you are already set up for a PPV, you have the ultimate safety net. If any lens material starts to migrate posteriorly, do not chase it with the phaco tip; just switch to the vitrector.
Pro-Tip: If the tear is peripheral and small, you can often still place a one-piece IOL in the capsular bag, provided you don't over-inflate it. However, if the tear is large or central, plan for a three-piece IOL in the sulcus with optic capture through the anterior capsulorhexis. This is safer particularly since the vitreous cavity will be filled with expansile gas at the end of the case. Once the anterior segment is stable and the IOL is secure, proceed with vitrectomy and membrane peeling for the macular hole."""
    },
    {
        "youtube_id": "KXDOVl0tLu8",
        "title": "CataractCoach™ 2807: phaco probe breaks the capsule",
        "url": "https://www.youtube.com/watch?v=KXDOVl0tLu8",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2026-01-12",
        "complication_category": "Posterior Capsule Rupture (Phaco Probe Induced)",
        "surgical_stage": "Nuclear Phacoemulsification / Last Fragment Removal",
        "instruments": ["Phaco handpiece", "Dispersive viscoelastic (Viscoat)", "Side-port cannula", "Bimanual anterior vitrectomy probe", "Sulcus IOL"],
        "anatomy": ["Anterior chamber", "Posterior capsule", "Anterior hyaloid face", "Pupillary plane"],
        "summary": "Direct phaco probe contact breaks the fragile posterior capsule during removal of the final quadrant. Core surgical rule: Do NOT withdraw phaco probe immediately. Maintain infusion, inject dispersive OVD through side port to tamponade vitreous, pressurize chamber, and perform bimanual anterior vitrectomy without fishing for fragments.",
        "transcript": """CataractCoach 2807: Phaco probe breaks the capsule.
When a posterior capsule rupture (PCR) occurs during phaco, the surgeon’s immediate priority is chamber stability.
First rule: do not abruptly remove the phaco probe. Maintaining infusion prevents anterior chamber collapse, which could otherwise cause massive vitreous prolapse.
Switch the machine to a 'Cortex' or 'Visco' setting and inject a dispersive viscoelastic through a side port to support the remaining lens fragments and tamponade the vitreous face.
Once the eye is pressurized and stabilized with viscoelastic, the phaco probe can be safely withdrawn.
If vitreous has entered the anterior chamber, a bimanual anterior vitrectomy is necessary to clear the incision sites and pupillary plane. Avoid 'fishing' for fragments in the vitreous cavity.
Once the vitreous is managed and cleared, the remaining cortex should be removed using dry aspiration.
Depending on the remaining anterior and posterior capsular support, determine if an intraocular lens can be placed in the sulcus with optic capture or if an alternative scleral fixation method is required."""
    },
    {
        "youtube_id": "2Q4cSQBrGiQ",
        "title": "CataractCoach™ 2770: first ever posterior capsule rupture",
        "url": "https://www.youtube.com/watch?v=2Q4cSQBrGiQ",
        "channel": "CataractCoach (Resident Surgeon Case)",
        "date": "2025-12-06",
        "complication_category": "Posterior Capsule Rupture (Cortical Clean-up)",
        "surgical_stage": "Irrigation / Aspiration of Cortex",
        "instruments": ["I/A handpiece", "Triamcinolone acetonide (Kenalog/Triesence)", "Anterior vitrector", "Viscoelastic", "3-piece IOL"],
        "anatomy": ["Posterior capsule", "Cortex", "Vitreous strands", "Capsulorhexis margin"],
        "summary": "Novice resident encounters their first posterior capsule rupture during cortical cleanup. Demonstrates step-by-step psychological composure, avoiding abrupt instrument removal, triamcinolone acetonide staining to visualize transparent vitreous strands, automated anterior vitrectomy, and sulcus IOL placement with optic capture.",
        "transcript": """CataractCoach 2770: First ever posterior capsule rupture.
This complication happens to 100% of cataract surgeons: the posterior capsule ruptures during phaco or cortical cleanup and now you must complete the case.
Managing your first cataract case complicated by a posterior capsule rupture (PCR) is stressful, but it is an essential learning milestone.
The most important principle is to slow down. Pause, stabilize the situation, and reassess. Avoid making sudden instrument movements, which can worsen the rupture or allow vitreous prolapse.
Maintain anterior chamber stability with gentle infusion and ensure you are not over-pressurizing or collapsing the eye.
Keep your mental checklist simple:
1. Identify whether vitreous is present in the anterior chamber using diluted triamcinolone acetonide staining.
2. Maintain a closed system and perform bimanual anterior vitrectomy to avoid traction on any vitreous strands.
3. Decide on the safest IOL option: whether sulcus placement of a 3-piece IOL with optic capture is appropriate or if leaving the eye aphakic is safer.
Most importantly, remember that complications happen to every surgeon. Your composure and methodical decision-making matter more than panic."""
    },
    {
        "youtube_id": "fR3HDVPDLNg",
        "title": "CataractCoach™ 2776: pre-existing posterior capsule rupture",
        "url": "https://www.youtube.com/watch?v=fR3HDVPDLNg",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2025-12-12",
        "complication_category": "Pre-existing Posterior Capsule Rupture (Post-Intravitreal Injection)",
        "surgical_stage": "Full Case / Post-Vitrectomy Eye",
        "instruments": ["B-scan ultrasound", "Dispersive OVD", "Low-fluidic phaco", "Anterior vitrector", "Sulcus IOL"],
        "anatomy": ["Posterior capsule defect", "Vitreous cavity", "Anterior hyaloid face (absent)", "Zonules"],
        "summary": "Cataract surgery in an eye with a pre-existing posterior capsule rupture secondary to a prior intravitreal needle injection. The eye has undergone previous vitrectomy, meaning no anterior hyaloid face exists and lens fragments can fall straight to the retina. Managed with minimal hydrodissection, low fluidics, and sulcus IOL fixation.",
        "transcript": """CataractCoach 2776: Pre-existing posterior capsule rupture from prior intravitreal injection.
Cataract surgery in an eye with a posterior capsule rupture from a prior intravitreal injection requires careful preoperative planning and intraoperative adaptability.
These cases present with compromised capsular support, vitreous prolapse, and altered lens stability, increasing risks of dropped nuclear material.
Crucially, this patient also had a prior pars plana vitrectomy, so there is NO anterior hyaloid face. Any displaced lens material will quickly fall back into the vitreous cavity without resistance.
A thorough slit-lamp assessment and ultrasound biomicroscopy or B-scan ultrasound help define the defect ahead of time.
Intraoperatively, priorities are:
- Maintaining anterior chamber stability.
- Minimizing lens manipulation and avoiding aggressive hydrodissection.
- Anticipating early vitreous involvement.
- Anterior vitrectomy to clear any prolapsed strands.
- Placing a 3-piece IOL in the ciliary sulcus with optic capture across the intact anterior capsulorhexis."""
    },
    {
        "youtube_id": "-iy_f_6SD9Y",
        "title": "CataractCoach™ 2751: cleaning the bag and then it splits open",
        "url": "https://www.youtube.com/watch?v=-iy_f_6SD9Y",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2025-11-17",
        "complication_category": "Posterior Capsule Rupture (Capsular Bag Polishing)",
        "surgical_stage": "Cortical Aspiration & Posterior Capsule Polishing",
        "instruments": ["I/A probe", "Dispersive OVD", "Triamcinolone", "Toric IOL / Sulcus 3-piece IOL"],
        "anatomy": ["Posterior capsule", "Capsular bag equator", "Anterior hyaloid face", "Corneal astigmatism"],
        "summary": "Posterior capsule tears during polishing/cleaning of the capsular bag after complete nucleus removal. Immediate cessation of aspiration and injection of dispersive OVD prevents anterior hyaloid face breach. Addresses the dilemma of toric IOL placement when the posterior capsule is compromised.",
        "transcript": """CataractCoach 2751: Cleaning the bag and then it splits open.
A highly experienced surgeon is more likely to experience a rupture of the posterior capsule during cortex removal or bag polishing rather than with nucleus phacoemulsification.
When a posterior capsule rupture occurs during capsular bag cleaning, the priority is to maintain anterior chamber stability and prevent vitreous prolapse.
Immediately stop irrigation/aspiration, inject a dispersive ophthalmic viscosurgical device (OVD) to tamponade the rent, and withdraw instruments carefully. Avoid chamber collapse at all costs.
Stain with triamcinolone if vitreous loss is suspected. If there is prolapsed vitreous, perform a meticulous anterior vitrectomy.
In this case, we were able to keep the anterior hyaloid face intact and avoid vitreous prolapse.
Now the critical question: where and how to implant the toric IOL needed to treat the patient’s corneal astigmatism when the posterior capsule is compromised?"""
    },
    {
        "youtube_id": "6zP9-kFTW4I",
        "title": "CataractCoach™ 3017: Argentinian flag in Expert Hands",
        "url": "https://www.youtube.com/watch?v=6zP9-kFTW4I",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2026-08-15",
        "complication_category": "Argentinian Flag Sign / Intumescent Capsular Tear",
        "surgical_stage": "Capsulorhexis / Capsulotomy Initiation",
        "instruments": ["Trypan blue", "Heavy cohesive OVD (Healon GV)", "27G decompression needle", "Micro-scissors", "Phaco chopper"],
        "anatomy": ["Intumescent white lens cortex", "Anterior capsule", "Zonules", "Peripheral capsular bag"],
        "summary": "White intumescent cataract with elevated intralenticular pressure leading to sudden radial extension of the anterior capsular tear (Argentinian Flag Sign). Managed with high-density OVD chamber over-pressurization, micro-scissors release of tension vectors, and converting to a stable modified capsulotomy.",
        "transcript": """CataractCoach 3017: Argentinian Flag Sign in Expert Hands.
In intumescent white cataracts, the intralenticular pressure is markedly elevated due to liquefied cortical material trapped within the capsular bag.
Upon puncturing the Trypan blue-stained anterior capsule, the vector forces can cause an instantaneous, uncontrolled radial tear that runs out 180 degrees to the equator, creating the classic blue-white-blue striping known as the Argentinian Flag sign.
Expert management demands immediate recognition:
1. Do not allow the anterior chamber to shallow; immediately inject high-density cohesive viscoelastic (such as Healon GV) to flatten the anterior dome of the lens and counter the intralenticular pressure.
2. Decompress the liquid cortex from the bag.
3. Use micro-scissors to cut each radial flap and re-direct the tearing vector back into a continuous curvilinear pattern or convert to a stable can-opener capsulotomy.
4. Proceed with low-vacuum, low-power phaco to prevent posterior extension of the tear."""
    },
    {
        "youtube_id": "6fktu4xwIdU",
        "title": "CataractCoach™ 2931: young registrar meets Argentinian flag!",
        "url": "https://www.youtube.com/watch?v=6fktu4xwIdU",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2026-05-10",
        "complication_category": "Argentinian Flag Sign / Radial Tear Run-out",
        "surgical_stage": "Capsulorhexis",
        "instruments": ["Cystotome", "Capsulorhexis forceps", "Viscoelastic", "Vannas scissors"],
        "anatomy": ["Anterior capsule", "Cortex", "Equator of capsular bag"],
        "summary": "Young surgical trainee encounters an unexpected Argentinian flag radial tear run-out while initiating capsulorhexis. The supervising consultant steps in to demonstrate stabilizing the chamber, relieving capsular tension, and rescuing the capsular opening with micro-scissors.",
        "transcript": """CataractCoach 2931: Young registrar meets Argentinian flag!
A teaching video showing a surgical resident / registrar operating on a white cataract.
As the cystotome enters the stained capsule, the sudden release of liquefied milky cortex causes the capsule flap to rapidly tear outward toward the zonules at both opposite poles.
Key teaching points for trainees:
- Never apply downward pressure when puncturing an intumescent lens.
- Pre-aspirate the liquefied cortex with a 27G needle on an empty syringe before attempting the rhexis.
- If the tear runs out, stop immediately. Do not pull on the flap.
- Refill the chamber with viscoelastic, use micro-scissors to make an intentional relaxing cut, and steer the tear back into a continuous curve."""
    },
    {
        "youtube_id": "v8B-tOOqQ_4",
        "title": "CataractCoach™ 2894: How do you save the capsule now? Argentinian Flag Sign",
        "url": "https://www.youtube.com/watch?v=v8B-tOOqQ_4",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2026-04-12",
        "complication_category": "Argentinian Flag Sign / Rescue Capsulotomy",
        "surgical_stage": "Capsulotomy Conversion",
        "instruments": ["Micro-scissors", "Cohesive OVD", "Phaco probe", "I/A probe"],
        "anatomy": ["Anterior capsule flaps", "Intumescent cortex", "Posterior capsule"],
        "summary": "Detailed video focusing on surgical mechanics of rescuing an eye after the Argentinian flag sign has already occurred. Step-by-step demonstration of using 23-gauge micro-scissors to cut each capsule leaf, creating a wide central aperture while maintaining equatorial integrity.",
        "transcript": """CataractCoach 2894: How do you save the capsule now? Argentinian Flag Sign.
Once the Argentinian flag sign has occurred, standard continuous curvilinear capsulorhexis is no longer possible.
How do you proceed with phacoemulsification without causing the radial tears to wrap around the equator and tear the posterior capsule?
1. Chamber stability: inject viscoelastic over the tears to hold the capsule flat against the lens cortex.
2. Mobilize the anterior flaps using micro-forceps.
3. Introduce intraocular micro-scissors through a paracentesis and make intentional cuts perpendicular to the radial tears.
4. Carefully aspirate the liquefied and loose cortex using gentle low-flow irrigation/aspiration before engaging the dense nucleus.
5. Use vertical phaco chop with minimal lateral vector forces to fracture the nucleus in the center, keeping all mechanical energy directed away from the weakened capsule edges."""
    },
    {
        "youtube_id": "xVaP1FEFoWU",
        "title": "CataractCoach™ 3010: traumatic zonulopathy >180 degrees",
        "url": "https://www.youtube.com/watch?v=xVaP1FEFoWU",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2026-08-03",
        "complication_category": "Severe Zonular Dialysis (>180°) & Scleral CTS Fixation",
        "surgical_stage": "Capsular Bag Stabilization & Phacoemulsification",
        "instruments": ["Capsule hooks (Mackool)", "Capsular Tension Segment (CTS)", "Capsular Tension Ring (CTR)", "Gore-Tex / Prolene suture", "Low-temperature cautery"],
        "anatomy": ["Ciliary zonules (compromised >180°)", "Capsular equator", "Sclera", "Ciliary sulcus"],
        "summary": "Severe post-traumatic cataract with over 180 degrees of zonular loss. Attempting standard phacoemulsification without specialized stabilization risks total dislocation of the crystalline lens into the vitreous cavity. Systematic strategy uses early capsular hooks, a scleral-fixated CTS, and in-the-bag CTR.",
        "transcript": """CataractCoach 3010: How will you fix this case of traumatic zonulopathy?
Severe ocular trauma with more than 180 degrees of zonular loss presents one of the most difficult challenges in anterior segment surgery.
When capsular support is so extensively compromised, attempting standard phacoemulsification without specialized stabilization risks total dislocation of the crystalline lens into the vitreous cavity.
A systematic strategy for managing massive zonular weakness involves:
1. Utilizing early capsular hooks to suspend the bag and re-establish centration and counter-traction before safely proceeding with nucleus removal.
2. Placing a Capsular Tension Segment (CTS) anchored with a trans-scleral suture to permanently secure the sector of missing zonules.
3. Placing a Capsular Tension Ring (CTR) to expand the equatorial bag.
4. Performing slow-motion, gentle phacoemulsification.
This restores long-term capsular centration and allows in-the-bag IOL placement without lens tilt or dislocation."""
    },
    {
        "youtube_id": "5iIdOj5yJNM",
        "title": "CataractCoach™ 2983: lot of phacodonesis in pseudo-exfoliation",
        "url": "https://www.youtube.com/watch?v=5iIdOj5yJNM",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2026-07-07",
        "complication_category": "Zonulopathy / Phacodonesis / Optic Capture",
        "surgical_stage": "Pseudo-exfoliation (PXF) Full Procedure",
        "instruments": ["Capsule hooks", "Capsular Tension Ring (CTR)", "Sinskey hook", "3-piece IOL"],
        "anatomy": ["Zonular apparatus (PXF laxity)", "Capsular bag equator", "Ciliary sulcus", "Pupillary margin"],
        "summary": "94-year-old patient with pseudoexfoliation syndrome and marked phacodonesis. Demonstrates proactive incremental 4-quadrant capsulorhexis with sequential capsule hook placement, slow-motion phaco, delayed CTR insertion after cortex removal, and definitive stabilization via 3-piece IOL sulcus placement with optic capture.",
        "transcript": """CataractCoach 2983: Lot of phacodonesis in pseudo-exfoliation cataract surgery.
This 94-year-old patient presents with marked phacodonesis in pseudoexfoliation (PXF) syndrome. In cases like this, waiting until a zonular dialysis occurs during phaco to react can lead to disaster.
The goal is early stabilization of the capsular bag before any major mechanical stress is applied.
Key surgical technique:
1. Sequential hook placement: perform capsulorhexis 90 degrees at a time, immediately placing a capsule hook under each completed quadrant to provide counter-traction.
2. Low-fluidics slow-motion phaco chop to protect remaining fragile zonules.
3. Cortical clean-up performed parallel to the hook support.
4. Save CTR placement until AFTER cortical clean-up so cortical fibers aren't trapped against the bag equator.
5. Optic capture: inject a 3-piece IOL with haptics in the ciliary sulcus and buttonhole the optic posteriorly through the intact capsulorhexis into the capsular bag. This creates an absolute mechanical lock, eliminating pseudo-phacodonesis and IOL wobble."""
    },
    {
        "youtube_id": "5P41f5qxgNE",
        "title": "CataractCoach™ 2975: traumatic cataract with zonular loss",
        "url": "https://www.youtube.com/watch?v=5P41f5qxgNE",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2026-06-29",
        "complication_category": "Traumatic Zonular Dialysis (6 Clock Hours)",
        "surgical_stage": "Femto-phaco & CTR Insertion",
        "instruments": ["Femtosecond laser", "Capsular Tension Ring (CTR)", "Sinskey hook", "Cohesive OVD"],
        "anatomy": ["Lens equator", "Broken zonules (6 clock hours)", "Vitreous face", "Posterior chamber"],
        "summary": "Severe traumatic cataract with 6 clock hours of zonular loss where the lens equator is visibly exposed. Discusses limits of standard CTR placement (typically ≤4 clock hours) versus necessity of scleral sutured capsular segments (CTS/Cionni) for extensive dialysis.",
        "transcript": """CataractCoach 2975: Traumatic cataract with zonular loss.
In this challenging surgical case, a patient presents with a severe traumatic cataract featuring six clock hours of extensive zonular loss.
The video captures the critical moment where the lens equator is clearly visible, requiring special techniques to stabilize the lens before removal.
Our guest surgeons use a femtosecond laser to create the anterior capsulotomy and soften and divide the lens nucleus.
In my opinion, the maximum degree of zonular loss that can safely be addressed with solely a capsular tension ring (CTR) is about 4 clock hours (120 degrees). Beyond that, using a suture to secure the capsular bag with a capsular tension segment or Cionni ring is strongly advised.
However, our guest surgeons elect to use just the CTR for 6 clock hours of zonular loss. It appears relatively stable, but prompts the question: is using just a CTR enough for long-term centration?"""
    },
    {
        "youtube_id": "biE_ghq1yZ0",
        "title": "CataractCoach™ 2711: tPA for supra-choroidal hemorrhage",
        "url": "https://www.youtube.com/watch?v=biE_ghq1yZ0",
        "channel": "Uday Devgan / Ramesh Ayyala MD",
        "date": "2025-10-08",
        "complication_category": "Suprachoroidal Hemorrhage & Intraocular Bleeding Rescue",
        "surgical_stage": "Complication Management / Scleral Drainage",
        "instruments": ["Recombinant tPA (tissue plasminogen activator)", "Sclerotomy blade", "Infusion cannula", "Syringe"],
        "anatomy": ["Suprachoroidal space", "Choroid", "Sclera", "Retina", "Anterior chamber"],
        "summary": "Suprachoroidal hemorrhage: catastrophic intraoperative / early postoperative bleeding complication into the suprachoroidal space. Demonstrates innovative enzymatic clot dissolution with recombinant tissue plasminogen activator (tPA) for accelerated, complete suprachoroidal drainage and IOP control.",
        "transcript": """CataractCoach 2711: tPA for suprachoroidal hemorrhage.
Suprachoroidal hemorrhage, often a devastating complication of intraocular surgery or trauma, results in rapid vision loss and severe intraocular pressure spike due to accumulation of blood in the suprachoroidal space.
Traditionally, delayed surgical drainage is performed only after spontaneous clot liquefaction, typically about 10 to 14 days post-hemorrhage.
Tissue plasminogen activator (tPA) has emerged as an adjunctive treatment to facilitate rapid clot liquefaction and safe surgical drainage.
Suprachoroidal injection of tPA enzymatically breaks down fibrin, enabling earlier, smoother, and more complete evacuation of the dense hemorrhage through radial posterior sclerotomies.
Using tPA as shown in this case reduces intraocular pressure spikes, restores retinal architecture, and helps preserve visual function in an otherwise catastrophic scenario."""
    },
    {
        "youtube_id": "4_tQy09RJD4",
        "title": "Managing complex cataract surgery: Advanced strategies (Malyugin)",
        "url": "https://www.youtube.com/watch?v=4_tQy09RJD4",
        "channel": "Prof. Boris Malyugin (UCLA Stein Eye Institute)",
        "date": "2024-10-15",
        "complication_category": "Complex Hard Cataract / Malyugin Ring / High PCR Risk",
        "surgical_stage": "Full Complex Cataract Procedure",
        "instruments": ["Malyugin ring", "Phaco chop handpiece", "Femtosecond laser", "Viscoelastic"],
        "anatomy": ["Iris sphincter", "Hard brunescent nucleus", "Posterior capsule", "Corneal endothelium"],
        "summary": "Comprehensive masterclass by Prof. Boris Malyugin on managing advanced cataract surgery risks. Covers small non-dilating pupils using the Malyugin Ring, protecting corneal endothelium, fluidic strategies for brunescent rock-hard cataracts, and techniques to prevent posterior capsule rupture and zonular dehiscence.",
        "transcript": """[0:04] Hi, my name is Boris Malyugin. I am a professor of ophthalmology at the Stein Eye Institute at UCLA. Today I will review recent advancements in complicated cataract surgeries.
Ultrasound breaks the cataract into pieces which are suctioned out. The lens capsule is cleaned and the IOL is inserted with a special tool.
The main challenge in complex cataract surgery is a hard nucleus that increases the risk of severe complications like capsule rupture or excessive stress on the lens supporting zonular structure.
The solutions include:
- Optimizing fluidics and ultrasound energy settings to minimize endothelial and incisional thermal wound burn.
- Using vertical phaco chop and stop-and-chop to mechanically divide the lens rather than sculpting with high ultrasound power.
- Employing pupil expansion devices such as the Malyugin Ring to maintain a safe 6.25mm or 7.0mm pupil aperture without sphincter tears.
- Being vigilant during fragment removal to prevent post-occlusion surge and posterior capsule rupture."""
    },
    {
        "youtube_id": "uD-1a_jEdaA",
        "title": "CataractCoach™ 2139: is the posterior capsule ruptured?",
        "url": "https://www.youtube.com/watch?v=uD-1a_jEdaA",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2024-03-20",
        "complication_category": "Suspected Posterior Capsule Rupture & Viscoshell Technique",
        "surgical_stage": "Epinucleus / Cortex Removal",
        "instruments": ["Phaco handpiece", "Dispersive OVD", "Bimanual I/A handpiece", "Trifocal IOL"],
        "anatomy": ["Epinuclear shell", "Posterior capsule", "Anterior hyaloid face"],
        "summary": "During epinucleus aspiration, a suspicious round defect appears behind the lens shell resembling the phaco tip outline. Surgeon avoids withdrawing probe, injects viscoelastic barrier beneath the fragment (Viscoshell maneuver), viscodissects cortex, confirms capsule integrity, and implants trifocal IOL.",
        "transcript": """CataractCoach 2139: Is the posterior capsule ruptured?
You notice this sign during phaco: that defect looks just like the phaco tip.
This is the part where your heart is beating fast: what is going on here and what do I do?
Rule 1: Don't come out of the eye! Stay in the eye with infusion on. If you want to come out, don't let the eye collapse—inject viscoelastic underneath the fragment first.
Viscoelastic is cheaper than vitreous! Inject viscoelastic under the piece, bring the piece up into the anterior chamber, and aspirate it gently.
Now look at that round defect: is it a hole in the capsule or a hole in the cortex?
You check with viscodissection: inject cohesive OVD behind the cortex to dissect it off the posterior capsule.
And look: no hole in the capsule! The phaco tip went through the epinuclear shell and stopped just microns shy of the posterior capsule.
Completed cortex cleanup using bimanual I/A and successfully placed a multifocal IOL in the bag."""
    },
    {
        "youtube_id": "qWeE-Va-sOM",
        "title": "CataractCoach™ 2605: combination pars plana vitrectomy and IOL exchange",
        "url": "https://www.youtube.com/watch?v=qWeE-Va-sOM",
        "channel": "Uday Devgan & Pradeep Prasad MD",
        "date": "2025-06-18",
        "complication_category": "Dislocated IOL & CTR / Pars Plana Vitrectomy / Yamane Fixation",
        "surgical_stage": "IOL Explantation & Secondary Fixation",
        "instruments": ["25G/27G Trocar system", "Pars plana vitrector", "Endolaser", "TSK thin-wall 30G needle", "Yamane double-needle", "Low-temperature cautery"],
        "anatomy": ["Vitreous cavity", "Dislocated capsular bag", "Retina", "Sclera"],
        "summary": "Late in-the-bag IOL and CTR dislocation into deep vitreous cavity secondary to progressive zonulopathy. Full pars plana vitrectomy performed to free vitreous traction, retrieve dislocated CTR and IOL, followed by scleral-fixated secondary IOL using the Yamane flanged intrascleral haptic fixation technique.",
        "transcript": """CataractCoach 2605: Combination pars plana vitrectomy and IOL exchange with a hidden CTR.
Dr. Pradeep Prasad and Dr. Uday Devgan discuss managing a completely dislocated IOL-capsular bag complex.
First, perform a full pars plana vitrectomy to completely clear vitreous around the dislocated lens and prevent vitreous traction on the peripheral retina.
Shave the vitreous base meticulously to decrease risk of retinal tear or detachment.
The dislocated CTR is grasped in the vitreous cavity and safely explanted through the anterior chamber.
Next, perform the Yamane double-needle technique:
- Introduce angled thin-wall 30-gauge needles 2mm posterior to the limbus through the sclera.
- Thread the leading and trailing PMMA haptics of a three-piece IOL into the needle lumens.
- Externalize the haptics, create terminal bulbs/flanges using low-temperature cautery, and push the flanges into the scleral tunnels to achieve perfect optic centration."""
    },
    {
        "youtube_id": "jNIiVGYwh_Y",
        "title": "CataractCoach™ 2719: traumatic cataract with zonular loss - this is tough!",
        "url": "https://www.youtube.com/watch?v=jNIiVGYwh_Y",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2025-10-20",
        "complication_category": "Traumatic Zonular Dialysis (6 Clock Hours)",
        "surgical_stage": "Cataract Extraction in Compromised Bag",
        "instruments": ["Vitrector", "Capsular Tension Ring (CTR)", "Sinskey hook", "Micro-forceps", "3-piece IOL"],
        "anatomy": ["Capsular bag", "6 clock-hours zonular loss", "Equatorial bag", "Vitreous"],
        "summary": "Severe post-traumatic zonular deficiency with lens displacement and scarred fibrotic capsule. The vitrector is used to create the anterior capsulotomy and aspirate the soft lens. Discussion highlights why a standard CTR alone is insufficient when zonular loss exceeds 4 clock hours.",
        "transcript": """CataractCoach 2719: Traumatic cataract with zonular loss.
This is a very tough case: prior ocular trauma resulted in severe zonular loss (at least 6 clock hours, 180 degrees) with the lens nucleus displaced.
First, check for vitreous prolapse. Use a vitrector cutter to create an anterior capsulotomy through the fibrotic capsule.
The soft lens material is aspirated directly using the vitrector in cortex removal mode.
Then the surgeon attempts to place a standard capsular tension ring (CTR) to support the bag.
Critical teaching pearl: A standard CTR only adequately redistributes forces for up to 3-4 clock hours of zonular deficiency. When zonular loss is 6 clock hours, placing a CTR alone is often futile and risks capsular bag rupture or late in-the-bag dislocation. Scleral-sutured segments (CTS) or converting to scleral fixation (Yamane technique) are needed."""
    },
    {
        "youtube_id": "5uioc-PYfWg",
        "title": "CataractCoach™ 2018: dislocated IOL replaced with an ACIOL",
        "url": "https://www.youtube.com/watch?v=5uioc-PYfWg",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2024-01-15",
        "complication_category": "Dislocated IOL / Severe Capsular Phimosis / ACIOL Placement",
        "surgical_stage": "IOL Explantation & Anterior Chamber IOL",
        "instruments": ["Forceps", "Calipers (white-to-white)", "Scleral tunnel blade", "Vitrector", "ACIOL", "Peripheral iridotomy punch"],
        "anatomy": ["Capsular phimosis", "Zonules", "Anterior chamber angle", "Iris", "Scleral tunnel"],
        "summary": "Severe capsular bag phimosis causing progressive 360° zonular failure and complete IOL dislocation. The mobile IOL-capsular complex is explanted through a scleral tunnel, anterior vitrectomy is performed, and a correctly sized Anterior Chamber IOL (ACIOL) with peripheral iridotomy is implanted.",
        "transcript": """CataractCoach 2018: Dislocated IOL replaced with an ACIOL.
Look at that severe capsular phimosis: progressive contracture of the anterior capsulorhexis pulls on the zonular attachments and erodes them over time.
There is virtually zero zonular support remaining; the entire IOL and capsular bag complex is floating and dislocated.
Surgical technique:
- Bring the entire capsular bag and IOL complex up into the anterior chamber above the iris plane so it cannot fall into the vitreous cavity.
- Create a 6.5mm scleral tunnel incision (superior to corneal incisions for large implants to prevent severe astigmatism).
- Explant the IOL and fibrotic bag.
- Measure corneal white-to-white diameter carefully for ACIOL footplate sizing.
- Perform an anterior vitrectomy with triamcinolone visualization.
- Insert the ACIOL in the correct vault orientation (feet in the scleral angle).
- Perform a surgical peripheral iridotomy (PI) to prevent pupillary block glaucoma."""
    },
    {
        "youtube_id": "wgP_QR0CsJY",
        "title": "CataractCoach™ 2332: anterior vitrectomy technique",
        "url": "https://www.youtube.com/watch?v=wgP_QR0CsJY",
        "channel": "Uday Devgan / CataractCoach",
        "date": "2024-09-10",
        "complication_category": "Anterior Vitrectomy Mechanics & Technique",
        "surgical_stage": "Vitreous Removal / Chamber Decompression",
        "instruments": ["27G One-step vitrector", "AC maintainer", "Calipers", "Triamcinolone", "BSS infusion"],
        "anatomy": ["Limbal entry", "Pars plana (3.5mm posterior)", "Anterior hyaloid", "Vitreous base"],
        "summary": "Masterclass on anterior vitrectomy instrumentation, fluidics, and safety. Covers 27-gauge vitrector entry (pars plana vs anterior chamber), high cut-rate (5,000-10,000 cpm) with low vacuum (40-80 mmHg) to minimize retinal traction, and self-sealing wound management.",
        "transcript": """CataractCoach 2332: Anterior vitrectomy technique.
Whether for vitreous prolapse during cataract surgery or floaters, the fundamentals of anterior vitrectomy are critical:
1. Closed system: Use an anterior chamber maintainer to keep constant IOP and prevent chamber collapse.
2. Port placement: Measure 3.5 mm posterior to the limbus for a pars plana approach, or enter through a separate paracentesis. Never vitrectomize through the main phaco wound, as this drags vitreous strands into the incision.
3. Machine parameters: Use high cut rate (5,000 to 10,000 cuts per minute) and low vacuum (40 to 80 mmHg). High cut rates ensure small vitreous bites, while low vacuum prevents pulling and traction on the peripheral retina.
4. Keep the cutter tip still: do not swirl or sweep the instrument; let vitreous come to the cutter.
5. Triamcinolone staining: always visualize vitreous strands to confirm complete clearance before closing wounds."""
    },
    {
        "youtube_id": "DH_iMVHi8OE",
        "title": "Ruptured posterior capsule during cataract surgery",
        "url": "https://www.youtube.com/watch?v=DH_iMVHi8OE",
        "channel": "Ophthalmic Surgery Education",
        "date": "2024-08-01",
        "complication_category": "Posterior Capsule Rupture & Sulcus Conversion",
        "surgical_stage": "Nuclear Phacoemulsification / IOL Implantation",
        "instruments": ["Phaco handpiece", "Dispersive OVD", "3-piece IOL", "Anterior vitrector"],
        "anatomy": ["Posterior capsule", "Ciliary sulcus", "Anterior capsulorhexis"],
        "summary": "Classic intraoperative posterior capsule rupture managed by immediate chamber inflation with dispersive OVD, bimanual anterior vitrectomy to clear the visual axis and paracenteses, followed by sulcus implantation of a three-piece IOL with optic capture.",
        "transcript": """Ruptured posterior capsule during cataract surgery.
This patient had uneventful surgery in the first eye, but in this second eye, a tear appears in the central posterior capsule during nuclear fragment removal.
Immediate response:
- The surgeon does not remove the phaco tip until viscoelastic is injected via the side-port to prevent vitreous from surging forward into the wound.
- A bimanual anterior vitrectomy is performed with infusion via one side port and the vitrector in the other.
- The anterior capsulorhexis rim is verified to be continuous and intact for 360 degrees.
- A three-piece foldable IOL is loaded and injected into the ciliary sulcus.
- Using a Kuglen hook, the optic is captured behind the anterior capsulorhexis rim, providing excellent centration and stability."""
    },
    {
        "youtube_id": "HIbazijhpzY",
        "title": "How to manage a posterior capsule rupture in cataract surgery",
        "url": "https://www.youtube.com/watch?v=HIbazijhpzY",
        "channel": "Eye Surgery Specialists",
        "date": "2024-06-15",
        "complication_category": "Posterior Capsule Rupture / Step-by-Step Management",
        "surgical_stage": "Complete Complication Protocol",
        "instruments": ["Dispersive OVD", "Anterior vitrector", "Triamcinolone", "Miotics (Acetylcholine/Carbachol)"],
        "anatomy": ["Anterior chamber", "Posterior capsule", "Pupil", "Wounds"],
        "summary": "Comprehensive instructional guide on handling a PCR. Outlines the 5 critical tenets: No sudden withdrawal, visco-tamponade, triamcinolone visualization, anterior vitrectomy without vitreous traction, and pupil constriction checks for peaked pupil / vitreous incarceration.",
        "transcript": """How to manage a posterior capsule rupture in cataract surgery.
If you perform cataract surgery, you must be prepared to manage a posterior capsule rupture.
Follow these 5 core rules:
Rule 1: Don't withdraw instruments rapidly—this causes anterior chamber collapse and pulls the vitreous body forward.
Rule 2: Inject dispersive OVD behind the nuclear fragments and over the capsular defect.
Rule 3: Use triamcinolone acetonide to stain vitreous strands in the anterior chamber.
Rule 4: Perform bimanual anterior vitrectomy with separate irrigation and cutting ports. Maintain cutting speed at maximum.
Rule 5: Check all incision wounds at the end of surgery with a Weck-cel sponge and inject intraocular miotics (carbachol or acetylcholine). If the pupil is round and does not peak, there is no vitreous incarceration in the wounds."""
    }
]

# Write individual video files
master_rows = []
for item in eval_records:
    vid = item['youtube_id']
    vid_dir = os.path.join(base_eval, vid)
    os.makedirs(vid_dir, exist_ok=True)
    
    # Write metadata.json
    with open(os.path.join(vid_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(item, f, indent=2, ensure_ascii=False)
        
    # Write transcript.txt
    with open(os.path.join(vid_dir, 'transcript.txt'), 'w', encoding='utf-8') as f:
        f.write(f"# Video Title: {item['title']}\n")
        f.write(f"# YouTube ID: {item['youtube_id']}\n")
        f.write(f"# URL: {item['url']}\n")
        f.write(f"# Complication: {item['complication_category']}\n\n")
        f.write(item['transcript'].strip() + "\n")
        
    master_rows.append({
        'youtube_id': item['youtube_id'],
        'title': item['title'],
        'url': item['url'],
        'complication_category': item['complication_category'],
        'surgical_stage': item['surgical_stage'],
        'instruments': ', '.join(item['instruments']),
        'anatomy': ', '.join(item['anatomy']),
        'summary': item['summary']
    })

# Write master index JSON
with open(os.path.join(base_eval, 'complication_evaluation_dataset.json'), 'w', encoding='utf-8') as f:
    json.dump(eval_records, f, indent=2, ensure_ascii=False)

# Write master index CSV
with open(os.path.join(base_eval, 'complication_evaluation_dataset.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
    writer.writeheader()
    writer.writerows(master_rows)

print(f"Successfully created evaluation dataset with {len(eval_records)} videos in:\n  {base_eval}")
