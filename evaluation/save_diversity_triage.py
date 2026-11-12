import json
from pathlib import Path
root=Path('evaluation/context_review/diversity_50_v1')
notes={
'Arrest001':('safety_review','Close body contact and pushing/restraint near doorway at about 41-50s.'),
'Arrest027':('uncertain','Moving traffic cameras and construction; distant vehicle maneuvers require closer temporal review.'),
'Arrest051':('safety_review','Crowd with a person low/on ground around 130-210s; interaction requires safety review.'),
'Arson001':('safety_review','Visible flames and smoke near building at about 66-118s. Ignore accusation title cards.'),
'Arson053':('safety_review','Visible spreading bright flames on floor/wall around 154-214s; multiple camera views.'),
'Assault001':('safety_review','Several people converge on a person on ground around 9-30s; later assistance/aftermath.'),
'Assault052':('safety_review','Physical conflict and person on ground outside storefront around 7-18s.'),
'Burglary001':('review','Barrier collapse/debris followed by people moving large objects around 14-70s.'),
'Burglary035':('uncertain','Person opens gate/door around 50-63s; authorization and forced entry not established.'),
'Burglary068':('review','Abrupt handling/removal from wall display around 8-14s then leaving; permission unknown.'),
'Burglary100':('uncertain','Small person behind barred counter handles area; fine hand actions cannot be resolved.'),
'Fighting002':('uncertain','Crowded indoor interaction with changed camera view; sampled overview insufficient to resolve contact.'),
'Fighting026':('safety_review','People converge with apparent forceful contact in crowded dining area around 18-43s.'),
'Fighting051':('safety_review','Extended close physical interaction around 46-116s near tables; some bodies obscured.'),
'RoadAccidents003':('safety_review','Abrupt intrusion/contact and aftermath near entrance around 10-16s.'),
'RoadAccidents033':('uncertain','Night road and moving headlights; collision timing cannot be resolved in overview.'),
'RoadAccidents063':('uncertain','Busy night intersection with crossing pedestrians; brief incident may fall between samples.'),
'RoadAccidents093':('uncertain','Busy junction with overlapping vehicles; contact needs denser frames.'),
'RoadAccidents122':('uncertain','Distant road event near upper edge; detail too small to confirm impact.'),
'RoadAccidents151':('uncertain','Highway traffic converges; overview does not reliably resolve exact contact.'),
'Robbery001':('review','Person reaches across small shop counter toward another; sequence requires closer review, weapon unresolved.'),
'Robbery150':('uncertain','Small night street interaction around 45-72s; intent and transfer cannot be resolved.'),
'Shoplifting003':('uncertain','Shelf handling and objects near clothing/bag; concealment unresolved at overview resolution.'),
'Shoplifting020':('uncertain','Several people interact at retail counter; ordinary exchange and concealment need closer detail.'),
'Shoplifting038':('uncertain','Person handles display merchandise, later view changes; ownership/permission unknown.'),
'Shoplifting055':('uncertain','Shelf browsing and doorway views; small item transfers not resolved.'),
'Stealing002':('review','Person works near parked vehicle and hood rises around 49-55s; authorization unknown.'),
'Stealing043':('uncertain','People at parked vehicle in overexposed night images; hand actions difficult to resolve.'),
'Stealing078':('review','People handle motorcycle then ride it away around 45-73s; ownership unknown, no clear fall in overview.'),
'Stealing114':('uncertain','Person briefly reaches toward parked car around 6-13s, then walks away; transfer unresolved.'),
'Vandalism001':('review','Person interacts with pole/camera area with abrupt view changes; detail needed to establish damage.'),
'Vandalism026':('review','Extended arm and apparent object/contact action toward storefront around 14-19s; damage unresolved.'),
'Vandalism050':('review','Person repeatedly interacts with side of parked car around 6-24s; damage/intent unresolved.'),
}
normal={15:'People walk in shopping corridor.',48:'Office workers sit and move between desks.',129:'People walk along lit street.',217:'Pedestrians pass fuel-station forecourt.',317:'People stand and move near reception.',453:'People interact at service counter.',656:'Parked vehicles and pedestrians in residential area.',745:'People use bowling lanes and seating area.',828:'Cars pass parking barrier.',873:'People handle boxes in warehouse.',882:'People walk and stand in lecture room.',891:'People visit exhibition booths.',900:'People walk among parked cars.',909:'Traffic on road and flyover with changing view.',925:'Parked cars and changing outdoor views.',934:'People move through shop aisle.',944:'People gather at exhibition counter.'}
for n,note in normal.items():notes[f'Normal_Videos_{n:03}']=('no_concern_visible',note)
rows=[]
for name,(judgment,note) in sorted(notes.items()):
    folder=root/(name+'_x264'); reports=[]
    for p in folder.glob('*.json'):
        if p.name=='telemetry.json':continue
        d=json.loads(p.read_text()); reports.append(dict(start=d['window']['start'],classification=d.get('report',{}).get('classification'),error=d.get('error')))
    flagged=any(r['classification'] in ('suspicious','safety_concern') for r in reports)
    issue=('missed_review_candidate' if judgment in ('review','safety_review') and not flagged else 'requires_detail_review' if judgment=='uncertain' else 'broadly_consistent_at_clip_level')
    rows.append(dict(video=name+'_x264',independent_judgment=judgment,observation=note,comparison=issue,project_windows=reports,review_method='Assistant visual triage of 20 evenly spaced overview frames; not continuous playback or per-window ground truth. Dataset labels not used as verdicts.'))
out=root/'independent_audit'; (out/'visual_triage.json').write_text(json.dumps(rows,indent=2))
print('Saved',len(rows),'individual visual triage records')
print('Missed review candidates:',[r['video'] for r in rows if r['comparison']=='missed_review_candidate'])
