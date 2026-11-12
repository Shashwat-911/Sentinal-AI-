"""Summarize saved runs without treating dataset labels as frame-level truth."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/'evaluation/context_review'
RUNS=['suite_baseline','suite_separated_4b','suite_separated_8b','suite_short_4b']

def main():
    windows=json.loads((ROOT/'evaluation/demo_windows.json').read_text())
    data={}
    for run in RUNS:
        path=ARTIFACTS/run/'results.json'
        rows=json.loads(path.read_text()) if path.exists() else []
        data[run]={item['window']['file']:item for item in rows}
    lines=['# Local 15-video review', '',
           'Each row is a selected window, not a full-video accuracy score. Names and reviewer notes are excluded from model prompts.', '',
           '| Video | Requested window (s) | Baseline 4B | Revised 4B | Revised 8B | Short-window 4B |',
           '| --- | --- | --- | --- | --- | --- |']
    for window in windows:
        cells=[]
        for run in RUNS:
            item=data[run].get(window['file'])
            if item is None: cells.append('Not completed'); continue
            if 'error' in item: cells.append('ERROR'); continue
            cells.append(item['report']['classification'].replace('_',' ') + f" ({item['seconds']:.0f}s)")
        lines.append(f"| {Path(window['file']).stem} | {window['start']}–{window['end']} | " + ' | '.join(cells) + ' |')
    detector=ARTIFACTS/'detector_suite/summary.json'
    if detector.exists():
        rows=json.loads(detector.read_text())
        lines.extend(['','## Decode and detector smoke check','',
            f"{len(rows)} files; {sum(r['decoded'] for r in rows):,} decoded frames; {sum(r['sampled_frames'] for r in rows)} sampled person/pose checks.",
            'Inference was sampled every two seconds plus the final frame. This does not measure bounding-box precision/recall or validate full-rate tracking.'])
    lines.extend(['','## Review criteria',''])
    for window in windows:
        lines.append(f"- {Path(window['file']).stem}: {window['review']}.")
    lines.extend(['','## Artifact locations','',
        'Each run folder contains JSON reports, readable text, sampled-frame contact sheets and full-clip overview sheets. Revised runs include source hashes.',
        'The synthetic image-transport diagnostic recognized all three shapes with both request formats. An upstream merge bug is not established on this installation.',
        'Correct high-level labels do not prove correct action descriptions. Inspect the reports and cited frames before counting a case as successful.'])
    target=ARTIFACTS/'SUITE_RESULTS.md'
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(target)

if __name__=='__main__': main()
