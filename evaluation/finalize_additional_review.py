"""Assemble final reviewed reports from the ten-video run and a targeted policy rerun."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentinel.context_report import format_report, review_output
SOURCES = [
    'sentinel/context.py', 'sentinel/context_report.py', 'sentinel/context_views.py',
    'evaluation/review_demo_context.py', 'evaluation/review_suite.py', 'evaluation/finalize_additional_review.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-run', required=True)
    parser.add_argument('--car-run', required=True)
    parser.add_argument('--street-run', required=True)
    parser.add_argument('--output-run', required=True)
    args = parser.parse_args()
    if any(Path(value).name != value for value in (args.batch_run, args.car_run, args.street_run, args.output_run)):
        parser.error('run names must be single directory names')
    artifact_root = ROOT / 'evaluation/context_review'
    batch = artifact_root / args.batch_run
    car = artifact_root / args.car_run
    street = artifact_root / args.street_run
    output = artifact_root / args.output_run
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    batch_results = json.loads((batch / 'results.json').read_text(encoding='utf-8'))
    car_file = next((name for name in (car / 'results.json').exists() and
                     json.loads((car / 'results.json').read_text(encoding='utf-8')) or []
                     if name.get('report', {}).get('classification') == 'suspicious'), None)
    if car_file is None:
        car_file = next((p for p in car.glob('pixabay_thief_car_*.json')), None)
        if car_file is None:
            raise RuntimeError('The targeted final car report is missing')
        car_result = json.loads(car_file.read_text(encoding='utf-8'))
    else:
        car_result = car_file
    street_data = json.loads((street / 'results.json').read_text(encoding='utf-8'))
    if len(street_data) != 1 or street_data[0].get('error'):
        raise RuntimeError('The full-window street report is missing or incomplete')
    street_result = street_data[0]
    output.mkdir(parents=True)
    final_results = []
    for original in batch_results:
        filename = Path(original['window']['file']).stem
        is_targeted = filename == 'pixabay_thief_car_3582'
        is_street = filename == 'pixabay_street_crowd_1694'
        if is_targeted:
            item = car_result
            origin = args.car_run
        elif is_street:
            item = street_result
            origin = args.street_run
        else:
            item = original
            origin = args.batch_run
        case = f"{filename}_{item['window']['start']}_{item['window']['end']}"
        if 'report' not in item or item.get('error'):
            raise RuntimeError(f'Incomplete report: {case}')
        item['report'] = review_output(item['report'], item.get('packet', {}))
        item['report_origin'] = origin
        (output / f'{case}.json').write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding='utf-8')
        (output / f'{case}.txt').write_text(
            format_report(item['report'], item.get('packet', {}), item['window']['end']), encoding='utf-8')
        (output / f'{case}.jpg').write_bytes((car / f'{case}.jpg' if is_targeted else batch / f'{case}.jpg').read_bytes())
        overview = batch / f'{filename}_overview.jpg'
        if overview.exists():
            shutil.copy2(overview, output / overview.name)
        final_results.append(item)
    (output / 'results.json').write_text(json.dumps(final_results, indent=2, ensure_ascii=False), encoding='utf-8')
    shutil.copy2(batch / 'review_windows.json', output / 'review_windows.json')
    shutil.copy2(batch / 'inventory.json', output / 'inventory.json')
    detector = artifact_root / 'additional_pixabay_qwen3vl4b_scaled/detector'
    if detector.exists():
        shutil.copytree(detector, output / 'detector')
    config = {
        'model': final_results[0]['model'],
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
        'report_sources': {'eight_full_clip_reviews': args.batch_run,
                           'vehicle_item_removal_policy_regression': args.car_run,
                           'full_storefront_clip_after_scene-cut_fix': args.street_run},
        'note': 'Composite final report bundle: eight reports from the ten-clip batch plus final targeted vehicle-item and full storefront reruns under the final policies.',
    }
    (output / 'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(output)

if __name__ == '__main__':
    main()
