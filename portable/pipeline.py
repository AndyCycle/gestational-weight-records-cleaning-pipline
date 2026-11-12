"""Standard-library CSV pipeline. Cohort-specific inference stays explicit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ['record_id', 'pregnancy_id', 'gestation_day', 'weight_raw', 'weight',
          'height', 'BMI', 'delivery_day', 'flags', 'excluded']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('CSV requires unique column names')
        return list(reader), reader.fieldnames


def number(value, name, optional=False):
    if optional and (value is None or str(value).strip() == ''):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name}: finite numeric value required')
    return result


def validate_config(config):
    allowed = {'columns', 'units', 'rules'}
    if set(config) - allowed:
        raise ValueError(f'Unknown config keys: {set(config) - allowed}')
    columns = config['columns']
    required = {'record_id', 'pregnancy_id', 'gestation_day', 'weight'}
    if not required <= columns.keys() or set(columns) - required - {'height', 'delivery_day'}:
        raise ValueError('Map record_id, pregnancy_id, gestation_day, weight; optional height/delivery_day')
    if len(set(columns.values())) != len(columns):
        raise ValueError('Column mappings must be distinct')
    units = config['units']
    if set(units) != {'weight', 'height'} or units['weight'] not in {'kg', 'jin', 'mixed'} or units['height'] not in {'cm', 'm'}:
        raise ValueError('Units: weight=kg|jin|mixed; height=cm|m')
    rules = config.get('rules', {})
    if set(rules) - {'weight_range', 'local_spike_kg', 'neighbor_agreement_kg', 'exclude_postpartum'}:
        raise ValueError('Unknown rule name')
    lo, hi = rules.get('weight_range', [30, 140])
    if not 0 < lo < hi or not all(math.isfinite(x) for x in (lo, hi)):
        raise ValueError('weight_range requires finite 0 < lower < upper')
    for key in ('local_spike_kg', 'neighbor_agreement_kg'):
        if key in rules and (not math.isfinite(rules[key]) or rules[key] <= 0):
            raise ValueError(f'{key} must be positive and finite')
    if 'exclude_postpartum' in rules and not isinstance(rules['exclude_postpartum'], bool):
        raise ValueError('exclude_postpartum must be boolean')
    if rules.get('exclude_postpartum') and 'delivery_day' not in columns:
        raise ValueError('Postpartum exclusion requires delivery_day mapping')


def normalize(raw, headers, config):
    validate_config(config)
    columns = config['columns']
    if set(columns.values()) - set(headers):
        raise ValueError(f'Missing columns: {set(columns.values()) - set(headers)}')
    records, seen = [], set()
    for source in raw:
        values = {key: source[col] for key, col in columns.items()}
        rid, pid = values['record_id'].strip(), values['pregnancy_id'].strip()
        if not rid or not pid or rid in seen:
            raise ValueError('record_id must be globally unique; pregnancy_id must be present')
        seen.add(rid)
        weight = number(values['weight'], 'weight', optional=True)
        height = number(values.get('height'), 'height', optional=True)
        day = number(values['gestation_day'], 'gestation_day')
        delivery = number(values.get('delivery_day'), 'delivery_day', optional=True)
        if height is not None:
            height *= 100 if config['units']['height'] == 'm' else 1
            if height <= 0:
                raise ValueError('height must be positive or blank')
        flags = []
        if day < 0:
            flags.append('negative_gestation_day')
        if config['units']['weight'] == 'mixed':
            flags.append('unit_unresolved')
        records.append(dict(record_id=rid, pregnancy_id=pid, gestation_day=day,
                            weight_raw=weight,
                            weight=weight / 2 if weight is not None and config['units']['weight'] == 'jin' else weight,
                            height=height, BMI=None, delivery_day=delivery,
                            flags=flags, excluded=False))
    return records


def apply_review(records, actions):
    lookup = {r['record_id']: r for r in records}
    used, audit = set(), []
    for a in actions:
        rid = a['record_id'].strip()
        if rid in used or rid not in lookup:
            raise ValueError('Review requires one final action per existing record_id')
        used.add(rid)
        r = lookup[rid]
        expected = number(a['expected_weight'], 'expected_weight', optional=True)
        if r['weight'] != expected or not a['reason'].strip():
            raise ValueError(f'Review precondition failed for {rid}')
        old = r['weight']
        if a['operation'] == 'set_weight':
            r['weight'] = number(a['new_weight'], 'new_weight')
            if r['weight'] <= 0:
                raise ValueError('Reviewed weight must be positive')
            r['flags'] = [f for f in r['flags'] if f != 'unit_unresolved']
        elif a['operation'] == 'exclude':
            r['excluded'] = True
        else:
            raise ValueError('Review operation must be set_weight or exclude')
        audit.append(dict(record_id=rid, operation=a['operation'], old_weight=old,
                          new_weight=r['weight'], reason=a['reason']))
    return audit


def screen(records, config):
    rules = config.get('rules', {})
    lo, hi = rules.get('weight_range', [30, 140])
    groups = defaultdict(list)
    for r in records:
        groups[r['pregnancy_id']].append(r)
    for rows in groups.values():
        deliveries = {r['delivery_day'] for r in rows if r['delivery_day'] is not None}
        if len(deliveries) > 1:
            raise ValueError('Conflicting delivery days within a pregnancy')
        delivery = next(iter(deliveries), None)
        counts = Counter(r['gestation_day'] for r in rows)
        for r in rows:
            if counts[r['gestation_day']] > 1:
                r['flags'].append('duplicate_pregnancy_day')
            w = r['weight']
            if w is None:
                r['flags'].append('missing_weight')
            elif not lo <= w <= hi:
                r['flags'].append('weight_outside_review_range')
            if w is not None and r['height'] and 'unit_unresolved' not in r['flags']:
                r['BMI'] = round(w / (r['height'] / 100) ** 2, 2)
            if delivery is not None and r['gestation_day'] > delivery:
                r['flags'].append('after_delivery')
                if rules.get('exclude_postpartum', False):
                    r['excluded'] = True
        # Only compare uniquely dated, observed, antepartum records; never interpolate.
        valid = sorted((r for r in rows if r['weight'] is not None and not r['excluded']
                        and counts[r['gestation_day']] == 1 and 'after_delivery' not in r['flags']),
                       key=lambda r: r['gestation_day'])
        for left, mid, right in zip(valid, valid[1:], valid[2:]):
            ref = (left['weight'] + right['weight']) / 2
            if abs(left['weight'] - right['weight']) <= rules.get('neighbor_agreement_kg', 12):
                if abs(mid['weight'] - ref) > rules.get('local_spike_kg', 15):
                    mid['flags'].append('local_spike_review')
                    if ref > 0 and any(abs(mid['weight'] * factor - ref) <= 2 for factor in (0.5, 2)):
                        mid['flags'].append('factor_two_candidate')
    return records


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if isinstance(row.get('flags'), list):
                row['flags'] = ';'.join(row['flags'])
            writer.writerow(row)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run(input_path, config_path, output_dir, review_path=None):
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    raw, headers = read_csv(input_path)
    records = normalize(raw, headers, config)
    audit = apply_review(records, read_csv(review_path)[0]) if review_path else []
    screen(records, config)
    # New directory per run: protects the source and previous results.
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'records.csv', records, FIELDS)
    write_csv(out / 'review_queue.csv', [r for r in records if r['flags']], FIELDS)
    write_csv(out / 'changes.csv', audit, ['record_id', 'operation', 'old_weight', 'new_weight', 'reason'])
    summary = dict(input_sha256=sha256(input_path), config=config,
                   review_sha256=sha256(review_path) if review_path else None,
                   input_rows=len(raw), output_rows=len(records),
                   pregnancies=len({r['pregnancy_id'] for r in records}),
                   excluded_rows=sum(r['excluded'] for r in records),
                   reviewed_rows=len(audit),
                   flag_counts=dict(Counter(f for r in records for f in r['flags'])))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.config, args.output, args.review), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
