"""AI-weighted scene-level fusion for the evidence window (80% AI / 20% Rules); scores are not probabilities."""

AI_WEIGHT = 0.8
RULE_WEIGHT = 0.2


def classify_window(report, packet):
    people = [p for e in packet.get('evidence', []) for p in e.get('people', [])]
    strong = {'SUSPICIOUS_LOITERING', 'HIGH_CONFIDENCE_SUSPICIOUS', 'REPEATED_MOVEMENT', 'SUDDEN_RUNNING', 'LONG_DWELL'}

    # Determine rule concern from telemetry, alerts, and behavior states
    rule_concern = False
    if packet.get('rule_assessment') == 'review':
        rule_concern = True
    elif packet.get('safety_event'):
        rule_concern = True
    elif people:
        for p in people:
            signals = set(p.get('signals', []))
            behavior = str(p.get('behavior', 'NORMAL'))
            risk = str(p.get('risk_level', p.get('severity', 'NORMAL')))
            if (signals & strong) or behavior in ('SUSPICIOUS', 'ALERT', 'CRITICAL') or risk in ('MEDIUM', 'HIGH', 'CRITICAL'):
                rule_concern = True
                break

    rule = int(rule_concern) if (people or packet.get('safety_event')) else None
    scope = report.get('subject_scope', 'unclear')
    if scope in ('vehicles', 'animals_only', 'empty', 'unclear'):
        rule = None
    ai = {'suspicious': 1, 'not_suspicious': 0}.get(report.get('classification'))

    concern_percent = None
    if report.get('classification') == 'safety_concern' or packet.get('safety_event'):
        verdict = 'safety_concern'
        concern_percent = 100
    elif rule is not None and ai is not None:
        weighted = AI_WEIGHT * ai + RULE_WEIGHT * rule
        concern_percent = round(100 * weighted)
        verdict = 'suspicious' if weighted >= 0.5 else 'not_suspicious'
    elif rule is None and ai == 1:
        verdict = 'suspicious'
        concern_percent = round(100 * AI_WEIGHT)
    else:
        verdict = 'uncertain'

    requested = packet.get('requested_window', {})
    actual_start = min((e['time'] for e in packet.get('evidence', []) if 'time' in e), default=None)
    incomplete = actual_start is not None and 'start' in requested and actual_start > requested['start'] + 2
    if incomplete and verdict == 'not_suspicious':
        verdict = 'uncertain'
    return dict(classification=verdict, ai_weight=AI_WEIGHT, rule_weight=RULE_WEIGHT,
                incomplete_window_coverage=incomplete,
                rule_status='available' if rule is not None else 'unavailable_or_outside_person_rule_scope',
                ai_vote=ai, rule_vote=rule,
                concern_vote_percent=concern_percent)


def vote_summary(report, packet):
    vote = classify_window(report, packet)
    names = {None: 'abstain', 0: 'no concern', 1: 'concern'}
    ai_status = names[vote['ai_vote']]
    rule_status = names[vote['rule_vote']]
    pct = vote.get('concern_vote_percent')
    pct_str = f" ({pct}%)" if pct is not None else ""
    ai_pct = round(vote['ai_weight'] * 100)
    rule_pct = round(vote['rule_weight'] * 100)
    weights_line = f"AI {ai_pct}%: {ai_status} · Rules {rule_pct}%: {rule_status}"

    if vote['classification'] == 'suspicious':
        if vote['ai_vote'] == 1 and vote['rule_vote'] == 0:
            detail = "AI-weighted fusion flagged SUSPICIOUS based on visual context."
        else:
            detail = "Both AI vision and detection rules indicate concerning activity."
        return f"COMBINED: SUSPICIOUS{pct_str} · {weights_line}\n{detail}"
    elif vote['classification'] == 'not_suspicious':
        if vote['ai_vote'] == 0 and vote['rule_vote'] == 1:
            detail = "AI-weighted fusion aligned on NOT SUSPICIOUS (minor rule warning overridden)."
        else:
            detail = "Both AI vision and detection rules indicate normal activity."
        return f"COMBINED: NOT SUSPICIOUS{pct_str} · {weights_line}\n{detail}"
    elif vote['classification'] == 'safety_concern':
        return (f"COMBINED: SAFETY CONCERN · {weights_line}\n"
                "Safety event or physical hazard detected. Immediate operator attention required.")
    else:
        classification_str = vote['classification'].replace('_', ' ').upper()
        return (f"COMBINED: {classification_str}{pct_str} · {weights_line}\n"
                "Evidence-window decision, not a probability.")
