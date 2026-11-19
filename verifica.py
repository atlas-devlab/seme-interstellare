"""Run the focused test suite and a small reproducible channel campaign."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import demo

ROOT = Path(__file__).resolve().parent


def run():
    suite = unittest.defaultTestLoader.discover(str(ROOT),pattern='test_*.py')
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not tests.wasSuccessful():
        print(log.getvalue())
        raise SystemExit(1)
    campaign = []
    with tempfile.TemporaryDirectory(prefix='sif-campaign-') as directory:
        for loss in [0,0.35,0.6,1]:
            for seed in [0,1,2]:
                report = demo.demonstration(Path(directory)/f'{loss}-{seed}',loss=loss,seed=seed)
                if report['reconstruction']['status']=='complete':
                    assert report['source_bytes_match']
                    assert report['independent_output_check_passed']
                else:
                    assert report['execution'] is None
                campaign.append(dict(loss_probability=loss,seed=seed,
                    packets_lost=report['packets_lost'],frames_altered=report['frames_altered'],
                    status=report['reconstruction']['status'],
                    rank=report['reconstruction'].get('rank'),
                    required_rank=report['reconstruction'].get('required_rank'),
                    output_check=report['independent_output_check_passed']))
    baseline = demo.demonstration(ROOT/'risultati'/'demo')
    assert baseline['reconstruction']['status']=='complete'
    result = dict(all_checks_passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        unittest_cases=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),
        channel_campaign_cases=len(campaign),campaign_outcomes=dict(Counter(row['status'] for row in campaign)),
        campaign=campaign,
        default_demo=dict(packets_sent=baseline['packets_sent'],packets_lost=baseline['packets_lost'],
            frames_altered=baseline['frames_altered'],rank=baseline['reconstruction']['rank'],
            source_blocks=baseline['source_blocks'],output=baseline['execution']['output'],
            source_bytes_match=baseline['source_bytes_match']),
        code_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(ROOT.glob('*.py'))},
        scope='Local constructed message, bounded VM and file channel; not an AI model or Wow decoding.')
    demo.save_json(ROOT/'verifica.json',result)
    print(json.dumps({key:value for key,value in result.items()
                      if key not in ['campaign','code_sha256']},ensure_ascii=False))


if __name__=='__main__':
    run()
