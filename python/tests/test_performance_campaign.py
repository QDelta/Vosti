import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pytest

from scripts.serving_benchmark.campaign import (
    checked_alias, inspect_trial, load_exclusions, validate_rate_exclusions,
)


class PerformanceCampaignTests(unittest.TestCase):
    def test_alias_never_overwrites_an_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, alias = root / 'target', root / 'alias'
            target.mkdir()
            checked_alias(alias, target)
            checked_alias(alias, target)
            self.assertEqual(alias.resolve(), target.resolve())
            with self.assertRaisesRegex(RuntimeError, 'refusing to replace'):
                checked_alias(alias, root / 'other')

    def test_incomplete_trial_cannot_be_imported_as_success(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('scripts.serving_benchmark.campaign.load_complete_telemetry'):
            root = Path(directory)
            (root / 'status.json').write_text(json.dumps(dict(complete=False)))
            with self.assertRaisesRegex(RuntimeError, 'did not finish'):
                inspect_trial(root, root / 'telemetry.json')

    def test_trial_jobs_must_match_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('scripts.serving_benchmark.campaign.load_complete_telemetry'):
            root = Path(directory)
            (root / 'status.json').write_text(json.dumps(dict(complete=True, jobs=[dict(id='wrong', returncode=0)])))
            (root / 'inputs.json').write_text(json.dumps(dict(jobs=[dict(id='expected')])) )
            with self.assertRaisesRegex(RuntimeError, 'differ from its input'):
                inspect_trial(root, root / 'telemetry.json')


class ExclusionTests(unittest.TestCase):
    def test_calibration_and_execution_use_same_exclusions(self):
        exclusions = {'alpha/fa3': 'backend unavailable'}
        plans = [dict(rate_selection=dict(exclusions=exclusions))]
        validate_rate_exclusions(plans, exclusions)
        for supplied in ({}, {'alpha/fa3': 'different reason'}):
            with self.assertRaisesRegex(RuntimeError, 'same exclusions'):
                validate_rate_exclusions(plans, supplied)
        validate_rate_exclusions([dict(scope='capacity')], exclusions)
        validate_rate_exclusions([dict(rate_selection=dict(schema='full-calibration'))], {})

    def test_only_planned_pairs_with_reasons_can_be_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'exclusions.json'
            pairs = {'alpha/fa3', 'alpha/triton'}
            self.assertEqual(load_exclusions(None, pairs), {})
            valid = {'alpha/fa3': 'backend unavailable'}
            path.write_text(json.dumps(valid))
            self.assertEqual(load_exclusions(path, pairs), valid)
            for invalid in ([], {'unknown/fa3': 'reason'}, {'alpha/fa3': ''},
                            {'alpha/fa3': ' '}, {'alpha/fa3': None}):
                path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, 'planned checkpoint/mode'):
                    load_exclusions(path, pairs)


def test_excluded_trials_remain_partial_and_resume_checks_inputs(tmp_path, monkeypatch):
    from scripts.serving_benchmark import campaign

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    spec = write('spec.json', dict(checkpoint=dict(key='alpha'), execution=dict(key='fa3')))
    jobs = write('jobs.json', [])
    trial = dict(id='alpha-fa3', spec=str(spec), jobs=str(jobs), output=str(tmp_path / 'trial'))
    plan = write('plan.json', dict(framework_source='frozen', trials=[trial]))
    exclusions = write('exclusions.json', {'alpha/fa3': 'backend unavailable'})
    output = tmp_path / 'campaign'
    argv = ['campaign', '--root', str(tmp_path), '--stack-root', str(tmp_path),
            '--build', str(tmp_path), '--plan', str(plan), '--output', str(output),
            '--exclusions', str(exclusions)]
    monkeypatch.setattr('sys.argv', argv)
    monkeypatch.setattr(campaign, 'source_identity', lambda root: dict(framework='frozen', kernel='fixed'))
    monkeypatch.setattr(campaign, 'package_inventory', lambda path: {})
    monkeypatch.setattr(campaign.subprocess, 'check_output', lambda *a, **kw: 'coordinator')
    with patch.object(campaign.subprocess, 'call') as launch:
        campaign.main()
        assert not launch.called
        assert not (output / 'complete.json').exists()
        partial = json.loads((output / 'partial.json').read_text())
        assert partial['unresolved'] == {'alpha-fa3': {'status': 'unrun'}}
        assert partial['exclusions'] == {'alpha/fa3': 'backend unavailable'}
        monkeypatch.setattr('sys.argv', [*argv, '--resume'])
        campaign.main()
        assert not launch.called
        exclusions.write_text(json.dumps({'alpha/fa3': 'changed reason'}))
        with pytest.raises(RuntimeError, match='inputs, packages, source or coordinator changed'):
            campaign.main()
