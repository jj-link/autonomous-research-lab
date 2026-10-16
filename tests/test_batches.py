"""Input projection and resource boundaries for coordinator-owned CPU batches."""
import json
from pathlib import Path
import shutil
import unittest
import uuid

from autonomous_research_lab import batches
from autonomous_research_lab.models import ExperimentSpec, LabError
from autonomous_research_lab.research import ResearchEngine
from autonomous_research_lab.store import Store


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1] / '.arl-cache' / 'tests' / uuid.uuid4().hex
        self.engine = ResearchEngine(Store(self.folder / 'state'))
        self.data = self.folder / 'data'
        self.data.mkdir()
        (self.data / 'a.json').write_text('{"private":2}')
        (self.data / 'b.json').write_text('{"private":3}')
        self.input_id = self.engine.store.register_input(self.data, 'private-cases')['id']
        self.code = self.folder / 'code'
        self.code.mkdir()
        (self.code / 'main.py').write_text('raise RuntimeError("unit fixture must not execute")\n')
        self.rows = [{'key': key, 'settings': {'case': key}, 'wall_seconds': 20,
                      'inputs': {'case': {'input': 'private', 'files': {'case.json': key + '.json'}}}}
                     for key in ('a', 'b')]

    def tearDown(self):
        shutil.rmtree(self.folder)

    def prepare(self, rows=None, descriptor=None, max_attempts=3):
        index = self.folder / ('index-' + uuid.uuid4().hex)
        index.mkdir()
        (index / 'jobs.json').write_text(json.dumps({'jobs': self.rows if rows is None else rows}))
        index_id = self.engine.store.register_input(index, 'job-index')['id']
        program = self.engine.create({'goal': 'Check isolated input projection',
            'model': {'base_url': 'https://api.example.com/v1', 'model': 'fixture'},
            'inputs': {'private': self.input_id, 'index': index_id}, 'private_inputs': ['private'],
            'max_parallel_workers': 4, 'max_experiments': max_attempts,
            'resources': {'memory_bytes': 128*1024**2, 'output_bytes': 1024**2, 'pids': 128}})
        spec = ExperimentSpec.from_dict({'name': 'batch-fixture', 'source': str(self.code), 'entrypoint': 'main.py',
            'inputs': program['brief']['inputs'], 'limits': {'wall_seconds': 30, 'memory_bytes': 64*1024**2, 'pids': 32}}, self.code)
        result = batches.prepare(self.engine, program, {}, spec, {'input': 'index', 'path': 'jobs.json', 'parallelism': 2, **(descriptor or {})})
        return program, result

    def test_each_worker_receives_only_its_own_private_file(self):
        program, batch = self.prepare()
        self.assertTrue(batch['private'])
        records = {}
        for job in batch['jobs']:
            manifest = self.engine.store.get_experiment(job['experiment_id'])['manifest']
            self.assertEqual(set(manifest['inputs']), {'case'})
            record = self.engine.store.verify_input(manifest['inputs']['case'])
            self.assertEqual([item['path'] for item in record['files']], ['case.json'])
            records[job['key']] = json.loads((self.engine.store.input_source(record['id']) / 'case.json').read_text())
        self.assertEqual(records, {'a': {'private': 2}, 'b': {'private': 3}})
        self.assertEqual(self.engine.store.list_attempts(), [])

    def test_projection_cannot_escape_or_select_an_unapproved_source(self):
        for projection in ({'input': 'unapproved', 'files': {'case.json': 'a.json'}},
                           {'input': 'private', 'files': {'../secret': 'a.json'}},
                           {'input': 'private', 'files': {'case.json': '../a.json'}},
                           {'input': 'private', 'files': {'case.json': 'missing.json'}}):
            with self.subTest(projection=projection), self.assertRaises(LabError):
                self.prepare(rows=[{**self.rows[0], 'inputs': {'case': projection}}])
        self.assertEqual(self.engine.store.list_attempts(), [])

    def test_duplicate_keys_and_incomplete_repair_selections_are_rejected(self):
        with self.assertRaises(LabError):
            self.prepare(rows=[self.rows[0], self.rows[0]])
        with self.assertRaises(LabError):
            self.prepare(descriptor={'keys': ['missing']})
        _, batch = self.prepare(descriptor={'keys': ['b']})
        self.assertEqual([job['key'] for job in batch['jobs']], ['b'])

    def test_concurrency_cannot_exceed_aggregate_memory_permission(self):
        with self.assertRaises(LabError):
            self.prepare(descriptor={'parallelism': 3})
        self.assertEqual(self.engine.store.list_experiments(), [])

    def test_attempt_budget_blocks_batch_before_execution(self):
        with self.assertRaises(LabError):
            self.prepare(max_attempts=1)
        self.assertEqual(self.engine.store.list_attempts(), [])

    def test_declared_directory_does_not_capture_sibling_paths(self):
        spec = ExperimentSpec.from_dict({'name': 'output-boundary', 'source': str(self.code), 'entrypoint': 'main.py',
                                         'outputs': ['results.json', 'responses/']}, self.code)
        candidates = ['results.json', 'responses/a.json', 'responses/nested/b.json', 'responses-extra/secret.json', 'scratch.json']
        self.assertEqual([path for path in candidates if spec.includes_output(path)], candidates[:3])
        self.assertEqual(spec.missing_outputs(['results.json', 'responses-extra/secret.json']), {'responses/'})


if __name__ == '__main__':
    unittest.main()
