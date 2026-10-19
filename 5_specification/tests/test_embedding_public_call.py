"""The public embedding API and ATen have different argument orders/defaults."""
import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from kernel_spec.num import Invalid, Unsupported


class EmbeddingPublicCall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'adapters/INDEX.py'
        spec = importlib.util.spec_from_file_location('index_embedding_public_review', path)
        cls.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.adapter)

    @staticmethod
    def record(value):
        return {'t': 'i64' if value.dtype == torch.int64 else 'f32',
                'shape': list(value.shape), 'v': value.tolist()}

    def test_public_order_keywords_and_explicit_defaults(self):
        weight = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        indices = torch.tensor([[2, 0], [1, 2]])
        w, i = self.record(weight), self.record(indices)
        expected = self.record(F.embedding(indices, weight))
        for args, kwargs in (([i, w], {}), ([], {'input': i, 'weight': w}),
                             ([i], {'weight': w, 'padding_idx': None, 'max_norm': None,
                                    'norm_type': 2.0, 'scale_grad_by_freq': False, 'sparse': False})):
            query = {'op': 'torch.nn.functional.embedding', 'args': args, 'kwargs': kwargs}
            with self.subTest(query=query):
                self.assertEqual(self.adapter.reference('real', query).fields['out'], expected['v'])
                self.assertTrue(self.adapter.check_output('real', query, {'out': expected}))

    def test_native_order_and_minus_one_sentinel_are_preserved(self):
        w = {'t': 'f32', 'v': [[1, 2], [3, 4]]}
        i = {'t': 'i64', 'v': [1, 0, 1]}
        native = {'op': 'aten.embedding', 'args': [w, i, -1, False, False]}
        self.assertEqual(self.adapter.reference('real', native).fields['out'], [[3, 4], [1, 2], [3, 4]])
        with self.assertRaises(Invalid):
            self.adapter.reference('real', {'op': 'aten.embedding', 'args': [w, i, None]})

    def test_mutation_and_nondefault_public_tasks_are_refused(self):
        w = {'t': 'f32', 'v': [[3, 4], [5, 6]]}
        i = {'t': 'i64', 'v': [0, 1]}
        for kwargs in ({'max_norm': 1}, {'padding_idx': -1}, {'padding_idx': 0},
                       {'norm_type': 1}, {'scale_grad_by_freq': True}, {'sparse': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(Unsupported):
                self.adapter.reference('real', {'op': 'torch.nn.functional.embedding', 'args': [i, w],
                                               'kwargs': kwargs})

    def test_public_decoding_stays_strict(self):
        w = {'t': 'f32', 'v': [[1, 2], [3, 4]]}
        i = {'t': 'i64', 'v': [0, 1]}
        for args, kwargs, error in (
            ([w, i], {}, Unsupported),
            ([{'t': 'i32', 'v': [0, 1]}, w], {}, Unsupported),
            ([{'t': 'i64', 'v': [0.9]}, w], {}, Invalid),
            ([{'t': 'i64', 'v': [2]}, w], {}, Invalid),
            ([i, w], {'input': i}, Invalid),
            ([i, w], {'norm_type': True}, Invalid),
            ([i, w], {'out': w}, Unsupported),
        ):
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(error):
                self.adapter.reference('real', {'op': 'torch.nn.functional.embedding', 'args': args,
                                               'kwargs': kwargs})

    def test_scalar_and_empty_index_shapes_match_public_api(self):
        weight = torch.tensor([[1., 2.], [3., 4.]])
        for indices in (torch.tensor(1), torch.empty((0,), dtype=torch.int64),
                        torch.empty((2, 0), dtype=torch.int64)):
            query = {'op': 'torch.nn.functional.embedding',
                     'args': [self.record(indices), self.record(weight)]}
            with self.subTest(shape=list(indices.shape)):
                self.assertTrue(self.adapter.check_output('real', query,
                    {'out': self.record(F.embedding(indices, weight))}))
