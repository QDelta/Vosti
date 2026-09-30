import copy
import unittest

from scripts.determinism_tests.native_calls import assemble_calls
from scripts.determinism_tests.native_observation import schedule_events


def fixture():
    calls = [dict(label='first', prompts=[[1, 2]], max_tokens=2,
                  record_last_rows=True, record_generated_position=0, record_all_rows=True),
             dict(label='followup', max_tokens=2, record_last_rows=True,
                  output_prefix_from=dict(call=0, request=0, count=1, suffix=[9]))]
    actual = [dict(label=call['label'], request_id_base=i, input_token_ids=[prompt],
                   output_token_ids=[[3,4]], cache_isolated=True, cache_page_tokens=64,
                   graph_stats=dict(cover_replay_count=0, poisoned_reason=None))
              for i, (call, prompt) in enumerate(zip(calls, [[1,2], [1,2,3,9]]))]
    steps, rows, layouts = [], [], []
    for i, length in enumerate([2,4]):
        for j in range(2):
            key, phase = f'{i}:{j}', f'call-{i}'
            steps.append(dict(engine_step=key, request_id_base=i, step=j, scheduled=[i],
                              emitted=[3+j], cached_prefix_blocks=[0], arrived=[i] if j==0 else []))
            rows.append(dict(engine_step=key, phase=phase, batch_index=0, argmax=3+j, finite=True))
            layouts.extend(dict(engine_step=key, phase=phase, lengths=[0,value])
                           for value in ([length,length] if j==0 else [1,length+1]))
    return calls, actual, rows, steps, layouts


class NativeCallsTests(unittest.TestCase):
    def assemble(self, data):
        return assemble_calls(*data, caching=False,
                              copy_row=lambda row,c,r,p: dict(metadata={}, argmax=row['argmax']))

    def test_all_positions_and_generated_prefix(self):
        result = self.assemble(fixture())
        self.assertEqual(result[1]['requests'][0]['input_token_ids'], [1,2,3,9])
        self.assertEqual(result[0]['requests'][0]['last_row']['argmax'], 3)
        self.assertEqual([row['argmax'] for row in result[0]['requests'][0]['output_rows']], [3,4])
        self.assertEqual(result[1]['requests'][0]['last_row']['argmax'], 4)
        self.assertEqual(result[1]['schedule_events'][0][0]['query_tokens'], 4)

    def test_rejects_cache_identity_and_missing_predictions(self):
        for change in ('cache', 'phase', 'output', 'graph', 'input', 'layout'):
            data = fixture()
            if change == 'cache': data[3][0]['cached_prefix_blocks'] = [1]
            if change == 'phase': data[2][0]['phase'] = 'wrong'
            if change == 'output': data[1][0]['output_token_ids'] = [[3]]
            if change == 'graph': data[1][0]['graph_stats']['poisoned_reason'] = 'bad'
            if change == 'input': data[1][1]['input_token_ids'] = [[1,2,8,9]]
            if change == 'layout': data[4].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.assemble(data)

    def test_layout_pair_shape_and_order(self):
        _, _, _, steps, layouts = fixture()
        rows = schedule_events(layouts, steps)
        self.assertEqual(rows['0:1'], [dict(request_id='0', query_tokens=1, prefix_tokens=2)])
        for bad in (layouts+[layouts[-1]], layouts[1:], copy.deepcopy(layouts)):
            if len(bad)==len(layouts): bad[0]['lengths'] = [0,3]
            with self.assertRaises(ValueError): schedule_events(bad, steps)

    def test_non_rectangular_batch(self):
        steps = [dict(engine_step='0:0', scheduled=[4,2])]
        layouts = [dict(engine_step='0:0', phase='call-0', lengths=x)
                   for x in ([0,64,65], [0,64,322])]
        self.assertEqual(schedule_events(layouts, steps)['0:0'], [
            dict(request_id='4', query_tokens=64, prefix_tokens=0),
            dict(request_id='2', query_tokens=1, prefix_tokens=257)])
