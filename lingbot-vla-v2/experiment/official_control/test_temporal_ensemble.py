import math,unittest
from .temporal_ensemble import TemporalEnsemble
class TemporalEnsembleTests(unittest.TestCase):
    def test_aligns_same_execution_time_not_first_rows(self):
        state=TemporalEnsemble(decay=0)
        self.assertEqual(state.push([[1,0],[10,0],[100,0]]),[1,0])
        self.assertEqual(state.push([[20,0],[200,0],[2000,0]]),[15,0])
        self.assertEqual(state.push([[300,0],[3000,0],[30000,0]]),[200,0])
    def test_newer_exponential_weight_and_zero_actions_are_valid(self):
        state=TemporalEnsemble(decay=math.log(2));state.push([[9],[0]])
        self.assertAlmostEqual(state.push([[3],[4]])[0],2)
    def test_reset_and_separate_instances(self):
        a=TemporalEnsemble();b=TemporalEnsemble();a.push([[2],[4]])
        self.assertEqual(b.push([[10],[20]]),[10]);a.reset();self.assertEqual(a.push([[8]]),[8])
    def test_bounded_history_and_short_predictions(self):
        state=TemporalEnsemble(horizon=2,decay=0)
        for _ in range(10):state.push([[1],[3],[5]])
        self.assertEqual(len(state.history),2);self.assertEqual(state.last_count,2)
        state=TemporalEnsemble(decay=0);state.push([[10]]);self.assertEqual(state.push([[20]]),[20])
    def test_invalid_inputs_fail_without_silent_nan(self):
        for args in [dict(horizon=0),dict(horizon=True),dict(decay=-1),dict(decay=float('nan'))]:
            with self.assertRaises(ValueError):TemporalEnsemble(**args)
        for chunk in [[],[[]],[[1],[1,2]],[[float('nan')]],[[True]]]:
            with self.assertRaises(ValueError):TemporalEnsemble().push(chunk)
        state=TemporalEnsemble();state.push([[1]])
        with self.assertRaises(ValueError):state.push([[1,2]])


class TemporalClientTests(unittest.TestCase):
    def setUp(self):
        try:import numpy as np
        except ImportError:self.skipTest('numpy integration requires simulator environment')
        self.np=np
    def client_type(self):
        np=self.np
        class Client:
            def __init__(self,offset=0):self.offset=offset;self.calls=0
            def infer(self,obs):
                if obs.get('reset'):self.calls=0;return {'action':None}
                self.calls+=1
                base=self.offset+10*self.calls
                return {'action':np.array([[base,0],[base+2,0],[base+4,0]],dtype=np.float32),'server_timing':0.125}
        return Client
    def test_wrapper_one_action_fresh_requests_reset_and_audit(self):
        import tempfile,json
        from pathlib import Path
        from .temporal_ensemble import install_client
        Client=self.client_type()
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'audit.json';install_client(Client,horizon=2,decay=0,audit_path=path)
            client=Client();client.infer({'reset':True})
            first=client.infer({});second=client.infer({})
            self.assertEqual(first['action'].shape,(2,));self.assertEqual(second['action'].tolist(),[16,0])
            self.assertEqual(client.calls,2);self.assertEqual(second['server_timing'],0.125)
            report=json.loads(path.read_text());self.assertEqual(report['source_chunk_length'],3);self.assertEqual(report['active_predictions'],2)
            self.assertEqual(report['executed_actions_per_response'],1)
            client.infer({'reset':True});self.assertEqual(client.infer({})['action'].tolist(),[10,0])
    def test_multiple_clients_cannot_share_history(self):
        from .temporal_ensemble import install_client
        Client=self.client_type();install_client(Client,horizon=2,decay=0)
        left=Client();right=Client(1000)
        left.infer({'reset':True});right.infer({'reset':True});left.infer({});right.infer({})
        self.assertEqual(left.infer({})['action'].tolist(),[16,0]);self.assertEqual(right.infer({})['action'].tolist(),[1016,0])
    def test_nonchunk_response_and_double_install_fail(self):
        from .temporal_ensemble import install_client
        np=self.np
        class Client:
            def infer(self,obs):return {'action':np.array([1.,2.])}
        install_client(Client)
        with self.assertRaises(ValueError):Client().infer({})
        with self.assertRaises(ValueError):install_client(Client)
