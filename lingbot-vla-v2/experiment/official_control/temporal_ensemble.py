"""Opt-in client-side overlap averaging; one fresh chunk per executed action."""
from collections import deque
import math
import numbers

class TemporalEnsemble:
    def __init__(self,horizon=8,decay=0.1):
        if type(horizon)is not int or horizon<1:raise ValueError('positive integer ensemble horizon required')
        if isinstance(decay,bool)or not isinstance(decay,numbers.Real)or not math.isfinite(decay)or decay<0:raise ValueError('finite nonnegative decay required')
        self.horizon=horizon;self.decay=float(decay);self.reset()
    def reset(self):
        self.history=deque(maxlen=self.horizon);self.width=None;self.steps=0;self.last_count=0
    def push(self,chunk):
        if not chunk or not chunk[0]:raise ValueError('nonempty H by D action chunk required')
        width=len(chunk[0])
        if self.width is not None and width!=self.width:raise ValueError('action dimension changed without reset')
        rows=[]
        for row in chunk:
            if len(row)!=width or any(isinstance(x,bool)or not isinstance(x,numbers.Real)or not math.isfinite(x)for x in row):raise ValueError('ragged or nonfinite action chunk')
            rows.append(tuple(float(x)for x in row))
        self.width=width;self.history.append(rows);self.steps+=1
        total=[0.]*width;denom=0.;count=0
        for age,prediction in enumerate(reversed(self.history)):
            if age>=len(prediction):continue
            weight=math.exp(-self.decay*age)
            for d,value in enumerate(prediction[age]):total[d]+=weight*value
            denom+=weight;count+=1
        self.last_count=count
        return [value/denom for value in total]

def install_client(client_type,horizon=8,decay=0.1,audit_path=None):
    """Client histories remain per instance and reset at every episode boundary."""
    if getattr(client_type,'_temporal_ensemble_installed',False):raise ValueError('temporal ensemble already installed')
    TemporalEnsemble(horizon,decay)
    original=client_type.infer
    def infer(client,observation):
        import numpy as np
        state=getattr(client,'_temporal_action_state',None)
        if state is None:state=TemporalEnsemble(horizon,decay);client._temporal_action_state=state
        if observation.get('reset'):state.reset()
        response=original(client,observation)
        if observation.get('reset'):return response
        array=np.asarray(response['action'])
        if array.ndim!=2:raise ValueError('temporal ensemble requires H by D returned chunks')
        merged=state.push(array.tolist())
        result=dict(response);result['action']=np.asarray(merged,dtype=array.dtype)
        if audit_path:
            import json
            from pathlib import Path
            target=Path(audit_path);temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(enabled=True,ensemble_horizon=horizon,decay=decay,prefer='newer',
                replan_every_actions=1,executed_actions_per_response=1,source_chunk_length=int(array.shape[0]),
                action_dimension=int(array.shape[1]),requests=state.steps,active_predictions=state.last_count)))
            temporary.replace(target)
        return result
    client_type.infer=infer;client_type._temporal_ensemble_installed=True
