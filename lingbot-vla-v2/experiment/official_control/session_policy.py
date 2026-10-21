"""One set of model weights, independent per-connection deployment state; standard library only."""
from threading import RLock

FIELDS=('global_step','last_action_chunk','last_normalized_action_chunk','robot_config','action_key')
MISSING=object()


class SessionPool:
    def __init__(self,policy):
        if not policy.chunk_ret:raise ValueError('parallel simulation requires chunk_ret=True')
        self.policy=policy;self.lock=RLock()
        self.connections=0;self.requests=0

    def connect(self):
        self.connections+=1
        return SessionPolicy(self,self.connections)


class SessionPolicy:
    def __init__(self,pool,number):
        self.pool=pool;self.number=number;self.state=None;self.transform=MISSING

    def infer(self,observation):
        if observation.get('path_to_pi_model') is not None:
            raise ValueError('session cannot reload shared model weights')
        if self.state is None and not observation.get('reset'):
            raise ValueError('new session must reset before inference')
        pool=self.pool;policy=pool.policy
        with pool.lock:
            old={name:getattr(policy,name,MISSING) for name in FIELDS}
            old_transform=getattr(policy.vla,'feature_transform',MISSING)
            try:
                if self.state is not None:
                    for name,value in self.state.items():assign(policy,name,value)
                    assign(policy.vla,'feature_transform',self.transform)
                result=policy.infer(observation)
                self.state={name:getattr(policy,name,MISSING) for name in FIELDS}
                self.transform=getattr(policy.vla,'feature_transform',MISSING)
                pool.requests+=1
                return result
            finally:
                for name,value in old.items():assign(policy,name,value)
                assign(policy.vla,'feature_transform',old_transform)


def assign(obj,name,value):
    if value is MISSING:
        if hasattr(obj,name):delattr(obj,name)
    else:setattr(obj,name,value)
