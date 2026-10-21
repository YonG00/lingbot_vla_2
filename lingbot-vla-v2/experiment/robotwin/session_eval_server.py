"""FP32 audit server with per-websocket state; model forwards stay serial to preserve B1 RNG."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from deploy import websocket_policy_server as ws
    from experiment.official_control.session_policy import SessionPool
    from experiment.robotwin.precision_eval_server import main as precision_main
    original=ws.WebsocketPolicyServer._handler
    async def handler(server,websocket):
        if not hasattr(server,'_session_pool'):
            server._session_pool=SessionPool(server._policy)
        bound=copy.copy(server)
        bound._policy=server._session_pool.connect()
        path=os.environ.get('LINGBOT_SESSION_AUDIT')
        try:
            return await original(bound,websocket)
        finally:
            if path:
                pool=server._session_pool
                Path(path).write_text(json.dumps(dict(model_loads=1,connections=pool.connections,
                    requests=pool.requests,forward_batch_size=1,session_isolation=True),indent=2))
    ws.WebsocketPolicyServer._handler=handler
    precision_main()

if __name__=='__main__':main()
