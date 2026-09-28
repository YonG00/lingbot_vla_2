import asyncio, http, logging, time, traceback
import torch, websockets.frames
import websockets.asyncio.server as _server
from .msgpack_numpy import Packer, unpackb
from .websocket_policy_server import _health_check, _optional_float_env

log = logging.getLogger(__name__)

class DynamicBatchWebsocketPolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=8006, metadata=None, max_batch=4, batch_wait_ms=20):
        if not getattr(policy, "chunk_ret", False): raise ValueError("Dynamic batch Quick Eval requires chunk_ret=True")
        self.policy, self.host, self.port, self.metadata = policy, host, port, metadata or {}
        self.max_batch, self.batch_wait = max(1, max_batch), batch_wait_ms / 1000.0
        self.queue = asyncio.Queue()

    def serve_forever(self): asyncio.run(self.run())

    async def run(self):
        worker = asyncio.create_task(self._batch_loop())
        try:
            async with _server.serve(self._handler, self.host, self.port, compression=None, max_size=None,
                ping_interval=_optional_float_env("WEBSOCKET_PING_INTERVAL"),
                ping_timeout=_optional_float_env("WEBSOCKET_PING_TIMEOUT"), process_request=_health_check) as server:
                print(f"Dynamic batch server: port={self.port}, max_batch={self.max_batch}, wait={self.batch_wait*1000:.0f}ms")
                await server.serve_forever()
        finally:
            worker.cancel()

    async def _handler(self, websocket):
        packer = Packer()
        await websocket.send(packer.pack(self.metadata))
        while True:
            try:
                t0 = time.monotonic()
                obs = unpackb(await websocket.recv())
                fut = asyncio.get_running_loop().create_future()
                await self.queue.put((obs, fut))
                action, infer_ms, batch_size = await fut
                action["server_timing"] = {"infer_ms": infer_ms, "batch_size": batch_size, "total_ms": (time.monotonic()-t0)*1000}
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed: break
            except Exception:
                try: await websocket.send(traceback.format_exc())
                except Exception: pass
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Internal server error")
                raise

    async def _batch_loop(self):
        while True:
            first = await self.queue.get()
            items = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_wait
            while len(items) < self.max_batch:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0: break
                try: items.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError: break

            resets, normal = [], []
            for item in items:
                (resets if item[0].get("reset", False) else normal).append(item)

            for obs, fut in resets:
                try:
                    t = time.monotonic(); out = self.policy.infer(obs)
                    fut.set_result((out, (time.monotonic()-t)*1000, 1))
                except Exception as e: fut.set_exception(e)

            if normal:
                obs_list = [x[0] for x in normal]
                try:
                    t = time.monotonic(); outputs = self._infer_auto(obs_list)
                    infer_ms = (time.monotonic()-t)*1000
                    for (_, fut), out in zip(normal, outputs): fut.set_result((out, infer_ms, len(obs_list)))
                except Exception as e:
                    for _, fut in normal: fut.set_exception(e)

    def _infer_auto(self, observations):
        try:
            out = self.policy.infer({"batch": observations})
            n = len(observations)
            return [{k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n else v) for k, v in out.items()} for i in range(n)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n = len(observations)
            if n == 1: raise
            mid = n // 2
            print(f"[batch OOM] batch={n} -> split {mid}+{n-mid}")
            return self._infer_auto(observations[:mid]) + self._infer_auto(observations[mid:])
