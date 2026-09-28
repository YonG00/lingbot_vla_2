import argparse
from .lingbot_vla_v2_policy import LingbotVLAv2Server, str2bool
from .websocket_batch_policy_server import DynamicBatchWebsocketPolicyServer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--port", type=int, default=9330)
    p.add_argument("--use_length", type=int, default=25)
    p.add_argument("--max_batch", type=int, default=4)
    p.add_argument("--batch_wait_ms", type=float, default=20)
    p.add_argument("--use_bf16", type=str2bool, default=True)
    p.add_argument("--use_fp32", type=str2bool, default=False)
    p.add_argument("--use_compile", type=str2bool, default=False)
    a = p.parse_args()
    policy = LingbotVLAv2Server(a.model_path, use_length=a.use_length, chunk_ret=True, use_bf16=a.use_bf16, use_fp32=a.use_fp32, use_compile=a.use_compile)
    DynamicBatchWebsocketPolicyServer(policy, port=a.port, max_batch=a.max_batch, batch_wait_ms=a.batch_wait_ms).serve_forever()

if __name__ == "__main__": main()
