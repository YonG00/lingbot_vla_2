# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .lr_scheduler import build_lr_scheduler
from .optimizer import build_muon_optimizer, build_optimizer


def build_flex_shard_dist_muon_optimizer(*args, **kwargs):
    """Load the FlexShard integration only when DistMuon is selected."""
    from .dist_muon import build_flex_shard_dist_muon_optimizer as _build

    return _build(*args, **kwargs)


__all__ = ["build_lr_scheduler", "build_muon_optimizer", "build_optimizer", "build_flex_shard_dist_muon_optimizer"]
