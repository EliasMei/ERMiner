"""From tianshou"""

import torch, numpy as np
from torch import nn

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch
from tianshou.utils.net.common import Net

ModuleType = Type[nn.Module]


class Rainbow(Net):
    def __init__(self, state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None) -> None:
        super().__init__(state_shape, action_shape, hidden_sizes, norm_layer, activation, device, softmax, concat, num_atoms, dueling_param)
    
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs = self.model(obs)
        bsz = obs.shape[0]
        q = self.Q(obs)
        q = q.view(bsz, -1, self.num_atoms)
        if self.use_dueling:
            v = self.V(obs)
            v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state