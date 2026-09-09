"""Pair two declared calls by live tensor identity, not invocation proximity.

The integration must check both callsites and supply the corresponding contract
ID. This helper neither changes outputs nor establishes binary identity.
"""


class PendingConvolutions:
    def __init__(self):
        self.pending = {}

    @staticmethod
    def key(value):
        return (str(value.device), value.untyped_storage().data_ptr(),
                value.storage_offset(), tuple(value.shape), tuple(value.stride()), value.dtype)

    def record(self, contract_id, inputs, weight, output):
        import torch
        if (not contract_id or inputs.dtype != torch.bfloat16
                or weight.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                or inputs.ndim != 3 or tuple(weight.shape) != (1536, 1, 4)
                or inputs.shape[1] != 1536
                or output.shape != (inputs.shape[0], 1536, inputs.shape[2]+3)
                or inputs.device != weight.device or output.device != inputs.device):
            raise ValueError('Convolution tensor contract differs')
        key = self.key(output)
        if key in self.pending:
            raise ValueError('Unconsumed convolution output would be overwritten')
        self.pending[key] = dict(contract_id=contract_id, inputs=inputs.detach().clone(),
                                 weight=weight.detach().clone(), output=output.detach().clone(),
                                 retained_output=output)

    def take_before_bias(self, contract_id, buffer, bias):
        import torch
        key = self.key(buffer)
        record = self.pending.get(key)
        if record is None or record['contract_id'] != contract_id:
            raise ValueError('No matching convolution for this bias call')
        if not torch.equal(record['output'], buffer):
            raise ValueError('Convolution output changed before bias invocation')
        if (bias.dtype != torch.bfloat16 or bias.shape != (1536,)
                or bias.device != buffer.device or not torch.isfinite(bias).all()):
            raise ValueError('Invalid channel bias')
        del self.pending[key]
        return dict(inputs=record['inputs'], weight=record['weight'],
                    convolution_output=record['output'], bias=bias.detach().clone())

    def require_empty(self):
        if self.pending:
            raise ValueError('Observed convolution without matching bias call')
