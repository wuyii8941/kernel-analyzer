"""One family registry for source discovery and same-input execution.

Adding a reference does not change statistics or imply runtime support.
Only audited, explicit families may be selected; there is no fallback.
"""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ReferenceSpec:
    module: str
    variants: tuple[str, ...]
    case_suffix: str
    takes_variant: bool = False

    @property
    def source_filename(self):
        return self.module + '.py'

    def check_source(self, source, symbol):
        return import_module('kernel_analyzer.' + self.module).check_source(source, symbol)

    def evaluate(self, metadata, candidate, contract, *, variant):
        if variant not in self.variants:
            raise ValueError('Variant is not declared for this reference family')
        module = import_module('kernel_analyzer.' + self.module)
        if self.takes_variant:
            return module.reference(metadata, candidate, contract, variant=variant)
        return module.reference(metadata, candidate, contract)


REFERENCES = {
    'ROW_SQUARE_SUM': ReferenceSpec('row_reduction_reference',
        ('FP32_NATIVE', 'FP32_REVERSE_FEATURE_ORDER', 'FP64_EVALUATION'), '-row-family', True),
    'RMS_BACKWARD': ReferenceSpec('rms_backward_reference', ('FP32_NATIVE',), '-rms-common-input'),
    'RMS_SIMPLE_BACKWARD': ReferenceSpec(
        'rms_simple_backward_registered_reference',
        ('FP32_NATIVE',), '-rms-simple-backward-common-input'),
    'SOFTMAX_BACKWARD': ReferenceSpec('softmax_backward_reference', ('FP32_NATIVE',), '-softmax-common-input'),
    'SILU_BACKWARD': ReferenceSpec('silu_backward_reference', ('FP32_NATIVE',), '-silu-common-input'),
    'SELECTED_SILU_PRODUCT': ReferenceSpec('selected_silu_product_reference',
        ('FP32_NATIVE',), '-selected-silu-common-input'),
    'SOFTPLUS_BIAS_BACKWARD': ReferenceSpec('softplus_bias_backward_reference',
        ('FP32_NATIVE',), '-softplus-bias-common-input'),
    'EMBEDDING_LOOKUP': ReferenceSpec('embedding_lookup_registered_reference',
        ('FP32_NATIVE',), '-embedding-lookup-common-input'),
    'EXPONENTIAL_WEIGHTED_REDUCTION': ReferenceSpec('exponential_weighted_reduction_reference',
        ('FP32_NATIVE', 'FP32_REVERSE_COMPONENT_ORDER', 'FP64_EVALUATION'),
        '-exp-weighted-reduction', True),
    'GROUPED_ROTARY_BACKWARD': ReferenceSpec('rotary_grouped_backward_reference',
        ('FP32_NATIVE',), '-grouped-rotary-common-input'),
    'SCALED_MASKED_SOFTMAX': ReferenceSpec('scaled_masked_softmax_reference',
        ('FP32_NATIVE',), '-scaled-masked-softmax-common-input'),
    'DECAYED_RECURRENCE': ReferenceSpec('decayed_recurrence_reference',
        ('FP32_NATIVE',), '-decayed-recurrence-common-input'),
    'GATED_CONV_GRADIENT': ReferenceSpec('gated_conv_gradient_reference',
        ('FP32_NATIVE',), '-gated-conv-gradient-common-input'),
    'SOFTCAPPED_NLL_BACKWARD': ReferenceSpec('softcapped_nll_registered_reference',
        ('FP32_NATIVE',), '-softcapped-nll-common-input'),
    'SELECTED_NLL_BACKWARD': ReferenceSpec('selected_nll_registered_reference',
        ('FP32_NATIVE',), '-selected-nll-common-input'),
    'GROUPED_CAUSAL_SOFTMAX_PROBABILITY': ReferenceSpec(
        'grouped_causal_softmax_registered_reference',
        ('FP32_NATIVE',), '-grouped-causal-softmax-probability-common-input'),
    'RESIDUAL_RMS_FORWARD_NORMALIZED': ReferenceSpec(
        'residual_rms_forward_registered_reference',
        ('FP32_NATIVE',), '-residual-rms-normalized-common-input'),
    'RMS_FORWARD_NORMALIZED': ReferenceSpec(
        'rms_forward_registered_reference',
        ('FP32_NATIVE',), '-rms-forward-normalized-common-input'),
    'GELU_PRODUCT_BACKWARD': ReferenceSpec(
        'gelu_product_registered_reference',
        ('FP32_NATIVE',), '-gelu-product-common-input'),
    'GELU_FORWARD_PRODUCT': ReferenceSpec(
        'gelu_forward_product_registered_reference',
        ('FP32_NATIVE',), '-gelu-forward-product-common-input'),
    'CHANNEL_BIAS_ADD': ReferenceSpec(
        'channel_bias_registered_reference',
        ('FP32_NATIVE',), '-channel-bias-common-input'),
    'ATTENTION_POSITION_SCALING': ReferenceSpec(
        'attention_position_scaling_registered_reference',
        ('FP32_NATIVE',), '-attention-position-scaling-common-input'),
    'FORWARD_STATE_RECURRENCE_FINAL': ReferenceSpec(
        'forward_state_recurrence_registered_reference',
        ('FP32_NATIVE',), '-forward-state-recurrence-final-common-input'),
}


def get_reference(family):
    try:
        return REFERENCES[family]
    except KeyError as exc:
        raise ValueError('Unsupported reference family: ' + str(family)) from exc
