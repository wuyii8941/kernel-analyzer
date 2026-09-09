"""Finite floating-point Top-k semantic checks and a declared stable-sort variant.

Equal-score index choices are not treated as numerical bugs. This is not a
training equivalence certificate or a claim that sorting is absolute truth.
"""
import torch


def stable_selection(scores, k, dim=-1, largest=True):
    if not scores.is_floating_point() or scores.ndim==0:
        raise ValueError('Non-scalar floating scores required')
    if not -scores.ndim<=dim<scores.ndim or not 0<k<=scores.shape[dim]:
        raise ValueError('Invalid selection dimension or k')
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite selection semantics not declared')
    # Stable tie choice is an explicit variant, not a requirement of Top-k.
    values, indices=torch.sort(scores,dim=dim,descending=largest,stable=True)
    chosen=values.narrow(dim,0,k)
    boundary_ties=False
    if k<scores.shape[dim]:
        boundary_ties=bool((values.select(dim,k-1)==values.select(dim,k)).any())
    return chosen,indices.narrow(dim,0,k),dict(
        boundary_ties=boundary_ties,tie_policy='STABLE_INPUT_INDEX_ORDER',
        reference_is_absolute_truth=False)


def validate_selection(scores, values, indices, k, dim=-1, largest=True, sorted_output=True):
    expected,_,diagnostic=stable_selection(scores,k,dim,largest)
    shape=list(scores.shape);shape[dim]=k
    if (list(values.shape)!=shape or list(indices.shape)!=shape
            or values.dtype!=scores.dtype or indices.dtype!=torch.int64
            or values.device!=scores.device or indices.device!=scores.device):
        raise ValueError('Selection shape, dtype or device differs')
    if ((indices<0)|(indices>=scores.shape[dim])).any():
        raise ValueError('Selection index out of bounds')
    ordered_indices=torch.sort(indices,dim=dim).values
    if k>1 and (ordered_indices.narrow(dim,1,k-1)==ordered_indices.narrow(dim,0,k-1)).any():
        raise ValueError('Selection indices repeat')
    if not torch.equal(values,torch.gather(scores,dim,indices)):
        raise ValueError('Selection values do not match indexed input')
    ordered_values=torch.sort(values,dim=dim,descending=largest).values
    if not torch.equal(ordered_values,expected):
        raise ValueError('Selection does not contain the required top values')
    if sorted_output and not torch.equal(values,ordered_values):
        raise ValueError('Selection order differs from sorted contract')
    return dict(**diagnostic,status='FINITE_TOPK_SEMANTICS_CHECKED',
                training_bias_confirmed=False)
