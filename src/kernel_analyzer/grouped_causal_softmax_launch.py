"""Validate resolved launch geometry, not guesses from a kernel name.

The capture integration must supply the actual selected XBLOCK and resolved
grid. A set of possible autotuning configurations is not this evidence.
"""


def validate_launch(contract, *, xnumel, r0_numel, xblock, grid):
    for name, value in [('xnumel', xnumel), ('r0_numel', r0_numel), ('XBLOCK', xblock)]:
        if type(value) is not int or value <= 0:
            raise ValueError('Positive integer launch argument required: ' + name)
    if xnumel != contract['rows'] or r0_numel != contract['width']:
        raise ValueError('Runtime dimensions differ from checked source')
    if xblock & (xblock - 1):
        raise ValueError('Power-of-two XBLOCK required')
    if (not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3
            or any(type(x) is not int or x <= 0 for x in grid)):
        raise ValueError('Resolved positive integer grid required')
    grid = tuple(grid) + (1,) * (3-len(grid))
    if grid[1:] != (1, 1):
        raise ValueError('Duplicate row launches in other grid dimensions')
    if grid[0] != (xnumel + xblock - 1) // xblock:
        raise ValueError('Launch omits rows or starts extra blocks')
    masked = contract.get('row_bounds_masked')
    if type(masked) is not bool:
        raise ValueError('Missing checked row mask contract')
    if not masked and grid[0] * xblock != xnumel:
        raise ValueError('Unmasked launch would access invalid rows')
    return dict(grid=list(grid), xblock=xblock, rows=xnumel, width=r0_numel,
                row_bounds_masked=masked, status='RESOLVED_GEOMETRY_CHECKED',
                execution_identity_proved=False)
