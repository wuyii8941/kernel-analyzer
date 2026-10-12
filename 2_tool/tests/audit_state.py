"""State shared between the test process and call files that ``kernel_analyzer.measure`` loads as fresh module
instances (``measure.resolve``): counters and recordings live here, imported by name, so both see the same objects."""
CALLS = {"row_sum_counted": 0}
SWITCHES = []
