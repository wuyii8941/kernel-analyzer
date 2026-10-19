"""Read contracts without silently replacing repeated explicit YAML fields."""
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError


class ContractLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self._checked_mappings = set()

    def flatten_mapping(self, node):
        # Check explicit fields before YAML merge expansion. Existing examples
        # intentionally inherit an input and override selected fields with <<.
        # Aliases can revisit an already expanded mapping, so inspect it once.
        if node not in self._checked_mappings:
            self._checked_mappings.add(node)
            seen = {}
            for key_node, _ in node.value:
                key = ("<<" if key_node.tag == "tag:yaml.org,2002:merge"
                       else self.construct_object(key_node, deep=True))
                try:
                    if key in seen:
                        raise ConstructorError("while reading a contract mapping", node.start_mark,
                                               f"duplicate field {key!r}", key_node.start_mark)
                    seen[key] = key_node
                except TypeError as exc:
                    raise ConstructorError("while reading a contract mapping", node.start_mark,
                                           "unhashable mapping key", key_node.start_mark) from exc
        super().flatten_mapping(node)


def load_contract(path: Path):
    return yaml.load(Path(path).read_text(), Loader=ContractLoader)
