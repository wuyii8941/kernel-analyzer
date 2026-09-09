import hashlib
import json
from pathlib import Path
import tempfile

import pytest
from scripts.snapshot_frozen_sources import preserve


def test_preserves_matching_sources_and_never_overwrites():
    with tempfile.TemporaryDirectory(dir='/data1/tzh',prefix='.snapshot-test-') as directory:
        root=Path(directory)
        code=root/'code.py'
        code.write_text('value = 1\n')
        protocol=root/'protocol.json'
        protocol.write_text(json.dumps(dict(source_sha256={str(code):hashlib.sha256(code.read_bytes()).hexdigest()})))
        output=root/'snapshot.json'
        assert preserve(protocol,output)==dict(preserved_python_sources=1)
        assert json.loads(output.read_text())[str(code)]==code.read_text()
        with pytest.raises(ValueError,match='new output'):
            preserve(protocol,output)
        code.write_text('value = 2\n')
        with pytest.raises(ValueError,match='changed'):
            preserve(protocol,root/'wrong.json')
        assert not (root/'wrong.json').exists()
