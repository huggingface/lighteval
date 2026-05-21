import base64
import json
import pickle
import zlib

import pytest

from lighteval.tasks.tasks.lcb.codegen_metrics import translate_private_test_cases


def _encode_pickled_payload(payload):
    return base64.b64encode(zlib.compress(pickle.dumps(payload))).decode()


def test_translate_private_test_cases_loads_json_string_payload():
    encoded_data = _encode_pickled_payload(json.dumps([{"input": "1", "output": "2"}]))

    assert translate_private_test_cases(encoded_data) == [{"input": "1", "output": "2"}]


def test_translate_private_test_cases_rejects_pickle_globals():
    class GlobalConstructor:
        def __reduce__(self):
            return (str, ("[]",))

    encoded_data = _encode_pickled_payload(GlobalConstructor())

    with pytest.raises(pickle.UnpicklingError):
        translate_private_test_cases(encoded_data)
