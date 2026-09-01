import pytest

pytest.importorskip("sonar")  # [sonar] extra; skipped on core-only installs

import pytest

from cogito_estella import metrics as m

pytestmark = pytest.mark.integration


def test_roundtrip_english_sentence():
    from cogito_estella.sonar_codec import SonarCodec

    codec = SonarCodec()
    [out] = codec.roundtrip(["The quick brown fox jumps over the lazy dog."], lang="eng_Latn")
    assert m.chrf("The quick brown fox jumps over the lazy dog.", out) > 60
