import config
from conftest import DAY0, blob, png_bytes


def test_stale_rainy_repeat_detected_dry_repeat_allowed(store):
    rainy, dry = png_bytes(blob()), png_bytes(blob(value=0.0))
    store.save(DAY0, rainy)
    assert store.is_stale_repeat(DAY0 + config.STEP, rainy)
    assert not store.is_stale_repeat(DAY0 + config.STEP, png_bytes(blob(shift=3)))
    store.save(DAY0 + 10 * config.STEP, dry)
    assert not store.is_stale_repeat(DAY0 + 11 * config.STEP, dry)
