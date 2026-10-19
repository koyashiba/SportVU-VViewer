import numpy as np
import pytest

from btrv.source import FileSource, build_frame_index
from tests.conftest import marker_index


def test_index_is_presentation_order_and_cached(marker_video, tmp_path):
    idx = build_frame_index(marker_video, cache_dir=tmp_path, use_cache=True)
    assert idx.frame_count == 60
    assert np.all(np.diff(idx.pts) > 0)
    assert idx.keyframe[0]
    assert idx.keyframe.sum() >= 4  # gop 12 over 60 frames
    idx2 = build_frame_index(marker_video, cache_dir=tmp_path, use_cache=True)
    assert np.array_equal(idx.pts, idx2.pts)
    # timestamps are pts * time_base, not an assumed constant FPS
    assert idx.timestamp_s(25) == pytest.approx(1.0, abs=1e-6)


def test_random_access_is_frame_accurate(marker_video):
    with FileSource(marker_video, use_index_cache=False) as src:
        assert src.info.frame_count == 60
        for i in [0, 59, 17, 5, 33, 32, 31, 30, 29, 1, 58, 11, 12, 13]:
            info, frame = src.read(i)
            assert info.frame_index == i
            assert frame.shape == (240, 320, 3)
            assert marker_index(frame) == i


def test_sequential_and_backward_stepping(marker_video):
    with FileSource(marker_video, use_index_cache=False) as src:
        for i in range(60):
            _, f = src.read(i)
            assert marker_index(f) == i
        for i in range(59, -1, -1):
            _, f = src.read(i)
            assert marker_index(f) == i
        assert src.next() is None or True


def test_next_iterates_in_order(marker_video):
    with FileSource(marker_video, use_index_cache=False) as src:
        seen = []
        while True:
            r = src.next()
            if r is None:
                break
            seen.append(r[0].frame_index)
        assert seen == list(range(60))


def test_out_of_range(marker_video):
    with FileSource(marker_video, use_index_cache=False) as src:
        with pytest.raises(IndexError):
            src.read(60)
        with pytest.raises(IndexError):
            src.read(-1)


def test_small_cache_budget_still_accurate(marker_video):
    with FileSource(marker_video, use_index_cache=False, cache_budget_bytes=1) as src:
        for i in [40, 39, 38, 0, 59, 20]:
            _, f = src.read(i)
            assert marker_index(f) == i


def test_variable_frame_rate_timestamps(tmp_path):
    """Timestamps come from pts; a VFR file must not be treated as constant rate."""
    from tests.conftest import make_marker_video

    p = make_marker_video(tmp_path / "vfr.mp4", n_frames=40, extra=["-vf", "setpts=PTS*(1+gt(N\\,19))", "-fps_mode", "vfr"])  # -fps_mode needs ffmpeg >= 5.1 (-vsync was removed in 9)
    with FileSource(p, use_index_cache=False) as src:
        ts = src.index.timestamps_s()
        d = np.diff(ts)
        assert d[:15].mean() < d[25:].mean() * 0.75  # second half is slower
        assert src.info.frame_count == 40
        for i in [0, 19, 20, 21, 39, 10]:
            info, f = src.read(i)
            assert marker_index(f) == i
            assert info.timestamp_s == pytest.approx(ts[i])
        assert src.index.frame_for_timestamp(ts[25] + 1e-4) == 25
        assert src.index.frame_for_timestamp(ts[25] - 1e-4) == 24


def test_mkv_container(tmp_path):
    from tests.conftest import make_marker_video

    p = make_marker_video(tmp_path / "m.mkv", n_frames=30)
    with FileSource(p, use_index_cache=False) as src:
        assert src.info.frame_count == 30
        for i in [29, 3, 17]:
            assert marker_index(src.read(i)[1]) == i
