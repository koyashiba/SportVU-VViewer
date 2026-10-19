import numpy as np

from btrv.core.types import Detection
from btrv.team.color import TeamAssigner, TeamConfig, jersey_feature, kmeans2
from btrv.track.base import TrackerConfig, create_tracker


def _frame_with_players(colors, boxes):
    img = np.full((240, 320, 3), (120, 140, 150), dtype=np.uint8)
    for c, (x1, y1, x2, y2) in zip(colors, boxes):
        img[y1:y2, x1:x2] = c
    return img


def test_jersey_feature_and_two_team_clustering():
    red, blue = (30, 30, 220), (220, 150, 30)
    boxes = [(10, 20, 40, 100), (60, 20, 90, 100), (110, 20, 140, 100), (160, 20, 190, 100)]
    ta = TeamAssigner(TeamConfig(min_observations=2))
    for f in range(4):
        img = _frame_with_players([red, blue, red, blue], boxes)
        for t, b in enumerate(boxes):
            ta.observe(img, b, track_id=t, frame_index=f, ordinal=t)
    teams = ta.fit()
    assert teams[0].team_id == teams[2].team_id and teams[1].team_id == teams[3].team_id
    assert teams[0].team_id != teams[1].team_id
    assert all(a.confidence > 0.6 for a in teams.values())
    assert ta.separation() > 20


def test_ambiguous_track_is_marked_undecided():
    red, blue, mid = (30, 30, 220), (220, 150, 30), (125, 90, 125)
    boxes = [(10, 20, 40, 100), (60, 20, 90, 100), (110, 20, 140, 100), (160, 20, 190, 100), (210, 20, 240, 100)]
    ta = TeamAssigner(TeamConfig())
    for f in range(3):
        img = _frame_with_players([red, blue, red, blue, mid], boxes)
        for t, b in enumerate(boxes):
            ta.observe(img, b, track_id=t)
    teams = ta.fit()
    # the purple-grey track is far from both centroids relative to its nearest one: lower confidence than real jerseys
    assert teams[4].confidence < min(teams[t].confidence for t in range(4))
    # a feature exactly between the two centroids is undecided
    mid = ta.centroids.mean(axis=0)
    a = ta._assign_feature(mid, 3)
    assert a.team_id is None and a.confidence < 0.3


def test_feature_none_for_tiny_box():
    img = np.zeros((50, 50, 3), np.uint8)
    assert jersey_feature(img, (0, 0, 1, 1)) is None


def test_kmeans2_separates():
    X = np.concatenate([np.random.default_rng(0).normal(0, 1, (20, 3)), np.random.default_rng(1).normal(10, 1, (20, 3))])
    labels, C = kmeans2(X)
    assert labels[:20].min() == labels[:20].max() and labels[20:].min() == labels[20:].max() and labels[0] != labels[-1]


def _moving_dets(frame, n=4, occlude_frames=()):
    dets = []
    for k in range(n):
        if k == 1 and frame in occlude_frames:
            continue
        x = 50 + 120 * k + 3 * frame
        dets.append(Detection((x, 100.0, x + 30, 180.0), "players", 0, 0.9))
    return dets


def test_bytetrack_keeps_ids_and_survives_short_occlusion():
    tr = create_tracker(TrackerConfig(lost_track_buffer=30, frame_rate=25))
    ids_per_frame = []
    for f in range(30):
        out = tr.update(_moving_dets(f, occlude_frames=range(10, 14)), f, f / 25.0)
        ids_per_frame.append(sorted(t.track_id for t in out if t.track_id is not None))
    assert len(ids_per_frame[5]) == 4
    # the same ids persist from frame 5 to frame 29 (including after the occlusion gap)
    assert ids_per_frame[5] == ids_per_frame[29]
    assert len(ids_per_frame[12]) == 3  # occluded player absent, no ghost
    tr.reset()
    out = tr.update(_moving_dets(0), 0, 0.0)
    assert len(out) == 4


def test_bytetrack_low_confidence_cannot_spawn_but_can_continue():
    tr = create_tracker(TrackerConfig(track_activation_threshold=0.5))
    low = [Detection((10.0, 10.0, 40.0, 90.0), "players", 0, 0.35)]
    out = tr.update(low, 0, 0.0)
    assert out[0].track_id is None
    high = [Detection((12.0, 10.0, 42.0, 90.0), "players", 0, 0.9)]
    out = tr.update(high, 1, 0.04)
    assert out[0].track_id is None  # tentative on first sighting (library semantics)
    out = tr.update([Detection((14.0, 10.0, 44.0, 90.0), "players", 0, 0.9)], 2, 0.08)
    assert out[0].track_id is not None
    tid = out[0].track_id
    out = tr.update([Detection((16.0, 10.0, 46.0, 90.0), "players", 0, 0.35)], 3, 0.12)
    assert out[0].track_id == tid  # low-confidence detection continues an existing track
