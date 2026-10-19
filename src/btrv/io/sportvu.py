"""SportVU export/import following linouk23/NBA-Player-Movements.

Verified structure of that repository's JSON (Game.py / Event.py / Moment.py /
Player.py / Ball.py, fetched 2026-10-19):

    {"gameid": str, "gamedate": str,
     "events": [{"eventId": str,
                 "visitor": {"name", "teamid", "abbreviation", "players": [{"lastname","firstname","playerid","jersey","position"}]},
                 "home":    {...same...},
                 "moments": [[quarter, unix_time_ms, game_clock_s, shot_clock_s, null,
                              [[team_id, player_id, x, y, z], ...]], ...]}]}

* ``moment[0]`` quarter, ``moment[1]`` wall-clock ms, ``moment[2]`` game clock,
  ``moment[3]`` shot clock, ``moment[4]`` unused (null), ``moment[5]`` entities.
* The first entity is the ball ``[-1, -1, x, y, z]``; ``Ball.py`` reads x, y and
  uses z (index 4) as the drawn radius.  Players follow: ``Player.py`` reads
  team id, player id, x, y.
* The court is drawn with ``extent=[0, 94, 50, 0]`` i.e. x in [0, 94] along the
  length, y in [0, 50] with y = 0 at the top of the drawing.  That is exactly
  btrv's ``sportvu_feet_v1`` convention, so no conversion is applied.
* ``Team.py`` colours teams by looking up the team id in a table of the 30 NBA
  ids; unknown ids raise KeyError there, so the placeholder team ids below are
  two real NBA ids purely for compatibility (documented in docs/sportvu.md).

Placeholders (documented, see docs/sportvu.md):
  quarter = 0, game_clock = 0.0, shot_clock = 0.0, unix_time_ms = EPOCH_MS + round(timestamp_s * 1000)
  with EPOCH_MS = 0 by default, ball z default when the external ball has no height = BALL_Z_DEFAULT_FT.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from btrv.core.types import COURT_CONVENTION
from btrv.storage.boxes_table import BoxesTable
from .ball import BallRecord
from .errors import ExportValidationError, ImportValidationError

PLACEHOLDER_QUARTER = 0
PLACEHOLDER_GAME_CLOCK_S = 0.0
PLACEHOLDER_SHOT_CLOCK_S = 0.0
BALL_Z_DEFAULT_FT = 0.0
BALL_TEAM_ID = -1
BALL_PLAYER_ID = -1
# Real NBA ids used only so the reference viewer's colour table resolves (ATL, BOS).
PLACEHOLDER_TEAM_IDS = (1610612737, 1610612738)
DEBUG_UNKNOWN_TEAM_ID = 0  # player-only debug export: team not decided
DEBUG_UNTRACKED_PLAYER_ID = -2  # player-only debug export: box without a track id


@dataclass
class SportVUExportOptions:
    game_id: str = "btrv"
    game_date: str = ""
    event_id: str = "1"
    team_ids: tuple[int, int] = PLACEHOLDER_TEAM_IDS
    team_names: tuple[str, str] = ("BTRV_TEAM_0", "BTRV_TEAM_1")
    epoch_ms: int = 0
    ball_z_default_ft: float = BALL_Z_DEFAULT_FT
    quarter: int = PLACEHOLDER_QUARTER
    game_clock_s: float = PLACEHOLDER_GAME_CLOCK_S
    shot_clock_s: float = PLACEHOLDER_SHOT_CLOCK_S
    require_team: bool = True  # conforming export drops boxes without a team (documented)
    frame_start: Optional[int] = None
    frame_stop: Optional[int] = None


def _team_block(team_id: int, name: str, player_ids: Iterable[int]) -> dict[str, Any]:
    return {
        "name": name,
        "teamid": int(team_id),
        "abbreviation": name[:3].upper(),
        "players": [{"lastname": f"T{pid}", "firstname": "Track", "playerid": int(pid), "jersey": str(pid), "position": ""} for pid in sorted(set(player_ids))],
    }


def _frames_in_range(table: BoxesTable, opts: SportVUExportOptions) -> np.ndarray:
    fi = table.frame_indices
    if opts.frame_start is not None:
        fi = fi[fi >= opts.frame_start]
    if opts.frame_stop is not None:
        fi = fi[fi < opts.frame_stop]
    return fi


def export_sportvu(table: BoxesTable, ball: Optional[list[BallRecord]], opts: SportVUExportOptions = SportVUExportOptions(), player_only_debug: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the SportVU document and a metadata sidecar.

    Conforming export requires ball coordinates for every exported frame.  The
    player-only debug export (``player_only_debug=True``) omits the ball entry
    entirely and is NOT SportVU conforming; it never synthesises a ball.
    """
    frames = _frames_in_range(table, opts)
    if frames.size == 0:
        raise ExportValidationError("nothing to export: the run has no frames in the requested range")
    ball_by_frame: dict[int, BallRecord] = {}
    if not player_only_debug:
        if not ball:
            raise ExportValidationError(
                "Conforming SportVU export needs externally supplied ball coordinates (btrv.ball records in court feet) "
                "because btrv does not track the ball. Import a ball file (File > Import ball coordinates, or POST /ball), "
                "or use the clearly non-standard player-only debug export instead."
            )
        ball_by_frame = {b.frame_index: b for b in ball}
        missing = [int(f) for f in frames if int(f) not in ball_by_frame]
        if missing:
            preview = ", ".join(str(m) for m in missing[:10]) + (" ..." if len(missing) > 10 else "")
            raise ExportValidationError(
                f"ball coordinates are missing for {len(missing)} of {frames.size} frames (e.g. frames {preview}). "
                "Provide a ball record for every exported frame, restrict the export range, or use the player-only debug export."
            )
    team_ids = opts.team_ids
    moments: list[list[Any]] = []
    players_by_team: dict[int, set[int]] = {team_ids[0]: set(), team_ids[1]: set()}
    stats = {"frames": 0, "entities": 0, "dropped_no_team": 0, "dropped_no_track": 0, "dropped_invalid_court": 0, "frames_without_calibration": 0}
    for f in frames:
        f = int(f)
        fb = table.frame(f)
        if fb is None:
            continue
        entities: list[list[Any]] = []
        if not player_only_debug:
            b = ball_by_frame[f]
            z = opts.ball_z_default_ft if b.z is None else float(b.z)
            entities.append([BALL_TEAM_ID, BALL_PLAYER_ID, float(b.x), float(b.y), z])
        if fb.calibration is not None and not fb.calibration.get("valid", True):
            stats["frames_without_calibration"] += 1
        rows = []
        for box in fb.boxes:
            if box.court is None or not box.court.valid:
                stats["dropped_invalid_court"] += 1
                continue
            if box.track_id is None:
                if not player_only_debug:
                    stats["dropped_no_track"] += 1
                    continue
                pid = DEBUG_UNTRACKED_PLAYER_ID
            else:
                pid = int(box.track_id)
            if box.team is None or box.team.team_id is None:
                if not player_only_debug and opts.require_team:
                    stats["dropped_no_team"] += 1
                    continue
                tid = DEBUG_UNKNOWN_TEAM_ID
            else:
                tid = team_ids[int(box.team.team_id)]
                players_by_team.setdefault(tid, set()).add(pid)
            rows.append([tid, pid, float(box.court.x), float(box.court.y), 0.0])
        # ball first, then teams in a stable order (team 0 block, team 1 block), players by id
        rows.sort(key=lambda r: ((0 if r[0] == team_ids[0] else 1 if r[0] == team_ids[1] else 2), r[1]))
        entities.extend(rows)
        unix_ms = int(opts.epoch_ms + round(fb.timestamp_s * 1000.0))
        moments.append([opts.quarter, unix_ms, opts.game_clock_s, opts.shot_clock_s, None, entities])
        stats["frames"] += 1
        stats["entities"] += len(entities)
    event = {
        "eventId": opts.event_id,
        "visitor": _team_block(team_ids[1], opts.team_names[1], players_by_team.get(team_ids[1], set())),
        "home": _team_block(team_ids[0], opts.team_names[0], players_by_team.get(team_ids[0], set())),
        "moments": moments,
    }
    doc = {"gameid": opts.game_id, "gamedate": opts.game_date, "events": [event]}
    meta = {
        "schema": "btrv.sportvu_export_meta",
        "schema_version": "1.0",
        "conforming": not player_only_debug,
        "kind": "player_only_debug_NONSTANDARD" if player_only_debug else "sportvu_conforming",
        "source_id": table.source_id,
        "coordinate_convention": COURT_CONVENTION,
        "reference_viewer": "https://github.com/linouk23/NBA-Player-Movements",
        "moment_layout": "[quarter, unix_time_ms, game_clock_s, shot_clock_s, null, [[team_id, player_id, x_ft, y_ft, z_ft], ...]]",
        "ball": "absent (NONSTANDARD)" if player_only_debug else "first entity, [-1, -1, x, y, z] from external btrv.ball records",
        "placeholders": {
            "quarter": opts.quarter,
            "game_clock_s": opts.game_clock_s,
            "shot_clock_s": opts.shot_clock_s,
            "unix_time_ms": f"epoch_ms ({opts.epoch_ms}) + round(source timestamp_s * 1000)",
            "ball_z_ft_when_unknown": opts.ball_z_default_ft,
            "player_z_ft": 0.0,
            "team_ids": {"team_0": team_ids[0], "team_1": team_ids[1], "note": "placeholder NBA ids so the reference viewer's colour table resolves"},
            "player_id": "btrv track id",
            "debug_unknown_team_id": DEBUG_UNKNOWN_TEAM_ID if player_only_debug else None,
            "debug_untracked_player_id": DEBUG_UNTRACKED_PLAYER_ID if player_only_debug else None,
        },
        "stats": stats,
        "frame_range": [int(frames[0]), int(frames[-1])],
    }
    return doc, meta


def write_sportvu(path: str | os.PathLike, doc: dict[str, Any], meta: dict[str, Any]) -> tuple[Path, Path]:
    path = Path(path)
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    meta_path = path.with_name(path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path, meta_path


# ---------------------------------------------------------------------------
# Import for the top-down court view
# ---------------------------------------------------------------------------
@dataclass
class SportVUMoment:
    quarter: Any
    unix_ms: int
    game_clock: Any
    shot_clock: Any
    ball: Optional[tuple[float, float, float]]
    players: list[tuple[int, int, float, float, float]]  # team_id, player_id, x, y, z


@dataclass
class SportVUTrack:
    moments: list[SportVUMoment]
    home_team_id: Optional[int] = None
    visitor_team_id: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

    def __len__(self) -> int:
        return len(self.moments)

    def times_ms(self) -> np.ndarray:
        return np.asarray([m.unix_ms for m in self.moments], dtype=np.int64)

    def moment_for_time(self, t_ms: int, tol_ms: Optional[int] = None) -> Optional[SportVUMoment]:
        """Latest moment with unix_ms <= t_ms (optionally within tol_ms)."""
        if not self.moments:
            return None
        t = self.times_ms()
        i = int(np.searchsorted(t, t_ms, side="right") - 1)
        if i < 0:
            return None
        if tol_ms is not None and t_ms - int(t[i]) > tol_ms:
            return None
        return self.moments[i]


def read_sportvu(path: str | os.PathLike, event_index: int = 0) -> SportVUTrack:
    path = Path(path)
    if not path.exists():
        raise ImportValidationError(f"SportVU file not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ImportValidationError(f"{path.name}: invalid JSON: {e.msg} (line {e.lineno})") from e
    events = doc.get("events") if isinstance(doc, dict) else None
    if not isinstance(events, list) or not events:
        raise ImportValidationError(f"{path.name}: expected a top-level 'events' list with at least one event")
    if not (0 <= event_index < len(events)):
        raise ImportValidationError(f"{path.name}: event index {event_index} out of range (file has {len(events)} events)")
    ev = events[event_index]
    raw = ev.get("moments")
    if not isinstance(raw, list):
        raise ImportValidationError(f"{path.name}: event {event_index} has no 'moments' list")
    moments: list[SportVUMoment] = []
    for i, m in enumerate(raw):
        if not isinstance(m, list) or len(m) < 6 or not isinstance(m[5], list):
            raise ImportValidationError(f"{path.name}: moment {i} must be [quarter, unix_ms, game_clock, shot_clock, null, entities]")
        ents = m[5]
        ball = None
        players: list[tuple[int, int, float, float, float]] = []
        for j, e in enumerate(ents):
            if not isinstance(e, list) or len(e) < 4:
                raise ImportValidationError(f"{path.name}: moment {i} entity {j} must be [team_id, player_id, x, y, z]")
            z = float(e[4]) if len(e) > 4 and e[4] is not None else 0.0
            if int(e[0]) == BALL_TEAM_ID and j == 0:
                ball = (float(e[2]), float(e[3]), z)
            else:
                players.append((int(e[0]), int(e[1]), float(e[2]), float(e[3]), z))
        moments.append(SportVUMoment(m[0], int(m[1]), m[2], m[3], ball, players))
    meta = {}
    mp = path.with_name(path.stem + ".meta.json")
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return SportVUTrack(moments, (ev.get("home") or {}).get("teamid"), (ev.get("visitor") or {}).get("teamid"), meta, str(path))
