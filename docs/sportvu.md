# SportVU export and import

## Reference structure (verified)

The exporter follows the JSON read by
[linouk23/NBA-Player-Movements](https://github.com/linouk23/NBA-Player-Movements)
(`Game.py`, `Event.py`, `Moment.py`, `Player.py`, `Ball.py`, `Constant.py`, checked on 2026-10-19):

```json
{"gameid": "...", "gamedate": "...",
 "events": [
   {"eventId": "1",
    "visitor": {"name": "...", "teamid": 1610612738, "abbreviation": "...", "players": [{"lastname": "...", "firstname": "...", "playerid": 1, "jersey": "1", "position": ""}]},
    "home":    {"name": "...", "teamid": 1610612737, "abbreviation": "...", "players": [...]},
    "moments": [
      [quarter, unix_time_ms, game_clock_s, shot_clock_s, null,
       [[-1, -1, ball_x, ball_y, ball_z],
        [team_id, player_id, x, y, 0.0], ...]]
    ]}
 ]}
```

* `Moment.py`: `quarter = moment[0]`, `game_clock = moment[2]`, `shot_clock = moment[3]`,
  `ball = moment[5][0]`, `players = moment[5][1:]`.
* `Ball.py`: `x = ball[2]`, `y = ball[3]`, `radius = ball[4]` (the drawn ball size scales with z).
* `Player.py`: `team = player[0]`, `id = player[1]`, `x = player[2]`, `y = player[3]`.
* `Event.py` draws the court with extent `[0, 94, 50, 0]` → x along the length, y across the width
  with y = 0 at the top; btrv's `sportvu_feet_v1` is identical, so coordinates are written unchanged.
* `Team.py` colours teams through a dict keyed by the 30 NBA team ids (unknown ids raise `KeyError`).
* `Game.py` loads the file with `pandas.read_json` and reads `events[i]`, `home.teamid`, `visitor.teamid`.
  Because of `read_json`, **no extra top-level keys are added**; btrv metadata goes to a sidecar
  `<name>.meta.json`.

## Conforming export (`btrv export-sportvu`, File > Export SportVU)

* One moment per processed frame in the selected range; **ball first** `[-1, -1, x, y, z]`, then the
  players of team 0 and team 1 (sorted by team block, then player id). Referees are never present (the
  pinned model only emits `players`; boxes with other class names are not exported).
* `player_id` = btrv track id; `team_id` = placeholder ids (below); player `z = 0.0`.
* Only boxes with a **valid court point** (valid calibration, on court within the margin), a track id and a
  decided team are exported. Dropped counts are reported in the sidecar `stats`.
* The ball **must** come from external `btrv.ball` records covering every exported frame. Without them the
  export fails with an actionable `ExportValidationError` (which frames are missing, how to restrict
  the range or import a ball file). btrv never synthesises a ball.
* The number of players per moment varies with what was observed (SportVU files always have 10); the
  reference viewer builds its player circles from the first moment, so a varying count can mis-draw there.

### Placeholders (documented values)

| field | value | reason |
|---|---|---|
| `quarter` | `0` | real quarters are 1–4+, so 0 marks "unknown" |
| `game_clock_s` | `0.0` | not derivable from video |
| `shot_clock_s` | `0.0` | not derivable from video |
| `unix_time_ms` | `epoch_ms + round(timestamp_s * 1000)` | derived from the **source frame time**; `epoch_ms` defaults to 0 and is recorded in the sidecar |
| ball `z` when the ball record has `z: null` | `0.0` ft (`--ball-z-default`) | documented default; note the reference viewer draws the ball with radius `z / 7`, so `0` makes it invisible there |
| player `z` | `0.0` | players are on the floor |
| `team_id` team 0 / team 1 | `1610612737` / `1610612738` | real NBA ids (ATL/BOS) only so the reference viewer's colour table resolves; configurable |
| `home` / `visitor` | team 0 / team 1 | arbitrary; both blocks list the track ids seen |
| `gameid`, `gamedate`, `eventId` | `"btrv"`, `""`, `"1"` | configurable |

## Player-only debug export (NONSTANDARD)

`--player-only-debug` / "Player-only debug export": same layout **without the ball entity** (so
`moment[5][0]` is a player, not the ball — the reference viewer will mis-read it, by design it is not
SportVU). Additionally boxes without a team get `team_id = 0` and boxes without a track get
`player_id = -2`. The file name defaults to `sportvu_player_only_debug.json` and the sidecar says
`"kind": "player_only_debug_NONSTANDARD"`.

## Import (top-down view)

`File > Import SportVU JSON` reads `events[0].moments` (any file with the structure above, including
real SportVU game files). Moments are aligned to video frames by time:
`moment unix_ms == epoch_ms + round(frame timestamp_s * 1000)` (latest moment at or before that time,
within a 200 ms tolerance). `epoch_ms` is taken from a btrv sidecar if present, otherwise the first
moment is aligned to the first frame; change it with *View > Set SportVU time offset*.
A SportVU file exported by btrv therefore renders frame-synchronously in the court view.
