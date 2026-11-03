"""Room and four-player rank decoding, plus immutable expert seat selection."""

from dataclasses import dataclass

from .replays import ReplayError, parse_reference, validate_head

ROOMS = {"jade-east": 11, "jade-south": 12, "throne-east": 15, "throne-south": 16}
ROOM_GAME_MODES = {11: 1, 12: 2, 15: 1, 16: 2}
MIN_RANKS = {"master": 4, "saint": 5, "celestial": 6}
RANK_NAMES = {1: "初心", 2: "雀士", 3: "雀杰", 4: "雀豪", 5: "雀圣", 6: "魂天"}


def rank(level_id):
    # Four-player and three-player ladders must never be compared numerically.
    if type(level_id) is not int or level_id // 10000 != 1:
        return None
    band, stars = (level_id % 10000) // 100, level_id % 100
    if 1 <= band <= 5 and 1 <= stars <= 3:
        return band
    if (band == 6 and stars == 1) or (band == 7 and 1 <= stars <= 20):
        return 6  # Legacy 10601 and current 10701..10720 are both 魂天.
    return None


@dataclass(frozen=True)
class ExpertFilter:
    rooms: tuple = ("jade-south", "throne-south")
    min_rank: str = "saint"
    min_players: int = 1

    def validate(self):
        if (
            not self.rooms
            or len(set(self.rooms)) != len(self.rooms)
            or any(r not in ROOMS for r in self.rooms)
        ):
            raise ReplayError("请选择 jade/throne 的 east/south 房间。")
        if self.min_rank not in MIN_RANKS or not 1 <= self.min_players <= 4:
            raise ReplayError("高手门槛需为 master/saint/celestial，人数需在 1..4。")

    def to_dict(self):
        return {
            "rooms": sorted(self.rooms),
            "min_rank": self.min_rank,
            "min_players": self.min_players,
        }

    def room_ids(self):
        return {ROOMS[r] for r in self.rooms}


def discover_candidate(head, selection):
    parse_reference(head.uuid)
    if head.game_config.category != 2 or head.game_config.meta.mode_id not in selection.room_ids():
        return None
    if (
        head.game_config.mode.mode != ROOM_GAME_MODES[head.game_config.meta.mode_id]
        or len(head.players) != 4
    ):
        return None
    account_ids = [p.account_id for p in head.players]
    if min(account_ids) <= 0 or len(set(account_ids)) != 4:
        return None
    # seat_list carries account IDs, not indices 0..3. Do not infer a seat from player order.
    if head.seat_list and (len(head.seat_list) != 4 or set(head.seat_list) != set(account_ids)):
        return None
    players = [
        {"account_id": p.account_id, "level_id": p.level.id, "rank": rank(p.level.id)}
        for p in head.players
    ]
    qualified = sum(
        p["rank"] is not None and p["rank"] >= MIN_RANKS[selection.min_rank] for p in players
    )
    if qualified < selection.min_players:
        return None
    return {
        "game_id": head.uuid,
        "room_mode_id": head.game_config.meta.mode_id,
        "start_time": head.start_time,
        "players": players,
    }


def completed_selection(head, candidate, selection):
    validate_head(head, candidate["game_id"])
    if head.config.meta.mode_id != candidate["room_mode_id"]:
        raise ReplayError("观战房间与最终牌谱不一致。")
    if head.config.mode.mode != ROOM_GAME_MODES.get(candidate["room_mode_id"]):
        raise ReplayError("观战房间与最终牌谱东风/半庄模式不一致。")
    if {a.account_id for a in head.accounts} != {p["account_id"] for p in candidate["players"]}:
        raise ReplayError("观战玩家与最终牌谱不一致。")
    levels, seats = {}, []
    for account in head.accounts:
        band = rank(account.level.id)
        levels[str(account.seat)] = {
            "level_id": account.level.id,
            "rank": band,
            "rank_name": RANK_NAMES.get(band, "unknown"),
        }
        if band is not None and band >= MIN_RANKS[selection.min_rank]:
            seats.append(account.seat)
    if len(seats) < selection.min_players:
        return None
    return {
        "expert_filter": selection.to_dict(),
        "player_levels": levels,
        "selected_seats": sorted(seats),
        "room_mode_id": candidate["room_mode_id"],
    }
