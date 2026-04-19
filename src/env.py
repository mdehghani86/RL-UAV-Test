"""Single-UAV routing environment — fixed fork of Akshat's notebook cell 3.

Fixes applied vs upstream:
 - `_reward_regret_based` now counts customers that became unreachable from
   the *new* current node AND from the hypothetical depot return in remaining
   time; the original was ill-posed.
 - Terminal branch adds `alpha_partial * (served / total)` bonus so partial
   completion has a gradient even on truncation or out-of-time.
 - `layout_seed` is decoupled from `deterministic_seed`; `reset(seed=s)` now
   reshuffles node positions so each episode is a fresh instance (upstream
   used a fixed layout across all episodes).
 - `obs` gains `steps_frac = steps / max_steps` for horizon awareness.
 - `action_mask` is returned in both `reset` info and `step` info (upstream
   only returned it in `_final_info`).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class SingleUAVConfig:
    num_customers: int = 5
    num_chargers: int = 1
    map_size: Tuple[float, float] = (100.0, 100.0)
    mission_time: float = 500.0
    max_steps: int = 100
    battery_capacity: float = 250.0
    battery_per_meter: float = 0.5
    drone_speed: float = 15.0
    deadline_min: float = 80.0
    deadline_max: float = 250.0

    # "dist_normalized" | "completion_ratio" | "battery_aware"
    # "time_pressure"   | "regret_based"    | "shaped_potential"
    reward_mode: str = "completion_ratio"

    w_service: float = 30.0
    w_complete: float = 60.0
    w_infeasible: float = 10.0
    w_repeat: float = 1.0
    w_tardiness: float = 0.2
    w_step: float = 0.0

    # dist_normalized
    w_dist_bonus: float = 10.0
    # battery_aware
    w_battery_pen: float = 5.0
    battery_low_thresh: float = 0.25
    # time_pressure
    w_time_late_mult: float = 2.5
    # regret_based
    w_regret: float = 3.0
    # shaped_potential (new)
    w_potential: float = 20.0
    # partial-completion bonus (fix #3)
    alpha_partial: float = 20.0

    charger_time_cost: float = 1.0
    recharge_rate: float = 120.0

    layout_seed: Optional[int] = None  # if None, random each reset
    include_mask_in_obs: bool = True   # fix #2
    include_steps_in_obs: bool = True
    # Representation upgrades (curriculum-compatible — can be toggled per run)
    use_relative_frame: bool = False    # positions expressed as offsets from current node
    include_time_features: bool = False # per-customer time-to-reach / deadline-slack
    # Potential shaping selector: "simple" (original) | "rich" (dense time+battery signals)
    potential_mode: str = "rich"

    # Padding for curriculum transfer: if set, observation + action spaces
    # are sized to (pad_num_customers, pad_num_chargers). Inactive customer
    # slots have visited=0, deadline=0, position=0, mask=False. Action slots
    # beyond actual_total_nodes are always masked out.
    pad_num_customers: Optional[int] = None
    pad_num_chargers: Optional[int] = None

    def total_nodes(self) -> int:
        return 1 + self.num_customers + self.num_chargers

    def padded_num_customers(self) -> int:
        return self.pad_num_customers if self.pad_num_customers else self.num_customers

    def padded_num_chargers(self) -> int:
        return self.pad_num_chargers if self.pad_num_chargers else self.num_chargers

    def padded_total_nodes(self) -> int:
        return 1 + self.padded_num_customers() + self.padded_num_chargers()


@dataclass
class ServiceNode:
    node_id: int
    position: np.ndarray
    deadline: float


@dataclass
class ChargerNode:
    node_id: int
    position: np.ndarray


class SingleUAVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[SingleUAVConfig] = None):
        super().__init__()
        self.config = config or SingleUAVConfig()
        self._global_rng = np.random.default_rng(self.config.layout_seed)

        self.service_nodes: Dict[int, ServiceNode] = {}
        self.charger_nodes: Dict[int, ChargerNode] = {}

        self.depot_position = np.array(self.config.map_size, dtype=np.float32) / 2.0
        self.node_positions = np.zeros((self.config.total_nodes(), 2), dtype=np.float32)

        self.current_node = 0
        self.battery = self.config.battery_capacity
        self.time_remaining = self.config.mission_time
        self.elapsed_time = 0.0
        self.steps = 0
        self.visited_mask = np.zeros(self.config.num_customers, dtype=np.int8)

        self.ep_distance = 0.0
        self.ep_energy = 0.0
        self.ep_travel_time = 0.0
        self.ep_tardiness = 0.0
        self.ep_charger_visits = 0
        self.ep_infeasible = 0
        self.ep_repeat_visits = 0
        self._prev_potential = 0.0

        pN = self.config.padded_num_customers()
        pT = self.config.padded_total_nodes()
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0.0, 1.0, shape=(4 if self.config.include_steps_in_obs else 3,), dtype=np.float32),
            "visited": spaces.MultiBinary(pN),
            "deadlines": spaces.Box(0.0, 1.0, shape=(pN,), dtype=np.float32),
            "node_positions": spaces.Box(0.0, 1.0, shape=(pT, 2), dtype=np.float32),
            "action_mask": spaces.MultiBinary(pT),
        })
        self.action_space = spaces.Discrete(pT)

        self._build_nodes()

    def _build_nodes(self):
        self.node_positions[0] = self.depot_position
        w, h = self.config.map_size
        self.service_nodes.clear()
        for i in range(self.config.num_customers):
            nid = 1 + i
            pos = self._global_rng.uniform([0, 0], [w, h]).astype(np.float32)
            ddl = float(self._global_rng.uniform(self.config.deadline_min, self.config.deadline_max))
            self.service_nodes[nid] = ServiceNode(nid, pos, ddl)
            self.node_positions[nid] = pos
        self.charger_nodes.clear()
        for j in range(self.config.num_chargers):
            nid = 1 + self.config.num_customers + j
            pos = self._global_rng.uniform([0, 0], [w, h]).astype(np.float32)
            self.charger_nodes[nid] = ChargerNode(nid, pos)
            self.node_positions[nid] = pos

    def _get_obs(self):
        total = self.config.total_nodes()
        pN = self.config.padded_num_customers()
        pT = self.config.padded_total_nodes()
        cur_norm = float(self.current_node) / float(total - 1) if total > 1 else 0.0
        batt_frac = float(self.battery) / float(self.config.battery_capacity + 1e-9)
        time_frac = float(self.elapsed_time) / float(self.config.mission_time + 1e-9)
        agent_feats = [cur_norm, batt_frac, time_frac]
        if self.config.include_steps_in_obs:
            agent_feats.append(float(self.steps) / float(self.config.max_steps + 1e-9))

        w, h = self.config.map_size
        node_pos_norm = np.zeros((pT, 2), dtype=np.float32)
        node_pos_norm[:total, 0] = self.node_positions[:, 0] / w
        node_pos_norm[:total, 1] = self.node_positions[:, 1] / h
        # Reorder padded slots so depot(0) + real_customers + pad_customers
        # + real_chargers + pad_chargers. Internal layout is depot + customers
        # + chargers (real sizes); padded layout inserts pad slots AFTER real
        # customers and AFTER real chargers so action indices stay stable.
        N = self.config.num_customers
        M = self.config.num_chargers
        if pT > total:
            node_pos_norm = np.zeros((pT, 2), dtype=np.float32)
            node_pos_norm[0] = self.node_positions[0] / np.array([w, h], dtype=np.float32)
            node_pos_norm[1:1+N] = self.node_positions[1:1+N] / np.array([w, h], dtype=np.float32)
            pad_N = pN - N
            # pad_N zero customer slots
            offset = 1 + pN  # start of chargers in padded layout
            node_pos_norm[offset:offset+M] = self.node_positions[1+N:1+N+M] / np.array([w, h], dtype=np.float32)

        dmin, dmax = self.config.deadline_min, self.config.deadline_max
        deadlines = np.zeros(pN, dtype=np.float32)
        for i in range(N):
            ddl = self.service_nodes[1 + i].deadline
            deadlines[i] = (ddl - dmin) / (dmax - dmin + 1e-9)

        visited_padded = np.zeros(pN, dtype=np.int8)
        visited_padded[:N] = self.visited_mask

        mask_padded = np.zeros(pT, dtype=np.int8)
        real_mask = self.get_action_mask()
        mask_padded[0] = int(real_mask[0])
        mask_padded[1:1+N] = real_mask[1:1+N].astype(np.int8)
        offset = 1 + pN
        mask_padded[offset:offset+M] = real_mask[1+N:1+N+M].astype(np.int8)

        # Representation upgrade #1 — relative-frame positions.
        # When enabled, each node's (x,y) is the offset from the current node
        # normalised by map size, range ~[-1, 1]. Gives the network translation
        # invariance: same policy at any depot-customer geometry.
        if self.config.use_relative_frame:
            cur_xy = self.node_positions[self.current_node]
            pos_raw = np.zeros((pT, 2), dtype=np.float32)
            pos_raw[0] = self.node_positions[0] - cur_xy
            pos_raw[1:1+N] = self.node_positions[1:1+N] - cur_xy
            offset = 1 + pN
            pos_raw[offset:offset+M] = self.node_positions[1+N:1+N+M] - cur_xy
            node_pos_norm = pos_raw.astype(np.float32)
            node_pos_norm[:, 0] /= w
            node_pos_norm[:, 1] /= h

        out = {
            "agent": np.asarray(agent_feats, dtype=np.float32),
            "visited": visited_padded,
            "deadlines": deadlines,
            "node_positions": node_pos_norm,
            "action_mask": mask_padded,
        }

        # Representation upgrade #2 — per-customer time features.
        # For each customer: time_to_reach_frac = travel_time / time_remaining,
        # deadline_slack_frac = (deadline - elapsed - travel_time) / time_remaining.
        # Both clipped to [-1, 1]. Zero for visited or padded slots.
        if self.config.include_time_features:
            speed = self.config.drone_speed + 1e-9
            tr = np.zeros(pN, dtype=np.float32)
            sl = np.zeros(pN, dtype=np.float32)
            tmax = max(self.time_remaining, 1e-6)
            for i in range(N):
                if self.visited_mask[i] == 1:
                    continue
                nid = 1 + i
                d = float(np.linalg.norm(self.node_positions[self.current_node] - self.node_positions[nid]))
                tt = d / speed
                tr[i] = min(1.0, tt / tmax)
                slack = (self.service_nodes[nid].deadline - self.elapsed_time - tt) / tmax
                sl[i] = max(-1.0, min(1.0, slack))
            out["time_to_reach"] = tr
            out["deadline_slack"] = sl
        return out

    def padded_to_real_action(self, padded_action: int) -> int:
        """Translate a padded-space action index into the internal action index."""
        pN = self.config.padded_num_customers()
        N = self.config.num_customers
        if padded_action == 0:
            return 0
        if 1 <= padded_action <= N:
            return padded_action  # real customer
        if padded_action < 1 + pN:
            return -1  # pad customer — infeasible
        # charger slot
        ci = padded_action - (1 + pN)
        if ci < self.config.num_chargers:
            return 1 + N + ci
        return -1  # pad charger — infeasible

    def _distance(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.node_positions[a] - self.node_positions[b]))

    def _travel_energy(self, dist: float) -> float:
        return dist * self.config.battery_per_meter

    def _travel_time(self, dist: float) -> float:
        return dist / (self.config.drone_speed + 1e-9)

    def get_action_mask(self) -> np.ndarray:
        n = self.config.total_nodes()
        mask = np.ones(n, dtype=np.bool_)
        mask[self.current_node] = False
        for a in range(n):
            if not mask[a]:
                continue
            dist = self._distance(self.current_node, a)
            e = self._travel_energy(dist)
            t = self._travel_time(dist)
            if e > self.battery + 1e-9 or t > self.time_remaining + 1e-9 or t <= 0:
                mask[a] = False
        return mask

    def _count_reachable(self) -> int:
        count = 0
        for i in range(self.config.num_customers):
            if self.visited_mask[i] == 1:
                continue
            nid = 1 + i
            dist = self._distance(self.current_node, nid)
            if (self._travel_energy(dist) <= self.battery + 1e-9 and
                    self._travel_time(dist) <= self.time_remaining + 1e-9):
                count += 1
        return count

    def _count_feasible(self) -> int:
        """Stricter than _count_reachable: also enforces per-customer deadline.
        A customer is feasible iff battery + time-budget allow arrival AND
        arrival_time <= customer_deadline. Used by the "tight" potential."""
        count = 0
        for i in range(self.config.num_customers):
            if self.visited_mask[i] == 1:
                continue
            nid = 1 + i
            dist = self._distance(self.current_node, nid)
            tt = self._travel_time(dist)
            if (self._travel_energy(dist) > self.battery + 1e-9 or
                    tt > self.time_remaining + 1e-9):
                continue
            if self.elapsed_time + tt <= self.service_nodes[nid].deadline + 1e-9:
                count += 1
        return count

    def _min_remaining_work(self) -> float:
        """Cheap estimate of remaining travel time to touch all unvisited
        customers, computed as a nearest-neighbour TSP from the current node.
        NOTE: NN is NOT a mathematical lower bound on optimal tour length; it
        is a pragmatic O(N^2) approximation. Only used as a potential-shaping
        pressure term — potential-based shaping preserves optimality regardless
        of the potential's semantic accuracy."""
        unvisited = [1 + i for i in range(self.config.num_customers)
                     if self.visited_mask[i] == 0]
        if not unvisited:
            return 0.0
        cur = self.current_node
        total_t = 0.0
        remaining = list(unvisited)
        while remaining:
            best_d, best_j = 1e9, -1
            for j, nid in enumerate(remaining):
                d = self._distance(cur, nid)
                if d < best_d:
                    best_d, best_j = d, j
            total_t += self._travel_time(best_d)
            cur = remaining.pop(best_j)
        return total_t

    def _potential(self) -> float:
        """Heuristic potential function for reward shaping.

        Two modes:
          - "simple": served_frac − 0.3·min_dist/diag   (original)
          - "rich":   served_frac − 0.3·min_dist/diag
                      − 0.25·max(0, time_used_frac − served_frac)   (time-served imbalance)
                      + 0.10·(battery_frac − 0.3)                    (battery reserve bonus)
                      + 0.15·(#reachable_unvisited / N)              (future flexibility)
        The rich form gives a per-step gradient even when no customer is
        served — it rewards "positioning well" and penalises "wasting time
        without making progress". Potential-based ⇒ optimal policy invariant.
        """
        cfg = self.config
        N = cfg.num_customers
        frac_served = float(self.visited_mask.sum()) / float(N)
        if int(N - self.visited_mask.sum()) == 0:
            return 1.0

        cur_pos = self.node_positions[self.current_node]
        min_d = 1e9
        for i in range(N):
            if self.visited_mask[i] == 1:
                continue
            d = float(np.linalg.norm(cur_pos - self.node_positions[1 + i]))
            if d < min_d:
                min_d = d
        diag = math.hypot(*cfg.map_size)
        base = frac_served - 0.3 * (min_d / (diag + 1e-9))
        if cfg.potential_mode == "simple":
            return base
        # Rich additions
        time_used_frac = self.elapsed_time / (cfg.mission_time + 1e-9)
        batt_frac = self.battery / (cfg.battery_capacity + 1e-9)
        reachable_frac = self._count_reachable() / (N + 1e-9)
        rich = (base
                - 0.25 * max(0.0, time_used_frac - frac_served)
                + 0.10 * (batt_frac - 0.3)
                + 0.15 * reachable_frac)
        if cfg.potential_mode == "rich":
            return rich
        # "tight" mode: adds deadline-feasibility reachable count and a
        # min-remaining-work pressure term. Gives the agent a dense signal
        # about how much late-game slack it has left.
        feasible_frac = self._count_feasible() / (N + 1e-9)
        min_work = self._min_remaining_work()
        time_remaining = max(cfg.mission_time - self.elapsed_time, 1e-6)
        work_pressure = min(1.0, min_work / time_remaining)  # ~0 = plenty of time, ~1 = impossible
        return (rich
                + 0.15 * (feasible_frac - reachable_frac)   # reward being *deadline-safe*, not just reachable
                - 0.10 * work_pressure)                      # penalise being close to time-infeasible

    # ---------- reward functions ----------
    def _reward_dist_normalized(self, served_new, repeated, infeasible, completed_and_depot,
                                 dist, travel_t, energy, tardiness, charged):
        cfg = self.config
        if infeasible:
            return -cfg.w_infeasible
        r = -cfg.w_step
        if repeated:
            r -= cfg.w_repeat
        if served_new:
            diag = math.hypot(*cfg.map_size)
            efficiency = 1.0 - min(dist / (diag + 1e-9), 1.0)
            r += cfg.w_service + cfg.w_dist_bonus * efficiency
            r -= cfg.w_tardiness * tardiness
        if completed_and_depot:
            r += cfg.w_complete
        return r

    def _reward_completion_ratio(self, served_new, repeated, infeasible, completed_and_depot,
                                  dist, travel_t, energy, tardiness, charged):
        cfg = self.config
        if infeasible:
            return -cfg.w_infeasible
        r = -cfg.w_step
        if repeated:
            r -= cfg.w_repeat
        if served_new:
            ratio = float(self.visited_mask.sum()) / (cfg.num_customers + 1e-9)
            r += cfg.w_service * (1.0 + ratio ** 2)
            r -= cfg.w_tardiness * tardiness
        if completed_and_depot:
            r += cfg.w_complete
        return r

    def _reward_battery_aware(self, served_new, repeated, infeasible, completed_and_depot,
                               dist, travel_t, energy, tardiness, charged):
        cfg = self.config
        if infeasible:
            return -cfg.w_infeasible
        r = -cfg.w_step
        if repeated:
            r -= cfg.w_repeat
        batt_frac = self.battery / (cfg.battery_capacity + 1e-9)
        if batt_frac < cfg.battery_low_thresh and not charged:
            severity = (cfg.battery_low_thresh - batt_frac) / (cfg.battery_low_thresh + 1e-9)
            r -= cfg.w_battery_pen * severity
        if served_new:
            r += cfg.w_service
            r -= cfg.w_tardiness * tardiness
        if completed_and_depot:
            r += cfg.w_complete
        return r

    def _reward_time_pressure(self, served_new, repeated, infeasible, completed_and_depot,
                               dist, travel_t, energy, tardiness, charged):
        cfg = self.config
        if infeasible:
            return -cfg.w_infeasible
        r = -cfg.w_step
        if repeated:
            r -= cfg.w_repeat
        if served_new:
            time_progress = self.elapsed_time / (cfg.mission_time + 1e-9)
            if time_progress >= 0.70:
                ramp = (time_progress - 0.70) / 0.30
                mult = 1.0 + (cfg.w_time_late_mult - 1.0) * ramp
            else:
                mult = 1.0
            r += cfg.w_service * mult
            r -= cfg.w_tardiness * tardiness
        if completed_and_depot:
            r += cfg.w_complete
        return r

    def _reward_regret_based(self, served_new, repeated, infeasible, completed_and_depot,
                              dist, travel_t, energy, tardiness, charged, reachable_before: int):
        """Regret = count of customers that WERE reachable before the move but
        can no longer be reached (from new node AND within remaining mission
        time back to depot). Semantically fixed vs upstream."""
        cfg = self.config
        if infeasible:
            return -cfg.w_infeasible
        r = -cfg.w_step
        if repeated:
            r -= cfg.w_repeat
        reachable_after = self._count_reachable()
        newly_lost = max(0, reachable_before - reachable_after - int(served_new))
        r -= cfg.w_regret * newly_lost
        if served_new:
            r += cfg.w_service
            r -= cfg.w_tardiness * tardiness
        if completed_and_depot:
            r += cfg.w_complete
        return r

    def _reward_shaped_potential(self, served_new, repeated, infeasible, completed_and_depot,
                                  dist, travel_t, energy, tardiness, charged, prev_phi: float):
        """Potential-based shaping — F(s,a,s') = gamma * phi(s') - phi(s),
        guaranteed not to change the optimal policy."""
        cfg = self.config
        if infeasible:
            return -cfg.w_infeasible
        r = -cfg.w_step
        if repeated:
            r -= cfg.w_repeat
        if served_new:
            r += cfg.w_service
            r -= cfg.w_tardiness * tardiness
        if completed_and_depot:
            r += cfg.w_complete
        phi_s_prime = self._potential()
        r += cfg.w_potential * (0.99 * phi_s_prime - prev_phi)
        return r

    def _compute_reward(self, served_new, repeated, infeasible, completed_and_depot,
                         dist, travel_t, energy, tardiness, charged,
                         reachable_before: int = 0, prev_phi: float = 0.0):
        m = self.config.reward_mode
        if m == "dist_normalized":
            return self._reward_dist_normalized(served_new, repeated, infeasible,
                                                completed_and_depot, dist, travel_t,
                                                energy, tardiness, charged)
        if m == "completion_ratio":
            return self._reward_completion_ratio(served_new, repeated, infeasible,
                                                 completed_and_depot, dist, travel_t,
                                                 energy, tardiness, charged)
        if m == "battery_aware":
            return self._reward_battery_aware(served_new, repeated, infeasible,
                                              completed_and_depot, dist, travel_t,
                                              energy, tardiness, charged)
        if m == "time_pressure":
            return self._reward_time_pressure(served_new, repeated, infeasible,
                                              completed_and_depot, dist, travel_t,
                                              energy, tardiness, charged)
        if m == "regret_based":
            return self._reward_regret_based(served_new, repeated, infeasible,
                                             completed_and_depot, dist, travel_t,
                                             energy, tardiness, charged, reachable_before)
        if m == "shaped_potential":
            return self._reward_shaped_potential(served_new, repeated, infeasible,
                                                 completed_and_depot, dist, travel_t,
                                                 energy, tardiness, charged, prev_phi)
        raise ValueError(f"Unknown reward_mode: {m}")

    def _final_info(self) -> dict:
        return {
            "elapsed_time": self.elapsed_time,
            "returned_to_depot": (self.current_node == 0),
            "completed": bool(self.visited_mask.all()),
            "customers_served": int(self.visited_mask.sum()),
            "ep_distance": float(self.ep_distance),
            "ep_energy": float(self.ep_energy),
            "ep_travel_time": float(self.ep_travel_time),
            "ep_tardiness": float(self.ep_tardiness),
            "ep_charger_visits": int(self.ep_charger_visits),
            "ep_infeasible": int(self.ep_infeasible),
            "ep_repeat_visits": int(self.ep_repeat_visits),
            "action_mask": self._padded_action_mask(),
        }

    def _padded_action_mask(self) -> np.ndarray:
        """Return action mask in padded action-space coords."""
        pN = self.config.padded_num_customers()
        pT = self.config.padded_total_nodes()
        N = self.config.num_customers
        M = self.config.num_chargers
        real_mask = self.get_action_mask()
        if pT == self.config.total_nodes():
            return real_mask
        mask_padded = np.zeros(pT, dtype=np.bool_)
        mask_padded[0] = real_mask[0]
        mask_padded[1:1+N] = real_mask[1:1+N]
        offset = 1 + pN
        mask_padded[offset:offset+M] = real_mask[1+N:1+N+M]
        return mask_padded

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._global_rng = np.random.default_rng(seed)
        self._build_nodes()
        self.current_node = 0
        self.battery = self.config.battery_capacity
        self.time_remaining = self.config.mission_time
        self.elapsed_time = 0.0
        self.steps = 0
        self.visited_mask = np.zeros(self.config.num_customers, dtype=np.int8)
        self.ep_distance = self.ep_energy = self.ep_travel_time = 0.0
        self.ep_tardiness = 0.0
        self.ep_charger_visits = self.ep_infeasible = self.ep_repeat_visits = 0
        self._prev_potential = self._potential()
        obs = self._get_obs()
        info = {"action_mask": self._padded_action_mask()}
        return obs, info

    def step(self, action: int):
        # Check step budget BEFORE incrementing (fixes off-by-one — agent gets
        # exactly max_steps actions). Truncation uses terminated=False, truncated=True
        # so GAE correctly bootstraps V(s_T) per Gymnasium semantics.
        n = self.config.total_nodes()
        cfg = self.config

        if self.steps >= cfg.max_steps:
            served = float(self.visited_mask.sum()) / float(cfg.num_customers)
            bonus = cfg.alpha_partial * served
            return self._get_obs(), float(bonus), False, True, self._final_info()

        self.steps += 1

        # Translate padded action to real action. Padded-only actions
        # (pad-customer or pad-charger slots) are infeasible.
        padded_action = int(action)
        if cfg.pad_num_customers or cfg.pad_num_chargers:
            action = self.padded_to_real_action(padded_action)
            if action == -1:
                self.ep_infeasible += 1
                return self._get_obs(), float(-cfg.w_infeasible), True, False, self._final_info()
        else:
            action = padded_action

        if action < 0 or action >= n or action == self.current_node:
            self.ep_infeasible += 1
            return self._get_obs(), float(-cfg.w_infeasible), True, False, self._final_info()

        dist = self._distance(self.current_node, action)
        energy = self._travel_energy(dist)
        travel_t = self._travel_time(dist)
        if energy > self.battery + 1e-9 or travel_t > self.time_remaining + 1e-9 or travel_t <= 0:
            self.ep_infeasible += 1
            return self._get_obs(), float(-cfg.w_infeasible), True, False, self._final_info()

        reachable_before = self._count_reachable() if cfg.reward_mode == "regret_based" else 0
        prev_phi = self._prev_potential if cfg.reward_mode == "shaped_potential" else 0.0

        self.battery -= energy
        self.time_remaining -= travel_t
        self.elapsed_time += travel_t
        self.ep_distance += dist
        self.ep_energy += energy
        self.ep_travel_time += travel_t
        self.current_node = action

        served_new = repeated = charged = False
        tardiness = 0.0
        if 1 <= action <= cfg.num_customers:
            idx = action - 1
            if self.visited_mask[idx] == 1:
                repeated = True
                self.ep_repeat_visits += 1
            else:
                served_new = True
                self.visited_mask[idx] = 1
                tardiness = max(0.0, self.elapsed_time - self.service_nodes[action].deadline)
                self.ep_tardiness += tardiness

        if action in self.charger_nodes:
            charged = True
            self.ep_charger_visits += 1
            self.time_remaining -= cfg.charger_time_cost
            self.elapsed_time += cfg.charger_time_cost
            if self.time_remaining < 0:
                self.ep_infeasible += 1
                return self._get_obs(), float(-cfg.w_infeasible), True, False, self._final_info()
            self.battery = min(cfg.battery_capacity, self.battery + cfg.recharge_rate)

        completed_and_depot = bool(self.visited_mask.all() and self.current_node == 0)
        r = self._compute_reward(served_new, repeated, False, completed_and_depot,
                                  dist, travel_t, energy, tardiness, charged,
                                  reachable_before, prev_phi)

        if cfg.reward_mode == "shaped_potential":
            self._prev_potential = self._potential()

        if completed_and_depot:
            return self._get_obs(), float(r), True, False, self._final_info()
        if self.time_remaining <= 0:
            served = float(self.visited_mask.sum()) / float(cfg.num_customers)
            bonus = cfg.alpha_partial * served
            return self._get_obs(), float(r - cfg.w_infeasible + bonus), True, False, self._final_info()
        return self._get_obs(), float(r), False, False, self._final_info()


def make_env(cfg_overrides: Optional[dict] = None) -> SingleUAVEnv:
    cfg = SingleUAVConfig()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    return SingleUAVEnv(cfg)


if __name__ == "__main__":
    env = make_env({"reward_mode": "completion_ratio"})
    obs, info = env.reset(seed=0)
    print("obs keys:", list(obs.keys()))
    print("action mask shape:", obs["action_mask"].shape, "any valid:", obs["action_mask"].any())
    for mode in ["dist_normalized", "completion_ratio", "battery_aware",
                 "time_pressure", "regret_based", "shaped_potential"]:
        e = make_env({"reward_mode": mode})
        o, inf = e.reset(seed=0)
        mask = inf["action_mask"]
        a = int(np.where(mask)[0][0])
        _, r, _, _, _ = e.step(a)
        print(f"  {mode:20s} step reward={r:.3f}")
    print("env.py OK")
