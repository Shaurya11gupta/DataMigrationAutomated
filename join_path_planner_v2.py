from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class _AdjEdge:
    neighbor: str
    edge: Any
    edge_id: Tuple[Any, ...]
    adjusted_confidence: float
    effective_prob: float
    edge_cost: float


@dataclass(frozen=True)
class _PathStep:
    from_table: str
    to_table: str
    edge: Any
    edge_id: Tuple[Any, ...]
    adjusted_confidence: float
    effective_prob: float
    edge_cost: float


class JoinPathPlannerV2:
    """
    Production-grade join path planner.

    Improvements over v1:
      1) Keeps multiple edges per table pair (single + composite candidates)
      2) Canonical edge IDs (orientation-safe) for robust banning/deduping
      3) Stable heap tie-breakers (no object comparison crashes)
      4) Cycle-safe shortest-path search
      5) Traversal-aware direction labeling
      6) Cost objective is consistent with reported path confidence
      7) Optional cardinality-aware confidence adjustment
      8) True top-k shortest simple paths using Yen-style spur search
    """

    def __init__(
        self,
        edges: Sequence[Any],
        composite_bonus: float = 0.05,
        directional_bonus: float = 0.03,
        reverse_directional_bonus_factor: float = 0.4,
        path_length_penalty: float = 0.97,
        max_edges_per_pair: int = 5,
        cardinality_multipliers: Optional[Dict[str, float]] = None,
    ):
        self.raw_edges = list(edges or [])
        self.composite_bonus = max(0.0, composite_bonus)
        self.directional_bonus = max(0.0, directional_bonus)
        self.reverse_directional_bonus_factor = max(0.0, reverse_directional_bonus_factor)
        self.path_length_penalty = max(1e-6, min(1.0, path_length_penalty))
        self.max_edges_per_pair = max(1, max_edges_per_pair)
        self.cardinality_multipliers = cardinality_multipliers or {
            "1:1": 1.00,
            "1:N": 0.985,
            "N:1": 0.985,
            "N:M": 0.88,
        }

        self.graph: Dict[str, List[_AdjEdge]] = defaultdict(list)
        self._tie_counter = count()
        self._build_graph()

    # =====================================================
    # Graph Build
    # =====================================================

    def _build_graph(self) -> None:
        # 1) Deduplicate exact join conditions by canonical edge ID, keep strongest confidence.
        unique_by_edge: Dict[Tuple[Any, ...], Any] = {}
        for e in self.raw_edges:
            eid = self._canonical_edge_id(e)
            if eid not in unique_by_edge or unique_by_edge[eid].confidence < e.confidence:
                unique_by_edge[eid] = e

        # 2) Bucket by table-pair and keep diverse top edges.
        pair_buckets: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for e in unique_by_edge.values():
            pair = tuple(sorted((e.left_table, e.right_table)))
            pair_buckets[pair].append(e)

        selected_edges: List[Any] = []
        for pair, bucket in pair_buckets.items():
            selected_edges.extend(self._select_diverse_edges(bucket))

        # 3) Build adjacency in both directions with traversal-specific confidence/cost.
        for e in selected_edges:
            eid = self._canonical_edge_id(e)

            left_conf = self._adjust_confidence(e, e.left_table, e.right_table)
            left_prob = max(1e-9, left_conf * self.path_length_penalty)
            left_cost = -math.log(left_prob)
            self.graph[e.left_table].append(
                _AdjEdge(
                    neighbor=e.right_table,
                    edge=e,
                    edge_id=eid,
                    adjusted_confidence=left_conf,
                    effective_prob=left_prob,
                    edge_cost=left_cost,
                )
            )

            right_conf = self._adjust_confidence(e, e.right_table, e.left_table)
            right_prob = max(1e-9, right_conf * self.path_length_penalty)
            right_cost = -math.log(right_prob)
            self.graph[e.right_table].append(
                _AdjEdge(
                    neighbor=e.left_table,
                    edge=e,
                    edge_id=eid,
                    adjusted_confidence=right_conf,
                    effective_prob=right_prob,
                    edge_cost=right_cost,
                )
            )

        # Deterministic ordering.
        for node in self.graph:
            self.graph[node].sort(
                key=lambda a: (a.neighbor, -a.adjusted_confidence, str(a.edge_id))
            )

    def _select_diverse_edges(self, edges: Sequence[Any]) -> List[Any]:
        """
        Keep both single and composite candidates when possible.
        """
        if not edges:
            return []

        def strength(e: Any) -> float:
            return self._adjust_confidence(
                e,
                getattr(e, "left_table"),
                getattr(e, "right_table"),
            )

        ranked = sorted(
            edges,
            key=lambda e: (
                -strength(e),
                -(len(getattr(e, "left_cols", []))),
                str(self._canonical_edge_id(e)),
            ),
        )

        singles = [e for e in ranked if len(getattr(e, "left_cols", [])) == 1]
        composites = [e for e in ranked if len(getattr(e, "left_cols", [])) > 1]

        selected: List[Any] = []
        seen: Set[Tuple[Any, ...]] = set()

        # Ensure diversity first.
        for pool in (composites, singles):
            if pool:
                eid = self._canonical_edge_id(pool[0])
                if eid not in seen:
                    selected.append(pool[0])
                    seen.add(eid)

        # Fill remaining by global strength.
        for e in ranked:
            if len(selected) >= self.max_edges_per_pair:
                break
            eid = self._canonical_edge_id(e)
            if eid in seen:
                continue
            selected.append(e)
            seen.add(eid)

        return selected

    # =====================================================
    # Confidence + Cost
    # =====================================================

    def _adjust_confidence(self, edge: Any, from_table: str, to_table: str) -> float:
        conf = float(max(0.0, min(1.0, getattr(edge, "confidence", 0.0))))

        # Composite joins generally carry stronger structural evidence.
        if len(getattr(edge, "left_cols", [])) > 1:
            conf += self.composite_bonus

        # Cardinality-aware damping.
        card = getattr(edge, "cardinality", "N:M")
        conf *= self.cardinality_multipliers.get(card, 1.0)

        # Traversal-aware directional preference.
        direction = getattr(edge, "direction", "undirected")
        if direction == "left_parent":
            if from_table == edge.left_table and to_table == edge.right_table:
                conf += self.directional_bonus
            elif from_table == edge.right_table and to_table == edge.left_table:
                conf += self.directional_bonus * self.reverse_directional_bonus_factor
        elif direction == "right_parent":
            if from_table == edge.right_table and to_table == edge.left_table:
                conf += self.directional_bonus
            elif from_table == edge.left_table and to_table == edge.right_table:
                conf += self.directional_bonus * self.reverse_directional_bonus_factor

        return float(max(1e-6, min(1.0, conf)))

    # =====================================================
    # Public search API
    # =====================================================

    def find_path(
        self,
        start: str,
        target: str,
        min_conf: float = 0.55,
        max_hops: int = 6,
        banned_edges: Optional[Set[Tuple[Any, ...]]] = None,
        banned_tables: Optional[Set[str]] = None,
        required_tables: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns the best simple path under constraints, or None.
        """
        internal = self._shortest_path_internal(
            start=start,
            target=target,
            min_conf=min_conf,
            max_hops=max_hops,
            banned_edges=banned_edges or set(),
            banned_tables=banned_tables or set(),
            required_tables=required_tables or set(),
        )
        if not internal:
            return None
        return self._materialize(internal)

    def find_top_k_paths(
        self,
        start: str,
        target: str,
        k: int = 3,
        min_conf: float = 0.55,
        max_hops: int = 6,
        banned_tables: Optional[Set[str]] = None,
        required_tables: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Yen-style top-k shortest simple paths.
        """
        if k <= 0:
            return []

        banned_tables = banned_tables or set()
        required_tables = required_tables or set()

        first = self._shortest_path_internal(
            start=start,
            target=target,
            min_conf=min_conf,
            max_hops=max_hops,
            banned_edges=set(),
            banned_tables=banned_tables,
            required_tables=required_tables,
        )
        if not first:
            return []

        A: List[Dict[str, Any]] = [first]
        B_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        seen_candidates: Set[Tuple[Any, ...]] = set()
        accepted_signatures: Set[Tuple[Any, ...]] = {self._path_signature(first)}

        for _ in range(1, k):
            prev = A[-1]
            prev_nodes: List[str] = prev["nodes"]
            prev_steps: List[_PathStep] = prev["steps"]

            for spur_idx in range(len(prev_nodes) - 1):
                spur_node = prev_nodes[spur_idx]
                root_steps = prev_steps[:spur_idx]
                root_nodes = prev_nodes[: spur_idx + 1]

                root_cost = sum(s.edge_cost for s in root_steps)
                root_conf = 1.0
                for s in root_steps:
                    root_conf *= s.effective_prob

                # Ban one outgoing edge for every accepted path sharing this exact root.
                banned_edges: Set[Tuple[Any, ...]] = set()
                root_sig = self._steps_signature(root_steps)
                for accepted in A:
                    acc_steps = accepted["steps"]
                    if len(acc_steps) <= spur_idx:
                        continue
                    if self._steps_signature(acc_steps[:spur_idx]) == root_sig:
                        banned_edges.add(acc_steps[spur_idx].edge_id)

                # Prevent loops through root prefix nodes (except spur node itself).
                spur_banned_tables = set(banned_tables) | set(root_nodes[:-1])

                remaining_required = set(required_tables) - set(root_nodes)
                remaining_hops = max_hops - spur_idx
                if remaining_hops <= 0:
                    continue

                spur = self._shortest_path_internal(
                    start=spur_node,
                    target=target,
                    min_conf=min_conf,
                    max_hops=remaining_hops,
                    banned_edges=banned_edges,
                    banned_tables=spur_banned_tables,
                    required_tables=remaining_required,
                )
                if not spur:
                    continue

                total_steps = root_steps + spur["steps"]
                total_nodes = root_nodes[:-1] + spur["nodes"]
                total_cost = root_cost + spur["cost"]
                total_conf = root_conf * spur["path_conf"]
                candidate = {
                    "steps": total_steps,
                    "nodes": total_nodes,
                    "cost": total_cost,
                    "path_conf": total_conf,
                }
                sig = self._path_signature(candidate)
                if sig in accepted_signatures or sig in seen_candidates:
                    continue
                seen_candidates.add(sig)
                heapq.heappush(B_heap, (total_cost, next(self._tie_counter), candidate))

            if not B_heap:
                break

            _, _, best_cand = heapq.heappop(B_heap)
            A.append(best_cand)
            accepted_signatures.add(self._path_signature(best_cand))

        return [self._materialize(p) for p in A[:k]]

    # =====================================================
    # Shortest path core
    # =====================================================

    def _shortest_path_internal(
        self,
        start: str,
        target: str,
        min_conf: float,
        max_hops: int,
        banned_edges: Set[Tuple[Any, ...]],
        banned_tables: Set[str],
        required_tables: Set[str],
    ) -> Optional[Dict[str, Any]]:
        if start == target:
            if required_tables and start not in required_tables:
                return None
            return {"steps": [], "nodes": [start], "cost": 0.0, "path_conf": 1.0}

        if start not in self.graph or target not in self.graph:
            return None
        if start in banned_tables or target in banned_tables:
            return None
        if max_hops <= 0:
            return None

        # (total_cost, hops, tie, current_node, steps, path_conf, visited_nodes)
        pq: List[Tuple[float, int, int, str, List[_PathStep], float, Set[str]]] = [
            (0.0, 0, next(self._tie_counter), start, [], 1.0, {start})
        ]

        # Dominance pruning by node+hops+required_covered signature.
        best_cost: Dict[Tuple[str, int, Tuple[str, ...]], float] = {}

        while pq:
            total_cost, hops, _, node, steps, conf, visited = heapq.heappop(pq)

            covered_req = tuple(sorted((set(visited) | {node}) & required_tables))
            state = (node, hops, covered_req)
            if state in best_cost and total_cost >= best_cost[state]:
                continue
            best_cost[state] = total_cost

            if node == target:
                if not required_tables or required_tables.issubset(set(visited) | {node}):
                    return {
                        "steps": steps,
                        "nodes": self._nodes_from_steps(start, steps),
                        "cost": total_cost,
                        "path_conf": conf,
                    }

            if hops >= max_hops:
                continue

            for adj in self.graph[node]:
                nxt = adj.neighbor

                if adj.edge_id in banned_edges:
                    continue
                if adj.adjusted_confidence < min_conf:
                    continue
                if nxt in banned_tables:
                    continue
                if nxt in visited:
                    # simple paths only -> cycle-safe
                    continue

                new_cost = total_cost + adj.edge_cost
                new_conf = conf * adj.effective_prob
                step = _PathStep(
                    from_table=node,
                    to_table=nxt,
                    edge=adj.edge,
                    edge_id=adj.edge_id,
                    adjusted_confidence=adj.adjusted_confidence,
                    effective_prob=adj.effective_prob,
                    edge_cost=adj.edge_cost,
                )
                new_steps = steps + [step]
                new_visited = set(visited)
                new_visited.add(nxt)

                heapq.heappush(
                    pq,
                    (
                        new_cost,
                        hops + 1,
                        next(self._tie_counter),
                        nxt,
                        new_steps,
                        new_conf,
                        new_visited,
                    ),
                )

        return None

    # =====================================================
    # Materialization / output helpers
    # =====================================================

    def _materialize(self, path: Dict[str, Any]) -> Dict[str, Any]:
        steps_out = []
        for s in path["steps"]:
            lcols, rcols = self._oriented_cols(s.edge, s.from_table, s.to_table)
            traversal_dir = self._traversal_direction(s.edge, s.from_table, s.to_table)
            steps_out.append(
                {
                    "from": s.from_table,
                    "to": s.to_table,
                    "left_cols": list(lcols),
                    "right_cols": list(rcols),
                    "confidence": round(s.adjusted_confidence, 4),
                    "effective_prob": round(s.effective_prob, 6),
                    "base_confidence": round(float(getattr(s.edge, "confidence", 0.0)), 4),
                    "cardinality": getattr(s.edge, "cardinality", "N:M"),
                    "edge_direction": getattr(s.edge, "direction", "undirected"),
                    "traversal_direction": traversal_dir,
                    "edge_id": s.edge_id,
                }
            )

        return {
            "steps": steps_out,
            "path_confidence": round(path["path_conf"], 6),
            "path_cost": round(path["cost"], 6),
            "hop_count": len(steps_out),
            "node_path": self._nodes_from_materialized_steps(steps_out),
        }

    def print_path(self, path: Optional[Dict[str, Any]]) -> None:
        if not path:
            print("No path found")
            return
        print("\nJOIN PATH")
        print("=" * 60)
        for idx, s in enumerate(path["steps"], 1):
            print(
                f"{idx:02d}. {s['from']} -> {s['to']} | "
                f"{s['left_cols']} = {s['right_cols']} | "
                f"adj_conf={s['confidence']} eff_prob={s['effective_prob']} "
                f"card={s['cardinality']} trav={s['traversal_direction']}"
            )
        print("-" * 60)
        print("nodes:", " -> ".join(path["node_path"]))
        print("path_confidence:", path["path_confidence"])
        print("path_cost:", path["path_cost"])

    # =====================================================
    # ID / signature helpers
    # =====================================================

    def _canonical_edge_id(self, edge: Any) -> Tuple[Any, ...]:
        lt = edge.left_table
        rt = edge.right_table
        lcols = list(edge.left_cols)
        rcols = list(edge.right_cols)

        if lt <= rt:
            pairs = tuple(sorted(zip(lcols, rcols), key=lambda x: (x[0], x[1])))
            return (lt, rt, pairs)

        pairs = tuple(sorted(zip(rcols, lcols), key=lambda x: (x[0], x[1])))
        return (rt, lt, pairs)

    def _path_signature(self, path: Dict[str, Any]) -> Tuple[Any, ...]:
        steps: List[_PathStep] = path["steps"]
        return tuple((s.from_table, s.to_table, s.edge_id) for s in steps)

    def _steps_signature(self, steps: Sequence[_PathStep]) -> Tuple[Any, ...]:
        return tuple((s.from_table, s.to_table, s.edge_id) for s in steps)

    def _nodes_from_steps(self, start: str, steps: Sequence[_PathStep]) -> List[str]:
        nodes = [start]
        for s in steps:
            nodes.append(s.to_table)
        return nodes

    def _nodes_from_materialized_steps(self, steps: Sequence[Dict[str, Any]]) -> List[str]:
        if not steps:
            return []
        nodes = [steps[0]["from"]]
        for s in steps:
            nodes.append(s["to"])
        return nodes

    def _oriented_cols(self, edge: Any, from_table: str, to_table: str) -> Tuple[List[str], List[str]]:
        if edge.left_table == from_table and edge.right_table == to_table:
            return list(edge.left_cols), list(edge.right_cols)
        return list(edge.right_cols), list(edge.left_cols)

    def _traversal_direction(self, edge: Any, from_table: str, to_table: str) -> str:
        direction = getattr(edge, "direction", "undirected")
        if direction == "undirected":
            return "undirected"
        if direction == "left_parent":
            if from_table == edge.left_table and to_table == edge.right_table:
                return "parent_to_child"
            if from_table == edge.right_table and to_table == edge.left_table:
                return "child_to_parent"
        if direction == "right_parent":
            if from_table == edge.right_table and to_table == edge.left_table:
                return "parent_to_child"
            if from_table == edge.left_table and to_table == edge.right_table:
                return "child_to_parent"
        return "undirected"


if __name__ == "__main__":
    # Demo against JoinGraphBuilderV2 if available.
    try:
        from join_graph_builder_v2 import demo_run
    except Exception:
        demo_run = None

    if demo_run is None:
        print("Demo unavailable: join_graph_builder_v2.demo_run not found")
    else:
        edges = demo_run()
        planner = JoinPathPlannerV2(edges)

        print("\n==============================")
        print("JoinPathPlannerV2 Demo")
        print("==============================\n")

        for s, t in [
            ("customers", "order_items"),
            ("customers", "fulfillment_events"),
            ("fulfillment_events", "customers"),
        ]:
            print(f"Path: {s} -> {t}")
            p = planner.find_path(s, t, min_conf=0.5, max_hops=5)
            planner.print_path(p)
            print()

        print("Top-3 paths: customers -> fulfillment_events")
        topk = planner.find_top_k_paths("customers", "fulfillment_events", k=3, min_conf=0.5, max_hops=5)
        for i, p in enumerate(topk, 1):
            print(f"\nCandidate #{i}")
            planner.print_path(p)
