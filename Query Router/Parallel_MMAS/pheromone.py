"""
Pheromone Table — MMAS (Max-Min Ant System) bounded pheromone management.

Each Colony owns one PheromoneTable that maps worker names to pheromone
values.  All values are strictly clamped to [tau_min, tau_max] on every
update cycle.

Key MMAS properties:
  - tau_max = 1 / (rho * f_best)  where f_best is best solution quality
  - tau_min = tau_max / (2 * n)    where n is the number of workers
  - Initial pheromone = tau_max    (encourages early exploration)
  - Only the iteration-best ant deposits pheromone
  - Evaporation: tau = (1 - rho) * tau + delta_tau
"""

import math
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("mmas.pheromone")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration defaults
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MMASConfig:
    """MMAS hyperparameters."""
    rho: float = 0.1               # Evaporation rate (0 < rho < 1)
    alpha: float = 1.0             # Pheromone importance in probability
    beta: float = 2.0              # Heuristic importance in probability
    tau_max_initial: float = 1.0   # Starting tau_max before any solutions
    tau_min_ratio: float = 0.5     # tau_min = tau_max / (2 * n * ratio)
    f_best_default: float = 1.0    # Default best fitness (before first update)


# ═══════════════════════════════════════════════════════════════════════════════
# Pheromone Table
# ═══════════════════════════════════════════════════════════════════════════════

class PheromoneTable:
    """
    MMAS-bounded pheromone table for a single colony.

    Workers are identified by string names (e.g., "quran_agent").
    Thread-safe via an internal lock.
    """

    def __init__(
        self,
        worker_names: List[str],
        config: Optional[MMASConfig] = None,
    ):
        self.config = config or MMASConfig()
        self.worker_names = list(worker_names)
        self.n = len(self.worker_names)
        self._lock = threading.Lock()

        # ── Compute initial bounds ──
        self.f_best = self.config.f_best_default
        self.tau_max = self.config.tau_max_initial
        self.tau_min = self._compute_tau_min()

        # ── Initialize all trails to tau_max (MMAS: encourage exploration) ──
        self.trails: Dict[str, float] = {
            name: self.tau_max for name in self.worker_names
        }

        logger.info(
            f"PheromoneTable initialized: workers={self.worker_names}, "
            f"tau_max={self.tau_max:.4f}, tau_min={self.tau_min:.4f}"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_tau_min(self) -> float:
        """tau_min = tau_max / (2 * n) — standard MMAS lower bound."""
        if self.n == 0:
            return 0.001
        return max(self.tau_max / (2 * self.n), 0.001)

    def _clamp(self, value: float) -> float:
        """Enforce MMAS bounds: tau_min <= value <= tau_max."""
        return max(self.tau_min, min(self.tau_max, value))

    # ── Public interface ─────────────────────────────────────────────────────

    def get_trails(self) -> Dict[str, float]:
        """Return a snapshot of current pheromone trails."""
        with self._lock:
            return dict(self.trails)

    def get_probabilities(self, heuristics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute selection probabilities using the ant colony formula:
            P(j) = [tau(j)]^alpha * [eta(j)]^beta  /  sum_k [tau(k)]^alpha * [eta(k)]^beta

        Args:
            heuristics: optional dict of worker_name -> heuristic value (eta).
                        If None, all workers get equal heuristic = 1.0.

        Returns:
            Dict of worker_name -> probability (sums to 1.0).
        """
        with self._lock:
            alpha = self.config.alpha
            beta = self.config.beta
            scores = {}
            for name in self.worker_names:
                tau = self.trails[name]
                eta = (heuristics or {}).get(name, 1.0)
                scores[name] = (tau ** alpha) * (eta ** beta)

            total = sum(scores.values())
            if total <= 0:
                # Uniform fallback
                uniform = 1.0 / self.n
                return {name: uniform for name in self.worker_names}

            return {name: s / total for name, s in scores.items()}

    def evaporate(self) -> None:
        """
        Apply pheromone evaporation: tau = (1 - rho) * tau.
        Clamps to [tau_min, tau_max] after evaporation.
        """
        with self._lock:
            rho = self.config.rho
            for name in self.worker_names:
                self.trails[name] = self._clamp(
                    (1 - rho) * self.trails[name]
                )
            logger.debug(
                f"Evaporation applied (rho={rho}): "
                f"trails={self._format_trails()}"
            )

    def deposit(self, worker_name: str, quality: float) -> None:
        """
        Deposit pheromone for the best-performing worker.

        delta_tau = quality (Inspector score for this worker).
        Only the iteration-best should call this (enforced by Colony).
        Clamps to [tau_min, tau_max] after deposit.
        """
        if worker_name not in self.trails:
            logger.warning(f"Unknown worker '{worker_name}' — skipping deposit")
            return

        with self._lock:
            old = self.trails[worker_name]
            self.trails[worker_name] = self._clamp(old + quality)
            logger.debug(
                f"Deposit: {worker_name} += {quality:.4f} "
                f"({old:.4f} -> {self.trails[worker_name]:.4f})"
            )

    def update_bounds(self, f_best: float) -> None:
        """
        Recompute tau_max and tau_min based on new best fitness.
            tau_max = 1 / (rho * f_best)
            tau_min = tau_max / (2 * n)

        All current trails are re-clamped to the new bounds.
        """
        with self._lock:
            self.f_best = max(f_best, 0.001)  # avoid division by zero
            rho = self.config.rho
            self.tau_max = 1.0 / (rho * self.f_best)
            self.tau_min = self._compute_tau_min()

            # Re-clamp all trails
            for name in self.worker_names:
                self.trails[name] = self._clamp(self.trails[name])

            logger.info(
                f"Bounds updated: f_best={self.f_best:.4f}, "
                f"tau_max={self.tau_max:.4f}, tau_min={self.tau_min:.4f}, "
                f"trails={self._format_trails()}"
            )

    def merge_from(self, other_trails: Dict[str, float], weight: float = 0.3) -> None:
        """
        Inter-colony synchronization: blend another colony's trails into this one.

        new_tau(j) = (1 - weight) * self.tau(j) + weight * other.tau(j)
        """
        with self._lock:
            for name in self.worker_names:
                if name in other_trails:
                    blended = (1 - weight) * self.trails[name] + weight * other_trails[name]
                    self.trails[name] = self._clamp(blended)
            logger.debug(
                f"Inter-colony merge (weight={weight}): "
                f"trails={self._format_trails()}"
            )

    def snapshot(self) -> dict:
        """Return a serializable snapshot of the table state."""
        with self._lock:
            return {
                "trails": dict(self.trails),
                "tau_max": self.tau_max,
                "tau_min": self.tau_min,
                "f_best": self.f_best,
            }

    def _format_trails(self) -> str:
        return ", ".join(f"{k}={v:.4f}" for k, v in self.trails.items())
