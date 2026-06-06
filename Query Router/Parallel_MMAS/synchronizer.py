"""
Synchronizer — Periodic inter-colony pheromone synchronization.

Blends pheromone trails between colonies so that knowledge gained
by one colony's worker selection can influence other colonies over time.

This implements a lightweight "global pheromone sharing" mechanism
where each colony merges a weighted portion of every other colony's trails.
"""

import logging
from typing import List

logger = logging.getLogger("mmas.synchronizer")


def synchronize_colonies(
    colonies: List,
    weight: float = 0.2,
) -> None:
    """
    Pairwise inter-colony trail synchronization.

    Each colony absorbs a fraction `weight` of every other colony's
    pheromone trails.  Only colonies sharing the same worker names
    actually exchange meaningful data (non-domain workers are ignored).

    Args:
        colonies: list of Colony objects to synchronize
        weight:   blending weight (0 = no sync, 1 = full overwrite)
    """
    n = len(colonies)
    if n < 2:
        return

    # Take snapshots before modification to avoid order-dependent bias
    snapshots = [(c, c.get_trails()) for c in colonies]

    merge_count = 0
    for i, (colony_i, _) in enumerate(snapshots):
        for j, (_, trails_j) in enumerate(snapshots):
            if i == j:
                continue
            colony_i.pheromone.merge_from(trails_j, weight=weight / (n - 1))
            merge_count += 1

    logger.info(
        f"Inter-colony sync complete: {n} colonies, "
        f"{merge_count} merges, weight={weight:.2f}"
    )
    print(
        f"   [Synchronizer] -> Synced {n} colonies "
        f"({merge_count} pheromone merges, weight={weight:.2f})"
    )
