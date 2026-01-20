"""
Helper functions for data collection during optimization.
"""

import json
import random
from typing import List

from src.data.collector import (
    ZoningCollector, StateRow, ActionRow,
    build_features_multizone, try_op
)


def mutate_layer_logged(
    zoning,                     
    collector: ZoningCollector, 
    run_id: str,                # RunMeta.run_id
    instance_id: str,           # Identifier for this grid instance (size/pattern/seed)
    layer_id: int,              
    attempts: int,             
    zone_pattern: str,          
    zone_params: dict,          
    num_zones: int,             
    allow_flip: bool = True,    
    Z_MAX: int = 6,             
    add_dist: bool = False      
) -> bool:
    
    positions = zoning.get_layer_positions(layer_id)
    random.shuffle(positions)

    tvars = list(zoning.h.transpose_patterns.keys())

    flip = {'n': (3, 2), 's': (3, 2), 'e': (2, 3), 'w': (2, 3)}

    for _ in range(attempts):
        feats = build_features_multizone(zoning.h, zoning.zones, layer_id, Z_MAX=Z_MAX, add_dist=add_dist)
        sample_id, feats_path = collector.save_features(feats)
        beforeC = zoning.compute_crossings()

        State_row = StateRow(
            sample_id=sample_id,
            run_id=run_id,
            instance_id=instance_id,
            step_t=getattr(zoning,"step_t", 0),
            layer_id=layer_id,
            grid_w=zoning.W,
            grid_h=zoning.Ht,
            num_zones=num_zones,
            zone_pattern=zone_pattern,
            zone_params=json.dumps(zone_params or {}),
            crossings_before=beforeC,
            features_file=feats_path
        )
        collector.log_state(State_row)

        # Action Trials
        action_rows: List[ActionRow] = []
        # Try all transposes at each layer
        for (x, y) in positions:
            for var in tvars:
                valid, bC, aC, dC = try_op(
                    zoning.h, zoning.compute_crossings,
                    op="transpose", x=x, y=y, subgrid_kind="3x3", variant=var
                )
                R = collector.reward(dC, valid)
                action_rows.append(ActionRow(
                    sample_id=sample_id, x=x, y=y, subgrid_kind="3x3",
                    orientation=var, op="transpose", valid=int(valid),
                    crossings_before=int(aC), crossings_after=int(aC), 
                    delta_cross=int(dC),
                    reward=float(R), best_in_state=0
                ))

        # Try flips if allowed
        if allow_flip:
            for (x, y) in positions:
                for var, (w, h) in flip.items():
                    kind = "3x2" if (w, h) == (3, 2) else "2x3"
                    valid, bC, aC, dC = try_op(
                        zoning.h, zoning.compute_crossings,
                        op="flip", x=x, y=y, subgrid_kind=kind, variant=var
                    )
                    R = collector.reward(dC, valid)
                    action_rows.append(ActionRow(
                        sample_id=sample_id, x=x, y=y, subgrid_kind=kind,
                        orientation=var, op="flip", valid=int(valid),
                        crossings_before=int(aC), crossings_after=int(aC), 
                        delta_cross=int(dC),
                        reward=float(R), best_in_state=0
                    ))

        collector.log_actions(action_rows)

        # Best Valid
        best_valid = [ar for ar in action_rows if ar.valid == 1]
        if best_valid:
            best = max(best_valid, key=lambda r: r.reward)
            if best.op == "transpose":
                sub = zoning.h.get_subgrid((best.x, best.y), (best.x + 2, best.y + 2))
                zoning.h.transpose_subgrid(sub, best.orientation)
            else:
                w, h = (3, 2) if best.subgrid_kind == "3x2" else (2, 3)
                sub = zoning.h.get_subgrid((best.x, best.y), (best.x + w - 1, best.y + h - 1))
                zoning.h.flip_subgrid(sub, best.orientation)

            zoning.step_t = getattr(zoning, "step_t", 0) + 1
            return True
    return False
