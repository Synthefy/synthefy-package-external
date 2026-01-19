from typing import Optional

from synthefy_pkg.configs.tabicl_config import (
    AssignmentConfig,
    TabICLCurriculumConfig,
    TabICLPriorConfig,
)
from synthefy_pkg.prior.curriculum.adaptive_curriculum import (
    AdaptiveCurriculum,
    FloorAdaptiveCurriculum,
)
from synthefy_pkg.prior.curriculum.simple_curriculum import (
    Curriculum,
    LinearCurriculum,
)


def get_curriculum_manager(
    curriculum_config: TabICLCurriculumConfig,
    config: TabICLPriorConfig,
    assignment_config: Optional[AssignmentConfig] = None,
) -> Optional[Curriculum]:
    if curriculum_config.curriculum_type == "linear":
        return LinearCurriculum(curriculum_config, config, assignment_config)
    elif curriculum_config.curriculum_type == "adaptive":
        return AdaptiveCurriculum(curriculum_config, config, assignment_config)
    elif curriculum_config.curriculum_type == "floor_adaptive":
        return FloorAdaptiveCurriculum(
            curriculum_config, config, assignment_config
        )
    elif curriculum_config.curriculum_type == "none":
        return None
    else:
        raise ValueError(
            f"Unknown curriculum type: {curriculum_config.curriculum_type}"
        )
