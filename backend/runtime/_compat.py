"""Compatibility layer: backend.runtime.* -> core_runtime.* re-exports.

All old import paths continue to work through these shims.
Deprecation warnings are emitted but no functionality breaks.
"""

import warnings

# backend.runtime.projection_scheduler -> core_runtime.core.runtime.scheduling
try:
    from core_runtime.core.scheduling.projection_scheduler import *  # noqa: F401 F403
except ImportError:
    pass

# backend.runtime.trajectory_analysis -> core_runtime.core.scheduling
try:
    from core_runtime.core.scheduling.trajectory_analysis import *  # noqa: F401 F403
except ImportError:
    pass

# backend.runtime.execution_scheduler -> core_runtime.core.routing
try:
    from core_runtime.core.routing.execution_scheduler import *  # noqa: F401 F403
except ImportError:
    pass

# backend.runtime.cost_estimator -> core_runtime.core.scheduling
try:
    from core_runtime.core.scheduling.cost_estimator import *  # noqa: F401 F403
except ImportError:
    pass

# backend.runtime.exact_cache -> core_runtime.core.memory
try:
    from core_runtime.core.memory.exact_cache import *  # noqa: F401 F403
except ImportError:
    pass

# backend.runtime.retrieval_memory -> core_runtime.core.memory
try:
    from core_runtime.core.memory.retrieval_memory import *  # noqa: F401 F403
except ImportError:
    pass

# backend.runtime.warmstart_runtime -> core_runtime.core.scheduling
try:
    from core_runtime.core.scheduling.warmstart_runtime import *  # noqa: F401 F403
except ImportError:
    pass

# backend.runtime.operational_experience_schema -> core_runtime.core.experience
try:
    from core_runtime.core.experience.operational_experience_schema import *  # noqa: F401 F403
except ImportError:
    pass

# backend.runtime.experience_dataset_schema -> core_runtime.core.experience
try:
    from core_runtime.core.experience.experience_dataset_schema import *  # noqa: F401 F403
except ImportError:
    pass

warnings.warn(
    "Importing from backend.runtime.* is deprecated. "
    "Use core_runtime.core.* instead. "
    "Compatibility shims will be removed in v3.1+.",
    DeprecationWarning,
    stacklevel=2,
)
