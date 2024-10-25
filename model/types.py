from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]

@dataclass
class Gaussian:
    means: Float[Tensor, "gaussian dim"]
    covariances: Float[Tensor, "gaussian dim dim"]
    harmonics: Float[Tensor, "gaussian 3 d_sh"]
    opacities: Float[Tensor, "gaussian"]
    scales: Float[Tensor, "gaussian 3"]
    rotations: Float[Tensor, "gaussian 4"]
