
# InflationPy


InflationPy is a package which uses inflation formulation in scalar–tensor theory to make calculations.

Theory contains three free functions $A(\phi)$, $B(\phi)$ and $V(\phi)$ which are used in action

$S = \int d^4x \sqrt{-g} [ \frac{M_p^2}{2} A(\phi)R  -\frac{1}{2}B(\phi)g^{\mu \nu} \nabla_{\mu} \phi \nabla_{\nu} \phi -  V(\phi) ]$



Action can be formulated by invariant quantities as (in Einstein frame)

$S = \int d^4x \sqrt{-g} [ \frac{M_p^2}{2} R  -\frac{1}{2}g^{\mu \nu} \nabla_{\mu} I_\phi \nabla_{\nu} I_\phi -  I_V(I_\phi) ]$



Problems what currently can be solved:

1. Find model predictions for observables ($n_s$ - scalar spectral index, 'r' tensor-to-scalar ratio)
2. Given two functions from (A, B, V) and also invariant potential then find third function.
3. Einstein and Jordan frame N-fold difference
4. Comapre model defined with A, B, V functions and model defined withinvariant potential.


## Installation

### Install using pip
Install the InflationPy package:
```
pip install inflationpy
```
