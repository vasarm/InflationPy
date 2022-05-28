
# InflationPy


InflationPy is a package which uses inflation formulation in scalar–tensor theory to make calculations.

Theory contains three free functions $A(\phi)$, $B(\phi)$ and $V(\phi)$ which are used in action

![equation](https://latex.codecogs.com/svg.image?S&space;=&space;\int&space;d^4x&space;\sqrt{-g}&space;[&space;\frac{M_p^2}{2}&space;A(\phi)R&space;&space;-\frac{1}{2}B(\phi)g^{\mu&space;\nu}&space;\nabla_{\mu}&space;\phi&space;\nabla_{\nu}&space;\phi&space;-&space;&space;V(\phi)&space;&space;&space;]&space;\&space;.)

Action can be formulated by invariant quantities as (in Einstein frame)

![equation](https://latex.codecogs.com/svg.image?S&space;=&space;\int&space;d^4x&space;\sqrt{-g}&space;[&space;\frac{M_p^2}{2}&space;R&space;&space;-\frac{1}{2}g^{\mu&space;\nu}&space;\nabla_{\mu}&space;I_\phi&space;\nabla_{\nu}&space;I_\phi&space;-&space;&space;I_V(I_\phi)]&space;\&space;.)


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
