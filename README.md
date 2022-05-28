
# InflationPy


InflationPy is a package which uses inflation formulation in scalar–tensor theory to make calculations.

Theory contains three free functions $A(\phi)$, $B(\phi)$ and $V(\phi)$ which are used in action

![equation]([https://latex.codecogs.com/svg.image?&space;S&space;=&space;\int&space;d^4x&space;\sqrt{-g}&space;\left\{&space;\frac{M_p^2}{2}&space;A(\phi)R&space;&space;-\frac{1}{2}B(\phi)g^{\mu&space;\nu}&space;\nabla_{\mu}&space;\phi&space;\nabla_{\nu}&space;\phi&space;-&space;&space;V(\phi)&space;&space;\right&space;\}&space;\&space;.](https://latex.codecogs.com/svg.image?S&space;=&space;%5Cint&space;d%5E4x&space;%5Csqrt%7B-g%7D&space;%5Cleft%5C%7B&space;%5Cfrac%7BM_p%5E2%7D%7B2%7D&space;A(%5Cphi)R&space;&space;-%5Cfrac%7B1%7D%7B2%7DB(%5Cphi)g%5E%7B%5Cmu&space;%5Cnu%7D&space;%5Cnabla_%7B%5Cmu%7D&space;%5Cphi&space;%5Cnabla_%7B%5Cnu%7D&space;%5Cphi&space;-&space;&space;V(%5Cphi)&space;&space;%5Cright&space;%5C%7D&space;%5C&space;.))

$ S = \int d^4x \sqrt{-g} \left\{ \frac{M_p^2}{2} A(\phi)R  -\frac{1}{2}B(\phi)g^{\mu \nu} \nabla_{\mu} \phi \nabla_{\nu} \phi -  V(\phi)  \right \} \ .$

Action can be formulated by invariant quantities as (in Einstein frame)

$$ S = \int d^4x \sqrt{-g} \left\{ \frac{M_p^2}{2} R  -\frac{1}{2}g^{\mu \nu} \nabla_{\mu} I_\phi \nabla_{\nu} I_\phi -  I_V(I_\phi)  \right \} \ .$$

Problems what currently can be solved:

1. Find model predictions for observables ($n_s$ - scalar spectral index, 'r' tensor-to-scalar ratio)
2. Given two functions from (A, B, V) and also invariant potential then find third function.
3. Einstein and Jordan frame N-fold difference
4. Comapre model defined with A, B, V functions and model defined withinvariant potential.
