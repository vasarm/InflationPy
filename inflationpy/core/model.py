from typing import Dict, List, Optional, Union

import sympy as sp

from inflationpy.core.functions import InflationFunction


class InflationModel:
    def __init__(
        self,
        A: Union[sp.Expr, str, None],
        B: Union[sp.Expr, str, None],
        V: Union[sp.Expr, str, None],
        I_V: Union[sp.Expr, str, None],
        symbol: Union[str, sp.Symbol] = sp.Symbol("phi", real=True),
    ) -> None:

        if A is None:
            self.A is None
        else:
            self.A = InflationFunction(A, symbol=symbol)

        if B is None:
            self.B is None
        else:
            self.B = InflationFunction(B, symbol=symbol)

        if V is None:
            self.V is None
        else:
            self.V = InflationFunction(V, symbol=symbol)

        if I_V is None:
            self.I_V is None
        else:
            self.I_V = InflationFunction(I_V, symbol=symbol)

    def _init_functions(self):
        """
        Define all analytical functions required for calculations:
        F, Ɛ, η, n_s, r, A_s
        """
        pass
