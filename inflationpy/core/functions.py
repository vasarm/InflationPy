from abc import ABC, abstractmethod
from pyclbr import Function
from typing import Dict, List, Optional, Union
from numpy import isin

import sympy as sp
import numpy as np


class FunctionClass(ABC):
    def __init__(
        self,
        function: Union[sp.Expr, str],
        symbol: Union[str, sp.Symbol] = sp.Symbol("phi", real=True),
    ):
        """_summary_

        Parameters
        ----------
        function : Union[sp.Expr, str]
            Analytical function represented with strings. Used to define function in symbolic form.
        symbol : str, optional
            Symbol to represent scalar field in analytical functions, by default "phi"
        """

        self.symbol = symbol

        # Analytical functions in sympy
        self.analytical_function = function
        self.analytical_derivative = None
        self.analytical_2_derivative = None

        # Python functions to be used by numpy (converted from sympy function where module='scipy')
        self.numerical_function_np = None
        self.numerical_derivative_np = None
        self.numerical_2_derivative_np = None

        # Python functions to be used by mpmath (converted from sympy function where module='mpmath')
        self.numerical_function_mp = None
        self.numerical_derivative_mp = None
        self.numerical_2_derivative_mp = None

        self._sorted_variables: Union[List[sp.Symbol], None] = None

    def __repr__(self):
        return str(self.analytical_function)

    def _repr_latex_(self):
        string = f"${str(sp.latex(self.analytical_function))}$"
        return string

    @property
    def symbol(self):
        return self._symbol

    @property
    def analytical_function(self):
        return self._analytical_function

    @property
    def analytical_derivative(self):
        return self._analytical_derivative

    @property
    def analytical_2_derivative(self):
        return self._analytical_2_derivative

    @property
    def numerical_function_np(self):
        return self._numerical_function_np

    @property
    def numerical_derivative_np(self):
        return self._numerical_derivative_np

    @property
    def numerical_2_derivative_np(self):
        return self._numerical_2_derivative_np

    @property
    def numerical_function_mp(self):
        return self._numerical_function_mp

    @property
    def numerical_derivative_mp(self):
        return self._numerical_derivative_mp

    @property
    def numerical_2_derivative_mp(self):
        return self._numerical_2_derivative_mp

    @symbol.setter
    def symbol(self, symbol):
        """Set symbol for scalar field.

        Parameters
        ----------
        symbol : sp.Symbol, str
            Inserted symbol for scalar field. If string, then use sympy to convert to sp.Symbol.
            Otherwise use same symbol. Always sets notion in sympy that symbol is real.
        Raises
        ------
        TypeError
            _description_
        """
        if isinstance(symbol, sp.Symbol):
            self._symbol = sp.Symbol(str(symbol), real=True)
        elif isinstance(symbol, str):
            self._symbol = sp.Symbol(symbol, real=True)
        else:
            raise TypeError("Symbol must be string on sp.Symbol type")

    @analytical_function.setter
    def analytical_function(self, function):
        if "_analytical_function" in self.__dict__:
            raise ValueError("Can't redefine already defined analytical function.")
        elif "_numerical_function" in self.__dict__:
            raise ValueError("Can't add analytical function if numerical already exists.")
        elif isinstance(function, sp.Expr):  # If sympy function
            for variable in function.free_symbols:
                if str(variable) == str(self.symbol):
                    function = function.subs(variable, self.symbol)
                else:
                    function = function.subs(variable, sp.Symbol(str(variable), real=True, positive=True))
            self._analytical_function = function
        elif isinstance(function, str):
            try:
                self._analytical_function = self._convert_string_to_function(function)
            except Exception as e:
                raise RuntimeError("Problem with converting string to sympy function\n{}".format(e))
        else:
            raise ValueError("Invalid type for analytical function.")

    @analytical_derivative.setter
    def analytical_derivative(self, function):
        if function is None:
            if isinstance(self.analytical_function, sp.Expr):
                self._analytical_derivative = sp.diff(self.analytical_function, self.symbol)
        else:
            raise ValueError("Can't redefine already defined analytical derivative.")

    @analytical_2_derivative.setter
    def analytical_2_derivative(self, function):
        if function is None:
            if isinstance(self.analytical_function, sp.Expr):
                self._analytical_2_derivative = sp.diff(self.analytical_function, self.symbol, 2)
        else:
            raise ValueError("Can't redefine already defined analytical derivative.")

    @numerical_function_np.setter
    def numerical_function_np(self, function):
        if "_numerical_function_np" in self.__dict__:
            raise ValueError("Can't redefine already defined numerical function.")
        elif function is None:
            variables = self._return_ordered_symbols()
            self._numerical_function_np = sp.lambdify(variables, self.analytical_function, "numpy")

    @numerical_derivative_np.setter
    def numerical_derivative_np(self, function):
        if "_numerical_derivative_np" in self.__dict__:
            raise ValueError("Can't redefine already defined numerical function.")
        elif function is None:
            variables = self._return_ordered_symbols()
            self._numerical_derivative_np = sp.lambdify(variables, self.analytical_derivative, "numpy")

    @numerical_2_derivative_np.setter
    def numerical_2_derivative_np(self, function):
        if "_numerical_2_derivative_np" in self.__dict__:
            raise ValueError("Can't redefine already defined numerical function.")
        elif function is None:
            variables = self._return_ordered_symbols()
            self._numerical_2_derivative_np = sp.lambdify(variables, self.analytical_2_derivative, "numpy")

    @numerical_function_mp.setter
    def numerical_function_mp(self, function):
        if "_numerical_function_mp" in self.__dict__:
            raise ValueError("Can't redefine already defined numerical function.")
        elif function is None:
            variables = self._return_ordered_symbols()
            self._numerical_function_mp = sp.lambdify(variables, self.analytical_function, "mpmath")

    @numerical_derivative_mp.setter
    def numerical_derivative_mp(self, function):
        if "_numerical_derivative_mp" in self.__dict__:
            raise ValueError("Can't redefine already defined numerical function.")
        elif function is None:
            variables = self._return_ordered_symbols()
            self._numerical_derivative_mp = sp.lambdify(variables, self.analytical_derivative, "mpmath")

    @numerical_2_derivative_mp.setter
    def numerical_2_derivative_mp(self, function):
        if "_numerical_2_derivative_mp" in self.__dict__:
            raise ValueError("Can't redefine already defined numerical function.")
        elif function is None:
            variables = self._return_ordered_symbols()
            self._numerical_2_derivative_mp = sp.lambdify(variables, self.analytical_2_derivative, "mpmath")

    def f_s(self, parameters: Optional[Dict[Union[sp.Expr, str], Union[float, str]]] = None):
        """Returns analytical function. If parameters dictionary is defined then substitute inserted symbols with given values.

        Parameters
        ----------
        parameters : Optional[Dict[Union[sp.Expr, str], Union[float, str]]], optional
            Dictionary to substitute symbols in analytical function, by default None
        """
        if parameters is None:
            parameters = {}
        return self.analytical_function.subs(parameters)

    def fd_s(self, parameters: Optional[Dict[Union[sp.Expr, str], Union[float, str]]] = None):
        """Returns analytical derivative of a function. If parameters dictionary is defined then substitute inserted symbols with given values.

        Parameters
        ----------
        parameters : Optional[Dict[Union[sp.Expr, str], Union[float, str]]], optional
            Dictionary to substitute symbols in analytical function, by default None
        """
        if parameters is None:
            parameters = {}
        return self.analytical_derivative.subs(parameters)  # type: ignore

    def f2d_s(self, parameters: Optional[Dict[Union[sp.Expr, str], Union[float, str]]] = None):
        """Returns analytical second derivative of a function. If parameters dictionary is defined then substitute inserted symbols with given values.

        Parameters
        ----------
        parameters : Optional[Dict[Union[sp.Expr, str], Union[float, str]]], optional
            Dictionary to substitute symbols in analytical function, by default None
        """
        if parameters is None:
            parameters = {}
        return self.analytical_2_derivative.subs(parameters)  # type: ignore

    @abstractmethod
    def f_n(self, x, mode: str, *args, **kwargs):
        """
        Calculate numerical values of the function in points 'x'.

        Parameters
        ----------
        mode : str
            Select which numerical package to use.

        *args:
            Values for function constant parameters.
        **kwargs:
            Dictionary of function constant parameters
        """

    @abstractmethod
    def fd_n(self, x, mode: str, *args, **kwargs):
        """
        Calculate numerical values of the function's derivative in points 'x'.

        Parameters
        ----------
        mode : str
           Select which numerical package to use.
        *args:
            Values for function constant parameters.
        **kwargs:
            Dictionary of function constant parameters
        """

    @abstractmethod
    def f2d_n(self, x, mode: str, *args, **kwargs):
        """
        Calculate numerical values of the function's second derivatve in points 'x'.

        Parameters
        ----------
        mode : str
           Select which numerical package to use.
        *args:
            Values for function constant parameters.
        **kwargs:
            Dictionary of function constant parameters
        """

    def _convert_string_to_function(self, string: str) -> sp.Expr:
        """
        Converts string into sympy expression. Also sets flags real and positive to True for each variable.
        """
        result: sp.Expr = sp.sympify(string)  # type: ignore
        for variable in result.free_symbols:
            if str(variable) == str(self.symbol):
                result = result.subs(variable, self.symbol)
            else:
                result = result.subs(variable, sp.Symbol(str(variable), real=True, positive=True))

        return result

    def _return_ordered_symbols(self) -> List[sp.Symbol]:
        if "_sorted_variables" in self.__dict__ and self.__dict__["_sorted_variables"] is not None:
            return self._sorted_variables  # type: ignore
        else:
            symbols = list(self.analytical_function.free_symbols - {self.symbol})
            symbols.sort(key=lambda x: str(x))
            result = [self.symbol, *symbols]
            self._sorted_variables = result
            return result


class InflationFunction(FunctionClass):
    def __init__(
        self,
        function: Union[sp.Expr, str],
        symbol: Union[str, sp.Symbol] = sp.Symbol("phi", real=True),
    ):

        super().__init__(function=function, symbol=symbol)

    def __add__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(self.analytical_function + other_function, symbol=self.symbol)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        self.analytical_function + other_function.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"Both function scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("All terms must be InflationFunction, int or float type.")

    def __sub__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(self.analytical_function - other_function, symbol=self.symbol)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        self.analytical_function - other_function.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"All factors' scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("All terms must be InflationFunction, int or float type.")

    def __mul__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(self.analytical_function * other_function, symbol=self.symbol)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        self.analytical_function * other_function.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"All factors' scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("All factors must be InflationFunction, int or float type.")

    def __truediv__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(self.analytical_function / other_function)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        self.analytical_function / other_function.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"Numerator and denominator scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("Numerator and denominator must be InflationFunction, int or float type.")

    def __pow__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(self.analytical_function**other_function, symbol=self.symbol)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        self.analytical_function**other_function.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"Base and exponent scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("Base and exponent must be InflationFunction, int or float type.")

    def __radd__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(self.analytical_function + other_function, symbol=self.symbol)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        self.analytical_function + other_function.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"Both function scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("All terms must be InflationFunction, int or float type.")

    def __rsub__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(other_function - self.analytical_function, symbol=self.symbol)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        other_function.analytical_function - self.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"All factors' scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("All terms must be InflationFunction, int or float type.")

    def __rmul__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(other_function * self.analytical_function, symbol=self.symbol)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        other_function.analytical_function * self.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"All factors' scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("All factors must be InflationFunction, int or float type.")

    def __rtruediv__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(other_function / self.analytical_function)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        other_function.analytical_function / self.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"Numerator and denominator scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("Numerator and denominator must be InflationFunction, int or float type.")

    def __rpow__(self, other_function):
        if isinstance(other_function, (InflationFunction, int, float, sp.Expr)):
            if isinstance(other_function, (int, float, sp.Expr)):
                return InflationFunction(other_function**self.analytical_function, symbol=self.symbol)
            else:
                if self.symbol == other_function.symbol:
                    return InflationFunction(
                        other_function.analytical_function**self.analytical_function, symbol=self.symbol
                    )
                else:
                    raise ValueError(
                        f"Base and exponent scalar field symbols must be the same. ({self.symbol, other_function.symbol})"
                    )
        else:
            raise TypeError("Base and exponent must be InflationFunction, int or float type.")

    def f_n(self, x, mode: str = "scipy", *args, **kwargs):
        """
        Calculate numerical values of the function in points 'x'.
        If mode is scipy/numpy then numpy arrays can be used for calculation. If mode is mpmath then arrays are not supported.


        Parameters
        ----------
        x : int, float, np.ndarray, mp.mpf
            Function will be evaluated in the position of "x".
        mode : str, optional
            Select which numerical package to use. Possible values are "numpy"/"scipy" or "mpmath", by default "scipy"
        *args:
            Values for function constant parameters.
        **kwargs:
            Dictionary of function constant parameters
        Returns
        -------
        int, float, np.ndarray, (mp.mpf)
            Return numerical value(s) of the function evaluated in the point(s) "x". If mode is mpmath then type can be mp.mpf as well.
        """

        if mode.strip() in ["scipy", "numpy"]:
            result = self.numerical_function_np(x, *args, **kwargs)
            if isinstance(x, np.ndarray) and isinstance(result, (int, float)):
                result = np.full(x.shape, result)
            return result
        elif mode == "mpmath":
            return self.numerical_function_mp(x, *args, **kwargs)
        else:
            raise ValueError("Mode can be 'scipy' or 'mpmath'.")

    def fd_n(self, x, mode: str = "scipy", *args, **kwargs):
        if mode.strip() in ["scipy", "numpy"]:
            result = self.numerical_derivative_np(x, *args, **kwargs)
            if isinstance(x, np.ndarray) and isinstance(result, (int, float)):
                result = np.full(x.shape, result)
            return result
        elif mode == "mpmath":
            return self.numerical_derivative_mp(x, *args, **kwargs)
        else:
            raise ValueError("Mode can be 'scipy' or 'mpmath'.")

    def f2d_n(self, x, mode: str = "scipy", *args, **kwargs):
        if mode.strip() in ["scipy", "numpy"]:
            result = self.numerical_2_derivative_np(x, *args, **kwargs)
            if isinstance(x, np.ndarray) and isinstance(result, (int, float)):
                result = np.full(x.shape, result)
            return result
        elif mode == "mpmath":
            return self.numerical_2_derivative_mp(x, *args, **kwargs)
        else:
            raise ValueError("Mode can be 'scipy' or 'mpmath'.")


"""
if mode == "scipy":
    return self.numerical_function_np(**kwargs)
elif mode == "mpmath":
    return self.numerical_function_mp(**kwargs)
else:
    raise ValueError("Mode can only be 'scipy' or 'mpmath'.")
"""
