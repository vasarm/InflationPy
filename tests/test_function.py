import sympy as sp

from inflationpy.core.functions import InflationFunction


def test_initiation_from_string():
    function = InflationFunction("phi**2 + 2")
    phi = sp.Symbol("phi", real=True)
    sympy_function = phi**2 + 2
    assert function.f_s() == sympy_function


def test_inition_from_expression_with_spcification_for_symbol():
    phi = sp.Symbol("phi", real=True)
    sympy_function = phi**2 + 2
    function = InflationFunction(phi**2 + 2)
    assert function.f_s() == sympy_function


def test_inition_from_expression_without_spcification_for_symbol():
    phi = sp.Symbol("phi", real=True)
    phi2 = sp.Symbol("phi")
    sympy_function = phi**2 + 2
    function = InflationFunction(phi2**2 + 2)
    assert function.f_s() == sympy_function


"""
    Testing calculation
"""
# Test for numpy and mpmath derivatives and functions
# Check if x is array for scipy mode then result is always array. Problem, if function is scalar.


def test_simple_numpy_calculation():
    phi = sp.Symbol("phi", real=True)
    function = InflationFunction(phi**2 + 2)
    assert function.f_n(x=2, mode="scipy") == 6


def test_simple_mpmath_calculation():
    phi = sp.Symbol("phi", real=True)
    function = InflationFunction(phi**2 + 2)
    assert function.f_n(x=2, mode="mpmath") == 6


"""
    Emulating numerical calculation
"""
# Tests for __add__ , __sub__, __truediv__, __pow__, __mul__ between InflationFunction types, int, float and sp.Expr as well
