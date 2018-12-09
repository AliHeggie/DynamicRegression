"""
Tests for dynamic regression time series models

Author: Alastair Heggie
License: Simplified-BSD
"""

"""
Possible tests:


Write down what I think the design/transition/selection/distrubance covarnace
and noise covariance matrices should be for a variety of processes.

write a method like run_ucm from test_structural that creates a DSR model
based on agruments passed describing exog model

Get results of simple model, say local level + drift using KFAS or simlar,
compare. Would need to create a class similar to results_structural



"""
from __future__ import division, absolute_import, print_function
import warnings

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest

import DynamicRegression

def test_True():
    a = 1+1
    assert_equal(a,2)

def test_False():
    a = 1+1
    assert_equal(a,3)