# -*- coding: utf-8 -*-

import numpy as np

def accuracy(actual, predicted):
    for item in predicted:
        if item in actual:
            return 1.0
    return 0.0