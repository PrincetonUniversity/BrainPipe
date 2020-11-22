#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: emilyjanedennis
"""

import pandas as pd

test_df = pd.read_csv(
    "/Users/emilydennis/Desktop/pooled_cell_measures/z266_cell_measures.csv")

test_df = test_df[test_df["z depth"] > 2]

test_df.to_csv(
    "/Users/emilydennis/Desktop/pooled_cell_measures/z266_gt2_cell_measures.csv")
