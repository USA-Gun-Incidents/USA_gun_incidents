# -*- coding: utf-8 -*-
# %%
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('..\\')) # TODO: c'è un modo per farlo meglio?
import plot_utils
import numpy as np

# read data
dirname = os.path.dirname(' ')
FOLDER = os.path.join(dirname, 'data')
incidents_path = os.path.join(FOLDER, 'incidents.csv')
incidents_data = pd.read_csv(incidents_path)

#TODO: controllare a mano quelli che il primo controllo segna non consistenti e cercare di migliorare
