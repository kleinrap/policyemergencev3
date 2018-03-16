import pysd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# The line below is only needed when the model needs to be imported into Python.
model_flood = pysd.read_vensim('Flood_Levees_14_Final.mdl')

## The following are the elevent parameters 

# Notes:
# To output specific data, use: return_columns=['',''] filling in with the parameter names
# To input specific data, use: params={'':, '':}
# To start at a different point or to start from a freeze of the model, use: initial_condition='current', return_timestamps=range(20,40)
stocks = model_flood.run(return_columns=['safety owing to levee quality', 'perceived current safety'], params={'construction time':10, 'FINAL TIME':5})

stocks2 = model_flood.run(return_columns=['safety owing to levee quality', 'perceived current safety'], params={'TIME STEP':0.0078125}, initial_condition='current', return_timestamps=range(5,20))

print(stocks)
stocks.plot()
plt.xlim((0,20))
plt.ylim((0,1))


print(stocks2)
stocks2.plot()
plt.xlim((0,20))
plt.ylim((0,1))

plt.show()