import random
import copy
import pysd

'''''
What are the states that are needed from the technical model?

Secondary beliefs (S)
1. Ageing time - endogenous (AT_state)
2. Obscolescence time - endogenous (OT_state)
3. Design time - endogenous (DT_state)
4. Flood perception time - endogenous (FPT_state)
5. Effects on renovation and construction - endogenous (ERC_state)
6. Renovation time - endogenous (RT_state)
7. Adjustment time - endogenous (AdT_state)
8. Planning horizon - endogenous (PH_state)
9. Renovation standard - endogenous (RS_state)
10. Construction time - endogenous (CT_state)

Mid-level beliefs (ML)
11. ML1_state
12. ML2_state
13. ML3_state
14. ML4_state

Policy core beliefs (PC)
15. Flood perception spending - calculated (FPS_state)
16. Safety - calculated (Sa_state)
'''''

def states_definition(model_technical, states_technical):

	'''''
	This function is used to assign the values from the technical model to their respective dictionnary.
	'''''

	# Assigning the values to the right dictionnary parameters
	states_technical["AT_state"] = model_technical.components.aging_time()
	states_technical["OT_state"] = model_technical.components.obsolescence_time()
	states_technical["DT_state"] = model_technical.components.design_time()
	states_technical["FPT_state"] = model_technical.components.flood_perception_time()
	# CHANGE THIS! - The ERC is currently the actual state of the model while it was intended to take the first value of the lookup function
	states_technical["ERC_state"] = model_technical.components.effect_on_renovation_and_construction()
	states_technical["RT_state"] = model_technical.components.renovation_time()
	states_technical["AdT_state"] = model_technical.components.adjustment_time()
	states_technical["PH_state"] = model_technical.components.planning_horizon()
	states_technical["RS_state"] = model_technical.components.renovation_standard()
	states_technical["CT_state"] = model_technical.components.construction_time()
	# CHANGE THIS! - The ML beliefs need to be changed still
	states_technical["ML1_state"] = 0
	states_technical["ML2_state"] = 0
	states_technical["ML3_state"] = 0
	states_technical["ML4_state"] = 0

	# CHANGE THIS! - The PC beliefs need to be changed still
	states_technical["FPS_state"] = 0
	states_technical["Sa_state"] = 0

	return states_technical

def states_calculation(states_technical, emergence_states):

	'''''
	This function is used to calculate the states into a -1,1 interval from the states obtained in the technical model.
	'''''

	# Calculation of the ageing time
	min_AT = 1
	max_AT = 100
	emergence_states["AT_state"] = ((states_technical["AT_state"] / (max_AT-min_AT)) * 2) - 1

	# Calculation of the obsolescence time
	min_OT = 10
	max_OT = 500
	emergence_states["OT_state"] = ((states_technical["OT_state"] / (max_OT-min_OT)) * 2) - 1

	# Calculation of the design time
	min_DT = 0.5
	max_DT = 10
	emergence_states["DT_state"] = ((states_technical["DT_state"] / (max_DT-min_DT)) * 2) - 1

	# Calculation of the flood perception time
	min_FPT = 0
	max_FPT = 3
	emergence_states["FPT_state"] = ((states_technical["FPT_state"] / (max_FPT-min_FPT)) * 2) - 1

	# Calculation of the effects on renovation and construction
	min_ERC = 0
	max_ERC = 20
	emergence_states["ERC_state"] = ((states_technical["ERC_state"] / (max_ERC-min_ERC)) * 2) - 1

	# Calculation of the renovation time
	min_RT = 0.5
	max_RT = 20
	emergence_states["RT_state"] = ((states_technical["RT_state"] / (max_RT-min_RT)) * 2) - 1

	# Calculation of the adjustment time
	min_AdT = 1
	max_AdT = 200
	emergence_states["AdT_state"] = ((states_technical["AdT_state"] / (max_AdT-min_AdT)) * 2) - 1

	# Calculation of the planning horizon
	min_PH = 10
	max_PH = 200
	emergence_states["PH_state"] = ((states_technical["PH_state"] / (max_PH-min_PH)) * 2) - 1

	# Calculation of the renovation standard
	min_RS = 0.05
	max_RS = 1
	emergence_states["RS_state"] = ((states_technical["RS_state"] / (max_RS-min_RS)) * 2) - 1

	# Calculation of the construction time
	min_CT = 0.5
	max_CT = 15
	emergence_states["CT_state"] = ((states_technical["CT_state"] / (max_CT-min_CT)) * 2) - 1

	# CHANGE THIS! These need to be adjusted (ML and PC calculations)
	# Calculation of the ML1
	min_ML1 = 0
	max_ML1 = 10
	emergence_states["ML1_state"] = ((states_technical["ML1_state"] / (max_ML1-min_ML1)) * 2) - 1

	# Calculation of the ML2
	min_ML2 = 0
	max_ML2 = 10
	emergence_states["ML2_state"] = ((states_technical["ML2_state"] / (max_ML2-min_ML2)) * 2) - 1

	# Calculation of the ML3
	min_ML3 = 0
	max_ML3 = 10
	emergence_states["ML3_state"] = ((states_technical["ML3_state"] / (max_ML3-min_ML3)) * 2) - 1

	# Calculation of the ML4
	min_ML4 = 0
	max_ML4 = 10
	emergence_states["ML4_state"] = ((states_technical["ML4_state"] / (max_ML4-min_ML4)) * 2) - 1

	# Calculation of the FPS
	min_FPS = 0
	max_FPS = 10
	emergence_states["FPS_state"] = ((states_technical["FPS_state"] / (max_FPS-min_FPS)) * 2) - 1

	# Calculation of the safety
	min_Sa = 0
	max_Sa = 10
	emergence_states["Sa_state"] = ((states_technical["Sa_state"] / (max_Sa-min_Sa)) * 2) - 1

	return emergence_states