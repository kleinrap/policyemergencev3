import random
import copy
import pysd

def states_definition(model_technical, states_technical):

	'''''
	What are the states that are needed from the technical model?

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
	11. Expertise - calculated (Ex_state)
	12. Public perception - calculated (PP_state)
	13. Resource allocation - calculated (RA_state)
	14. Investment level - calculated (IL_state)
	15. Flood perception spending - calculated (FPS_state)
	16. Safety - calculated (Sa_state)

	This function is used to assign the values from the technical model to their respective dictionnary.
	'''''

	# Assigning the values to the right dictionnary parameters
	states_technical["AT_state"] = model_technical.components.aging_time()
	states_technical["OT_state"] = model_technical.components.obsolescence_time()
	states_technical["DT_state"] = model_technical.components.design_time()
	states_technical["FPT_state"] = model_technical.components.effect_on_renovation_and_construction()
	states_technical["RT_state"] = model_technical.components.renovation_time()
	states_technical["AdT_state"] = model_technical.components.adjustment_time()
	states_technical["PH_state"] = model_technical.components.planning_horizon()
	states_technical["RS_state"] = model_technical.components.renovation_standard()
	states_technical["CT_state"] = model_technical.components.construction_time()
	# CHANGE THIS! - The ML beliefs will be changed, so they have not been added
	# CHANGE THIS! - The PC beliefs will be changes as well, so they have not been added


	print(states_technical)

	return states_technical

def states_calculation(states_technical):

	'''''
	This function is used to calculate the states into a -1,1 interval from the states obtained in the technical model.
	'''''


	emergence_states = 0

	return emergence_states