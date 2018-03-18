

"""
This is file 0. This file control the entire model including the two main parts: the policy emergence model and the technical model whatever it might be.

Some additional notes and remarks on the model below:
- For the moment, the possibilities to run a multitude of experimentation has been removed from the model.
- It is advised to keep to the backbone+ or the ACF models, the 3S approach is not mature and requires additional work.


NOTE:
- CHANGE THIS! text is added to the part that still need to be changed.
"""




#####################################################################
# Import of the different modules required for the run of the model #
#####################################################################

#####################################################################
# General imports
import copy
import random
import pandas as pd

#####################################################################
# Policy emergence model imports
from model import PolicyEmergence
from mesa.batchrunner import BatchRunner
from initialisation import initial_values
from initialisation_exploration import initial_values_exploration
import matplotlib.pyplot as plt
from technical_model import Technical_Model
from datacollection import DataCollector
from agent import Policymakers, Electorate, Externalparties, Truth, Policyentres

#####################################################################
# Technical model imports
import pysd
import numpy as np
from states import states_calculation, states_definition




#####################################################################
####### Initialisations of the different parts of the models ########
#####################################################################

"""
This part of the model contains all the inputs require to initialise the model with external parameters. This is the out of model intialisation that does not require random changes for each run. The in model initialisation is placed in the model run function as it will lead to changes (usually random ones) needed for each run.
"""

#####################################################################
# General initialisation

# Pseudo random generator seed - for verification and validation purposes
# random.seed(42)

# For tailored runs:
min_run_number = 0
max_run_number = 1

# For the total number of steps in years
run_time_year = 20

#####################################################################
# Policy emergence model initialisation

# Input dictionnary containing all inputs related to the policy emergence model
inputs_dict_emergence = dict()

# ACF principle belief of interest
PC_ACF_interest = 0

# Method chosen:
# 0: Backbone, 1: Backbone+, 2: 3S, 3: ACF
# Note right now several parts of the model require that the same method be used for both
# parts of the model - this is mostly related to the input method. This will be changed
# in the future.
AS_theory = 3
PF_theory = AS_theory

# This is set up to use only in the case of exploration
exploration = False

# Choosing the evernal event
event1 = False
event2 = False
event3 = False
event4 = False

# Reading the experiment input files
if exploration == False:
	experiment_input = pd.read_csv("Experiments_LHS.data",header=None)
if exploration == True and AS_theory == 0:
	experiment_input = pd.read_csv("Exploration_LHS_Backbone.data",header=None)
if exploration == True and AS_theory == 1:
	experiment_input = pd.read_csv("Exploration_LHS_Backbone+.data",header=None)
if exploration == True and AS_theory == 2:
	experiment_input = pd.read_csv("Exploration_LHS_3S.data",header=None)
if exploration == True and AS_theory == 3:
	experiment_input = pd.read_csv("Exploration_LHS_ACF.data",header=None)

# CHANGE THIS! - Should include the ratio of steps between the technical model and the policy emergence model
run_number_total = len(experiment_input.index)
if exploration == True:
	ticks = 250
else:
	ticks = 500

# CHANGE THIS! - This will need to be much more advanced, and preferably removed from the main step file where it currently resides
events = [0, event1, event2, event3, event4]


# Decision on the number of agents considered
policymaker_number = 6
externalparties_number = 6
agent_inputs = [policymaker_number, externalparties_number]

# Selecting the time step for the policy emergence model (time step interval in years)
time_step_emergence = 1

#####################################################################
# Technical model initialisation

# Selecting the time step for the system dynamics model (time step interval in years)
time_step_SD = 0.0078125

# CHANGE THIS! - The initialisation of the policy instruments should be here considering they are related to the technical model

states_technical = dict()

states_technical['AT_state'] = 0
states_technical['OT_state'] = 0
states_technical['DT_state'] = 0
states_technical['FPT_state'] = 0
states_technical['ERC_state'] = 0
states_technical['RT_state'] = 0
states_technical['AdT_state'] = 0
states_technical['PH_state'] = 0
states_technical['RS_state'] = 0
states_technical['CT_state'] = 0
states_technical['Ex_state'] = 0
states_technical['PP_state'] = 0
states_technical['RA_state'] = 0
states_technical['IL_state'] = 0
states_technical['FPS_state'] = 0
states_technical['Sa_state'] = 0

# These are the initial values for the exogenous parameters that can be changed through the model policy instruments and packages
AT_value = 20
OT_value = 25
DT_value = 2.5
FPT_value = 0.5
# CHANGE THIS! This needs to affect the values in the graph only
ERC_value = 0
RT_value = 3.5
AdT_value = 30
PH_value = 55
RS_value = 0.2
CT_value = 5

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
'''''


#####################################################################
##################### Running the model #############################
#####################################################################


"""
This is the part of the script that is used to run the model. It contains the running of the model through the different steps but also the collection of the data for the policy emergence model.


Warning:
For the time being, the experimentation part is being removed, only one experiment can be run at a time.
"""

# Setting the number of times the model will need to run
for run_number in range(run_number_total):

	if run_number >= min_run_number and run_number < max_run_number:

		print('   ')
		print('----------------------------------------------------')
		print('----------------------- RUN ' + str(run_number) + ' ---------------------')
		print('----------------------------------------------------')
		print('   ')

		#####################################################################
		# IN-MODEL INITIALISATION - Policy emergence model

		# Running the function to fill the input dictionary
		if exploration == False:
			initial_values(inputs_dict_emergence, experiment_input, run_number, agent_inputs, AS_theory, PF_theory)
		else:
			initial_values_exploration(inputs_dict_emergence, experiment_input, run_number, agent_inputs,  AS_theory, PF_theory)

		# Creation of the agent lists required for the policy emergence model
		agent_action_list = []
		for agents in inputs_dict_emergence["Agents"]:
			if type(agents) == Policymakers or type(agents) == Policyentres or type(agents) == Externalparties:
				agent_action_list.append(agents)

		# Assigning the run number of the action agents and the simulation
		for agents in agent_action_list:
			agents.run_number = run_number
		inputs_dict_emergence["Run_number"] = run_number

		# Selecting the data that needs to be collected throughout the model
		if AS_theory == 0 or PF_theory == 0:
			datacollector = DataCollector(
				# Model
				{"Run_number": lambda m: m.run_number,
				"Agenda_issue": lambda m: m.agenda_as_issue,
				"Chosen_instrument": lambda m: m.agenda_instrument,
				"Belieftruth_tree": lambda m: m.belieftree_truth
				},
				# Electorate
				{"Run_number": lambda a: a.run_number,
				"Belieftree": lambda a: a.belieftree_electorate,
				"Affiliation" : lambda a: a.affiliation
				},
				# Agents
				{"Run_number": lambda a: a.run_number,
				"Type": lambda a: type(a),
				"Belieftree": lambda a: a.belieftree[0],
				"Affiliation" : lambda a: a.affiliation
				})

		if AS_theory == 1 or PF_theory == 1:
			datacollector = DataCollector(
				# Model
				{"Run_number": lambda m: m.run_number,
				"Agenda_issue": lambda m: m.agenda_as_issue,
				"Chosen_instrument": lambda m: m.agenda_instrument,
				"Belieftruth_tree": lambda m: m.belieftree_truth
				},
				# Electorate
				{"Run_number": lambda a: a.run_number,
				"Belieftree": lambda a: a.belieftree_electorate,
				"Affiliation" : lambda a: a.affiliation
				},
				# Agents
				{"Run_number": lambda a: a.run_number,
				"Type": lambda a: type(a),
				"Belieftree": lambda a: a.belieftree[0],
				"Affiliation" : lambda a: a.affiliation
				},
				#  Links
				{"Agent1": lambda l:l.agent1,
				"Agent2": lambda l:l.agent2,
				"Agent3": lambda l: l.aware
				})

		if AS_theory == 2 or PF_theory == 2:
			datacollector = DataCollector(
				# Model
				{"Run_number": lambda m: m.run_number,
				"Agenda_issue": lambda m: m.agenda_as_issue,
				"Chosen_instrument": lambda m: m.agenda_instrument,
				"Belieftruth_tree": lambda m: m.truthagent.belieftree_truth,
				"Agenda_prob_3S": lambda m: m.agenda_prob_3S_as,
				"Agenda_poli_3S": lambda m: m.agenda_poli_3S_as,
				"Instruments": lambda m: m.instruments,
				"Policies": lambda m: m.policies
				},
				# Electorate
				{"Run_number": lambda a: a.run_number,
				"Belieftree": lambda a: a.belieftree_electorate,
				"Affiliation" : lambda a: a.affiliation
				},
				# Agents
				{"Run_number": lambda a: a.run_number,
				"Type": lambda a: type(a),
				"Belieftree": lambda a: a.belieftree[0],
				"Belieftree_policy": lambda a: a.belieftree_policy[0],
				"Belieftree_instrument": lambda a: a.belieftree_instrument[0],
				"Affiliation" : lambda a: a.affiliation
				},
				#  Links
				{"Agent1": lambda l:l.agent1,
				"Agent2": lambda l:l.agent2,
				"Agent3": lambda l: l.aware,
				},
				# Teams AS
				{"Lead": lambda c: c.lead.unique_id,
				"Issue": lambda c: c.issue,
				"Issue_type": lambda c:c.issue_type,
				"Creation": lambda c: c.creation,
				"Members": lambda c: c.members_id,
				"Resources" : lambda c: c.resources[0]},
				# Teams PF
				{"Lead": lambda c: c.lead.unique_id,
				"Issue": lambda c: c.issue,
				"Issue_type": lambda c:c.issue_type,
				"Creation": lambda c: c.creation,
				"Members": lambda c: c.members_id,
				"Resources" : lambda c: c.resources[0]})

		if AS_theory == 3 or PF_theory == 3:
			datacollector = DataCollector(
				# Model
				{"Run_number": lambda m: m.run_number,
				"Agenda_issue": lambda m: m.agenda_as_issue,
				"Chosen_instrument": lambda m: m.agenda_instrument,
				"Belieftruth_tree": lambda m: m.truthagent.belieftree_truth
				},
				# Electorate
				{"Run_number": lambda a: a.run_number,
				"Belieftree": lambda a: a.belieftree_electorate,
				"Affiliation" : lambda a: a.affiliation
				},
				# Agents
				{"Run_number": lambda a: a.run_number,
				"Type": lambda a: type(a),
				"Belieftree": lambda a: a.belieftree[0],
				"Affiliation" : lambda a: a.affiliation
				},
				#  Links
				{"Agent1": lambda l:l.agent1,
				"Agent2": lambda l:l.agent2,
				"Agent3": lambda l: l.aware,
				},
				# Teams AS
				{},
				# Teams PF
				{},
				# Coalitions AS
				{"Lead": lambda c: c.lead.unique_id,
				"Issue": lambda c: c.issue,
				"Creation": lambda c: c.creation,
				"Members": lambda c: c.members_id,
				"Resources" : lambda c: c.resources[0]},
				# Coalitions PF
				{"Lead": lambda c: c.lead.unique_id,
				"Issue": lambda c: c.issue,
				"Creation": lambda c: c.creation,
				"Members": lambda c: c.members_id,
				"Resources" : lambda c: c.resources[0]
				})
	
		#####################################################################
		# RUNNING THE MODEL

		# This initialise the policy emergence model (runs the __init__ part of the PolicyEmergence file)
		model_emergence = PolicyEmergence(PC_ACF_interest, datacollector, run_number, inputs_dict_emergence, events)

		# This initialises the technical model and transforms it from vensim to python.
		model_technical = pysd.read_vensim('Flood_Levees_14_Final.mdl')

		# Deciding on the number of steps that should be considered
		if time_step_SD < time_step_emergence:
			time_step_model = time_step_SD
		else:
			time_step_model = time_step_emergence

		# Running the technical model first (for one year time step only:
		# CHANGE THIS! - The function that has been commented out is the one with the ERC in it
		# model_technical_output = model_technical.run(params={'FINAL TIME':time_step_emergence, 'aging time':AT_value, 'obsolescence time':OT_value, 'design time':DT_value, 'flood perception time':FPT_value, 'effect on renovation and construction':ERC_value, 'renovation time':RT_value, 'adjustment time':AdT_value, 'planning horizon':PH_value, 'renovation standard':RS_value, 'construction time':CT_value})
		model_technical_output = model_technical.run(params={'FINAL TIME':time_step_emergence, 'aging time':AT_value, 'obsolescence time':OT_value, 'design time':DT_value, 'flood perception time':FPT_value, 'renovation time':RT_value, 'adjustment time':AdT_value, 'planning horizon':PH_value, 'renovation standard':RS_value, 'construction time':CT_value})

		# For loop for the running of the model (-1 as an initial step has already been run)
		for n in range(run_time_year - 1):

			print('   ')
			print('--------------------- STEP ' + str(int(n+1)) + ' ---------------------')
			print('   ')

			# CHANGE THIS! - The model are now running fine - What is missing is:
			# - Policy instruments
			# - Communication of the states
			# - Implementation of the policy instruments

			# Obtention of the states values from the technical model outputs
			states_technical = states_definition(model_technical, states_technical)
			print(states_technical)

			# This performs one step of the policy emergence model
			# CHANGE THIS! - The communication of the states must be placed into the step function
			emergence_states = states_calculation(states_technical)
			# model_emergence.step(AS_theory, PF_theory)

			# This is the part where the policy instruments are implemented
			# CHANGE THIS! - The function has not been written yet

			# CHANGE THIS! - Somewhere here should be the introduction of the policy instrument into the technical model

			# This performs 1 years worth of steps for the system dynamics model
			# CHANGE THIS - The function that has been commented out is the one with the ERC in it
			# model_technical_output_intermediate = model_technical.run(params={'aging time':AT_value, 'obsolescence time':OT_value, 'design time':DT_value, 'flood perception time':FPT_value, 'effect on renovation and construction':ERC_value, 'renovation time':RT_value, 'adjustment time':AdT_value, 'planning horizon':PH_value, 'renovation standard':RS_value, 'construction time':CT_value}, initial_condition='current', return_timestamps=np.linspace(1+n,1+n+1, 1/0.0078125))
			model_technical_output_intermediate = model_technical.run(params={'aging time':AT_value, 'obsolescence time':OT_value, 'design time':DT_value, 'flood perception time':FPT_value, 'renovation time':RT_value, 'adjustment time':AdT_value, 'planning horizon':PH_value, 'renovation standard':RS_value, 'construction time':CT_value}, initial_condition='current', return_timestamps=np.linspace(1+n,1+n+1, 1/0.0078125))
			model_technical_output = model_technical_output.append(model_technical_output_intermediate)






		# Plotting all of the results of the technical model
		model_technical_output.plot()
		plt.show()

		#####################################################################
		# STORING THE DATA - Policy emergence model

		# For the backbone
		if AS_theory == 0 or PF_theory == 0:
			if exploration == False and event1 == False and event2 == False and event3 == False and event4 == False:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_B_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B_' + str(run_number) + '.csv')
			if exploration == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_B_exp_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B_exp_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B_exp_' + str(run_number) + '.csv')
			if exploration == False and event1 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('2_model_B_event1_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('2_electorate_B_event1_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('2_agents_B_event1_' + str(run_number) + '.csv')
			if exploration == False and event2 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('2_model_B_event2_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('2_electorate_B_event2_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('2_agents_B_event2_' + str(run_number) + '.csv')
			if exploration == False and event3 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_B_event3_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B_event3_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B_event3_' + str(run_number) + '.csv')
			if exploration == False and event4 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_B_event4_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B_event4_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B_event4_' + str(run_number) + '.csv')
		# For the backbone+
		if AS_theory == 1 or PF_theory == 1:
			if exploration == False and event1 == False and event2 == False and event3 == False and event4 == False:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_B+_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B+_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B+_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_B+_' + str(run_number) + '.csv')
			if exploration == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_B+_exp_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B+_exp_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B+_exp_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_B+_exp_' + str(run_number) + '.csv')
			if exploration == False and event1 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('2_model_B+_event1_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('2_electorate_B+_event1_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('2_agents_B+_event1_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('2_links_B+_event1_' + str(run_number) + '.csv')
			if exploration == False and event2 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('_model_B+_event2_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B+_event2_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B+_event2_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_B+_event_2_' + str(run_number) + '.csv')
			if exploration == False and event3 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_B+_event3_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B+_event3_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B+_event3_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_B+_event3_' + str(run_number) + '.csv')
			if exploration == False and event4 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_B+_event4_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_B+_event4_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_B+_event4_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_B+_event4_' + str(run_number) + '.csv')
		# For the 3S
		if AS_theory == 2 or PF_theory == 2:
			if exploration == False and event1 == False and event2 == False and event3 == False and event4 == False:
				# frames_model.append(test_model.datacollector.get_model_vars_dataframe())
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_3S_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_3S_' + str(run_number) + '.csv')
				# frames_agents.append(test_model.datacollector.get_agent_vars_dataframe())
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_3S_' + str(run_number) + '.csv')
				# frames_teams_as.append(test_model.datacollector.get_team_as_vars_dataframe())
				df_team_as = test_model.datacollector.get_team_as_vars_dataframe()
				df_team_as.to_csv('1_teams_as_3S_' + str(run_number) + '.csv')
				# frames_teams_pf.append(test_model.datacollector.get_team_pf_vars_dataframe())
				df_team_pf = test_model.datacollector.get_team_pf_vars_dataframe()
				df_team_pf.to_csv('1_teams_pf_3S_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_3S_' + str(run_number) + '.csv')
			if exploration == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_3S_exp_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_3S_exp_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_3S_exp_' + str(run_number) + '.csv')
				df_team_as = test_model.datacollector.get_team_as_vars_dataframe()
				df_team_as.to_csv('1_teams_as_3S_exp_' + str(run_number) + '.csv')
				df_team_pf = test_model.datacollector.get_team_pf_vars_dataframe()
				df_team_pf.to_csv('1_teams_pf_3S_exp_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_3S_exp_' + str(run_number) + '.csv')
			if exploration == False and event1 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('2_model_3S_event1_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('2_electorate_3S_event1_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('2_agents_3S_event1_' + str(run_number) + '.csv')
				df_team_as = test_model.datacollector.get_team_as_vars_dataframe()
				df_team_as.to_csv('2_teams_as_3S_event1_' + str(run_number) + '.csv')
				df_team_pf = test_model.datacollector.get_team_pf_vars_dataframe()
				df_team_pf.to_csv('2_teams_pf_3S_event1_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('2_links_3S_event1_' + str(run_number) + '.csv')
			if exploration == False and event2 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_3S_event2_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_3S_event2_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_3S_event2_' + str(run_number) + '.csv')
				df_team_as = test_model.datacollector.get_team_as_vars_dataframe()
				df_team_as.to_csv('1_teams_as_3S_event2_' + str(run_number) + '.csv')
				df_team_pf = test_model.datacollector.get_team_pf_vars_dataframe()
				df_team_pf.to_csv('1_teams_pf_3S_event2_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_3S_event2_' + str(run_number) + '.csv')
			if exploration == False and event3 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_3S_event3_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_3S_event3_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_3S_event3_' + str(run_number) + '.csv')
				df_team_as = test_model.datacollector.get_team_as_vars_dataframe()
				df_team_as.to_csv('1_teams_as_3S_event3_' + str(run_number) + '.csv')
				df_team_pf = test_model.datacollector.get_team_pf_vars_dataframe()
				df_team_pf.to_csv('1_teams_pf_3S_event3_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_3S_event3_' + str(run_number) + '.csv')
			if exploration == False and event4 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_3S_event4_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_3S_event4_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_3S_event4_' + str(run_number) + '.csv')
				df_team_as = test_model.datacollector.get_team_as_vars_dataframe()
				df_team_as.to_csv('1_teams_as_3S_event4_' + str(run_number) + '.csv')
				df_team_pf = test_model.datacollector.get_team_pf_vars_dataframe()
				df_team_pf.to_csv('1_teams_pf_3S_event4_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_3S_event4_' + str(run_number) + '.csv')
		# For the ACF
		if AS_theory == 3 or PF_theory == 3:
			if exploration == False and event1 == False and event2 == False and event3 == False and event4 == False:
				# frames_model.append(test_model.datacollector.get_model_vars_dataframe())
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_ACF_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_ACF_' + str(run_number) + '.csv')
				# frames_agents.append(test_model.datacollector.get_agent_vars_dataframe())
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_ACF_' + str(run_number) + '.csv')
				# frames_coalitions_as.append(test_model.datacollector.get_coalition_as_vars_dataframe())
				df_coalition_as = test_model.datacollector.get_coalition_as_vars_dataframe()
				df_coalition_as.to_csv('1_coalitions_as_ACF_' + str(run_number) + '.csv')
				df_coalition_pf = test_model.datacollector.get_coalition_pf_vars_dataframe()
				df_coalition_pf.to_csv('1_coalitions_pf_ACF_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_ACF_' + str(run_number) + '.csv')
			if exploration == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_ACF_exp_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_ACF_exp_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_ACF_exp_' + str(run_number) + '.csv')
				df_coalition_as = test_model.datacollector.get_coalition_as_vars_dataframe()
				df_coalition_as.to_csv('1_coalitions_as_ACF_exp_' + str(run_number) + '.csv')
				df_coalition_pf = test_model.datacollector.get_coalition_pf_vars_dataframe()
				df_coalition_pf.to_csv('1_coalitions_pf_ACF_exp_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_ACF_exp_' + str(run_number) + '.csv')
			if exploration == False and event1 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('2_model_ACF_event1_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('2_electorate_ACF_event1_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('2_agents_ACF_event1_' + str(run_number) + '.csv')
				df_coalition_as = test_model.datacollector.get_coalition_as_vars_dataframe()
				df_coalition_as.to_csv('2_coalitions_as_ACF_event1_' + str(run_number) + '.csv')
				df_coalition_pf = test_model.datacollector.get_coalition_pf_vars_dataframe()
				df_coalition_pf.to_csv('2_coalitions_pf_ACF_event1_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('2_links_ACF_event1_' + str(run_number) + '.csv')
			if exploration == False and event2 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_ACF_event2_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_ACF_event2_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_ACF_event2_' + str(run_number) + '.csv')
				df_coalition_as = test_model.datacollector.get_coalition_as_vars_dataframe()
				df_coalition_as.to_csv('1_coalitions_as_ACF_event2_' + str(run_number) + '.csv')
				df_coalition_pf = test_model.datacollector.get_coalition_pf_vars_dataframe()
				df_coalition_pf.to_csv('1_coalitions_pf_ACF_event2_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_ACF_event2_' + str(run_number) + '.csv')
			if exploration == False and event3 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_ACF_event3_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_ACF_event3_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_ACF_event3_' + str(run_number) + '.csv')
				df_coalition_as = test_model.datacollector.get_coalition_as_vars_dataframe()
				df_coalition_as.to_csv('1_coalitions_as_ACF_event3_' + str(run_number) + '.csv')
				df_coalition_pf = test_model.datacollector.get_coalition_pf_vars_dataframe()
				df_coalition_pf.to_csv('1_coalitions_pf_ACF_event3_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_ACF_event3_' + str(run_number) + '.csv')
			if exploration == False and event4 == True:
				df_model = test_model.datacollector.get_model_vars_dataframe()
				df_model.to_csv('1_model_ACF_event4_' + str(run_number) + '.csv')
				df_electorate = test_model.datacollector.get_electorate_vars_dataframe()
				df_electorate.to_csv('1_electorate_ACF_event4_' + str(run_number) + '.csv')
				df_agents = test_model.datacollector.get_agent_vars_dataframe()
				df_agents.to_csv('1_agents_ACF_event4_' + str(run_number) + '.csv')
				df_coalition_as = test_model.datacollector.get_coalition_as_vars_dataframe()
				df_coalition_as.to_csv('1_coalitions_as_ACF_event4_' + str(run_number) + '.csv')
				df_coalition_pf = test_model.datacollector.get_coalition_pf_vars_dataframe()
				df_coalition_pf.to_csv('1_coalitions_pf_ACF_event4_' + str(run_number) + '.csv')
				df_links = test_model.datacollector.get_links_vars_dataframe()
				df_links.to_csv('1_links_ACF_event4_' + str(run_number) + '.csv')

		# CHANGE THIS! - The data from the system dynamics model should also be saved



# Assembling the dataframes and printing to file
# result_model = pd.concat(frames_model)
# result_model.to_csv('1_model_backbone.csv')
# result_agents = pd.concat(frames_agents)
# result_agents.to_csv('1_agent_backbone.csv')
# if AS_theory == 2 or PF_theory == 2:
# 	result_teams_as = pd.concat(frames_teams_as)
# 	result_teams_as.to_csv('1_teams_as_file2.csv')
# 	result_teams_pf = pd.concat(frames_teams_pf)
# 	result_teams_pf.to_csv('1_teams_pf_file2.csv')
# if AS_theory == 3 or PF_theory == 3:
# 	result_coalitions_as = pd.concat(frames_coalitions_as)
# 	result_coalitions_as.to_csv('1_coalitions_as_file2.csv')
# 	result_coalitions_pf = pd.concat(frames_coalitions_pf)
# 	result_coalitions_pf.to_csv('1_coalitions_pf_file2.csv')