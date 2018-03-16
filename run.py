# This is file 0. This file control the entire model including the two main parts: the policy emergence model and the 
# technical model whatever it might be.

#####################################################################
# Import of the different modules required for the run of the model #
#####################################################################
# General imports
import copy
import random
import pandas as pd

# Policy emergence model imports
from model import PolicyEmergence
from mesa.batchrunner import BatchRunner
from initialisation import initial_values
from initialisation_exploration import initial_values_exploration
import matplotlib.pyplot as plt
from technical_model import Technical_Model
from datacollection import DataCollector
from agent import Policymakers, Electorate, Externalparties, Truth, Policyentres

# Technical model imports


#####################################################################
####### Initialisations of the different parts of the models ########
#####################################################################

# Input dictionnary
inputs_dict = dict()

# Pseudo random generator seed
# random.seed(42)

# ACF principle belief of interest
Pr_ACF_interest = 0

# Method chosen:
# 0: Backbone, 1: Backbone+, 2: 3S, 3: ACF
# Note right now several parts of the model require that the same method be used for both
# parts of the model - this is mostly related to the input method. This will be changed
# in the future.
AS_theory = 2
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


run_number_total = len(experiment_input.index)
if exploration == True:
	ticks = 250
else:
	ticks = 500

events = [0, event1, event2, event3, event4]


policymaker_number = 6
externalparties_number = 6
agent_inputs = [policymaker_number, externalparties_number]

# The frames used to save the data output
# frames_model = []
# frames_agents = []
# frames_links = []
# frames_teams_as = []
# frames_teams_pf = []
# frames_coalitions_as = []
# frames_coalitions_pf = []

# Setting the number of times the model will need to run
for run_number in range(run_number_total):


	if run_number >= 16 and run_number < 30:

		print('   ')
		print('-------------------------------------------------------------------------')
		print('--------------------- RUN ' + str(run_number) + ' ---------------------')
		print('-------------------------------------------------------------------------')
		print('   ')


		# Running the function to fill the input dictionary
		if exploration == False:
			initial_values(inputs_dict, experiment_input, run_number, agent_inputs, AS_theory, PF_theory)
		else:
			initial_values_exploration(inputs_dict, experiment_input, run_number, agent_inputs,  AS_theory, PF_theory)


		agent_action_list = []
		for agents in inputs_dict["Agents"]:
			if type(agents) == Policymakers or type(agents) == Policyentres or type(agents) == Externalparties:
				agent_action_list.append(agents)


		# Assigning the run number of the action agents and the simulation
		for agents in agent_action_list:
			agents.run_number = run_number
		inputs_dict["Run_number"] = run_number

		# Selecting the data that needs to be collected throughout the model
		if AS_theory == 0 or PF_theory == 0:
			datacollector = DataCollector(
				# Model
				{"Run_number": lambda m: m.run_number,
				"Empty": lambda m: m.technical_model.cell_count("Empty"),
				"Burnt": lambda m: m.technical_model.cell_count("Burnt"),
				"Camp_site": lambda m: m.technical_model.cell_count("Camp site"),
				"Thin_forest": lambda m: m.technical_model.cell_count("Thin forest"),
				"Thick_forest": lambda m: m.technical_model.cell_count("Thick forest"),
				"Firefighters": lambda m: m.firefighter_force,
				"Prevention": lambda m: m.thin_burning_probability,
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
				"Empty": lambda m: m.technical_model.cell_count("Empty"),
				"Burnt": lambda m: m.technical_model.cell_count("Burnt"),
				"Camp_site": lambda m: m.technical_model.cell_count("Camp site"),
				"Thin_forest": lambda m: m.technical_model.cell_count("Thin forest"),
				"Thick_forest": lambda m: m.technical_model.cell_count("Thick forest"),
				"Firefighters": lambda m: m.firefighter_force,
				"Prevention": lambda m: m.thin_burning_probability,
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
				"Empty": lambda m: m.technical_model.cell_count("Empty"),
				"Burnt": lambda m: m.technical_model.cell_count("Burnt"),
				"Camp_site": lambda m: m.technical_model.cell_count("Camp site"),
				"Thin_forest": lambda m: m.technical_model.cell_count("Thin forest"),
				"Thick_forest": lambda m: m.technical_model.cell_count("Thick forest"),
				# "Instruments": lambda m: m.agenda_instrument,
				"Firefighters": lambda m: m.firefighter_force,
				"Prevention": lambda m: m.thin_burning_probability,
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
				"Empty": lambda m: m.technical_model.cell_count("Empty"),
				"Burnt": lambda m: m.technical_model.cell_count("Burnt"),
				"Camp_site": lambda m: m.technical_model.cell_count("Camp site"),
				"Thin_forest": lambda m: m.technical_model.cell_count("Thin forest"),
				"Thick_forest": lambda m: m.technical_model.cell_count("Thick forest"),
				# "Instruments": lambda m: m.agenda_instrument,
				"Firefighters": lambda m: m.firefighter_force,
				"Prevention": lambda m: m.thin_burning_probability,
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

		# Running the model
		test_model = PolicyEmergence(Pr_ACF_interest, datacollector, run_number, inputs_dict, events)
		for i in range(ticks):
			print('   ')
			print('--------------------- STEP ' + str(i+1) + ' ---------------------')
			print('   ')

			test_model.step(AS_theory, PF_theory)

		# Storing the data
		# For backbone and backbone+
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
		# For 3S
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
		# For ACF
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

