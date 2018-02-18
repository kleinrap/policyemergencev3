# 0. Import the appropriate files

# Importing functions from python
import random
import doctest
import math
import copy
import os
from collections import defaultdict, Counter

# Importing functions from mesa
from mesa import Model
from mesa.space import MultiGrid

# Importing functions from other PMT files
from datacollection import DataCollector
from technical_model import Technical_Model
from tree_cell import TreeCell
from agent import Policymakers, Electorate, Externalparties, Truth, Policyentres
from network_creation import PolicyNetworkLinks
from team_creation import Team
from coalition_creation import Coalition
from functions_actions import ActionFunctions

# When running from this file (no visualisation)

# 1. PolicyEmergence class
class PolicyEmergence(Model):

	# 1.1 Initialisation function __init__
	def __init__(self, Pr_ACF_interest=0, datacollector=0, run_number=0, inputs_dict=dict(), events=0):
		
		"""
		1.1 Initialisation function __init__
		===========================

		This is the initialisation function. It is used to set up the necessary lists as well as
		unpacking the inputs_dict dictionary containing all experiment specific inputs.

		"""	

		# 1.1.1 Extracting the inputs from the input dictionnary
		# From inputs:
		self.height = inputs_dict["height"]
		self.width = inputs_dict["width"]
		# Forest fire model related inputs
		self.instrument_campSites = inputs_dict["technical_input"][0]
		self.instrument_planting = inputs_dict["technical_input"][1]
		self.thin_burning_probability = inputs_dict["technical_input"][2]
		self.firefighter_force = inputs_dict["technical_input"][3]
		self.instrument_prevention = inputs_dict["technical_input"][4]
		# Agent numbers related inputs
		self.externalparties_number = inputs_dict["total_agent_number"][0]
		self.policymaker_number = inputs_dict["total_agent_number"][1]
		self.policyentre_number = inputs_dict["total_agent_number"][2]
		# Affiliations related inputs
		self.affiliation_number = inputs_dict["affiliation_input"][0]
		self.affiliation_weights = inputs_dict["affiliation_input"][1]
		# Agenda related inputs
		self.agenda_as_issue = inputs_dict["Agenda_inputs"][0]
		self.agenda_instrument = inputs_dict["Agenda_inputs"][1]
		self.agenda_prob_3S_as = inputs_dict["Agenda_inputs"][2]
		self.agenda_poli_3S_as = inputs_dict["Agenda_inputs"][3]
		# Belief structure related inputs
		self.deep_core = inputs_dict["deep_core"]
		self.len_Pr = len(self.deep_core)
		self.policy_core = inputs_dict["policy_core"]
		self.len_PC = len(self.policy_core)
		self.secondary = inputs_dict["secondary"]
		self.len_S = len(self.secondary)
		# Policy and instruments related inputs
		self.instruments = inputs_dict["Instruments"]
		self.policies = inputs_dict["Policies"]
		self.aware_decay_coefficient = inputs_dict["Trust_decay_coefficient"]
		self.conflict_level_coef = inputs_dict["conflict_level_coef"]
		self.coalition_threshold = inputs_dict["coalition_threshold"]
		self.team_gap_threshold  = inputs_dict["team_gap_threshold"]
		self.team_belief_problem_threshold = inputs_dict["team_belief_problem_threshold"]
		self.team_belief_policy_threshold = inputs_dict["team_belief_policy_threshold"]
		self.no_interest_states = inputs_dict["No_interest_states"]
		self.electorate_influence_coefficient = inputs_dict["electorate_influence_coefficient"]
		self.electorate_number = self.affiliation_number
		self.Pr_ACF_interest = Pr_ACF_interest
		self.datacollector = datacollector
		self.run_number = inputs_dict["Run_number"]
		self.representation = inputs_dict["representation"]
		self.resources_weight_action = inputs_dict["resources_weight_action"]
		self.resources_potency = inputs_dict["resources_potency"]
		self.events = events

		# 1.1.2 Technical model parameter inputs
		self.grid = MultiGrid(self.height, self.width, torus=True)
		self.technical_model = Technical_Model(self.len_Pr, self.len_PC, self.len_S)

		# 1.1.3 Derived inputs:
		self.total_agent_number = self.externalparties_number + self.policymaker_number + self.policyentre_number
		self.issues_number = self.len_Pr + self.len_PC + self.len_S
		self.causalrelation_number = self.len_Pr*self.len_PC + self.len_PC*self.len_S

		# 1.1.4 Creation of the master and agent action lists
		self.master_list = []
		self.agent_action_list = []
		self.electorate_list = []
		for agents in inputs_dict["Agents"]:
			self.master_list.append(agents)
			if type(agents) == Policymakers or type(agents) == Policyentres or type(agents) == Externalparties:
				self.agent_action_list.append(agents)
			if type(agents) == Truth:
				self.truthagent = agents
			if type(agents) == Electorate:
				self.electorate_list.append(agents)
		self.action_agent_number = len(self.agent_action_list)
		print("This is the list of active agents: " + str(self.agent_action_list))
		print(' ')

		# 1.1.5 Creation of the tree cells for the technical forest fire model
		k= 0
		for (contents, x, y) in self.grid.coord_iter():
			p = random.random()
			k = k+1
			# Create a tree
			new_tree = TreeCell((x, y), self)
			# 30% is for thin forest, 30% is for thick forest,
			# 10% is for camp sites and the rest is empty
			if p <= 0.3:
				new_tree.condition = "Thin forest"
			elif p <= 0.6 and p > 0.3:
				new_tree.condition = "Thick forest"
			elif p <= 0.62 and p > 0.6:
				new_tree.condition = "Camp site"
			else:
				new_tree.condition = "Empty"
			self.grid._place_agent((x, y), new_tree)
			self.technical_model.add(new_tree)

		# 1.1.6 None interests and partial knowledge initialisation
		# Creation of the truth belieftree (containing the real states of the world)
		self.belieftree_truth = [None for i in range(self.len_Pr + self.len_PC + self.len_S)]
		# Informing the agents about the none interest of the EP
		# External parties belief update and preference recalculation
		for agents in self.master_list:
			if type(agents) == Externalparties:
				for i in range(len(self.belieftree_truth)):
					# print(no_interest_states[agent.agent_id][i])
					if self.no_interest_states[agents.agent_id][i] != 1:
						agents.belieftree[0][i][0] = 'No'

		# Assigning all agent partial knowledge knowledge that these EP do not consider these states
		for agents in self.agent_action_list:
			for agents_ep in self.agent_action_list:
				# Only select the external party agents
				if type(agents_ep) == Externalparties:
					for i in range(len(self.belieftree_truth)):
						if agents_ep.belieftree[0][i][0] == 'No':
							agents.belieftree[1 + agents_ep.unique_id][i][0] = 'No'

		# 1.1.7 Creation of the list of all links
		self.link_list = inputs_dict["Link_list"]

		# 1.1.8 Initial update of the conflict levels
		self.conflict_level_update(self.link_list, self.deep_core, self.policy_core, self.secondary, self.conflict_level_coef)
		
		# 1.1 9 Initialisation of 3S-related parameters
		# For the three streams theory, creation of the team lists:
		# This is the list of active teams (agenda setting)
		self.team_list_as = []
		# This is the list that contains all teams (agenda setting)
		self.team_list_as_total = []
		self.team_number_as = [0]
		# This is the list of active teams (policy formulation)
		self.team_list_pf = []
		# This is the list that contains all teams (policy formulation)
		self.team_list_pf_total = []
		self.team_number_pf = [0]
		# Tick check for disbanding time requirements:
		self.tick_number = 0
		# # Creation of the network for the 3S shadow network
		# This is the list of active links (agenda setting)
		self.threeS_link_list_as = []
		# This is the list that contains all links (agenda setting)
		self.threeS_link_list_as_total = []
		self.threeS_link_id_as = [0]
		# This is the list of active links (policy formulation)
		self.threeS_link_list_pf = []
		# This is the list that contains all links (policy formulation)
		self.threeS_link_list_pf_total = []
		self.threeS_link_id_pf =[0]

		# 1.1 10 Initialisation of 3S-related parameters
		# For the ACF theory, creation of the coalitions lists:
		self.coalitions_list_as = []
		self.coalitions_list_as_total = []
		self.coalitions_number_as = [0]
		self.coalitions_list_pf = []
		self.coalitions_list_pf_total = []
		self.coalitions_number_pf = [0]
		# # Creation of the network for the ACF shadow network
		# This is the list of active links (agenda setting)
		self.ACF_link_list_as = []
		# This is the list that contains all links (agenda setting)
		self.ACF_link_list_as_total = []
		self.ACF_link_id_as = [0]
		# This is the list of active links (policy formulation)
		self.ACF_link_list_pf = []
		# This is the list that contains all links (policy formulation)
		self.ACF_link_list_pf_total = []
		self.ACF_link_id_pf =[0]

		
		print('   ')
		print('Cleared initialisation.')
		print('   ')

	def step(self, AS_theory, PF_theory):	

		"""
		1.2 The step function
		===========================

		This function is the function that runs the whole cycle. One run of this function
		represents one tick in the agent based model. It is composed of four main parts:

		1.2.1/ Tick initialisation
		1.2.9/ Agenda setting
		1.2.13/ Policy formulation
		1.2.18/ End of tick procedures

		"""	

		####################################################################################################
		# 1.2.1 - Tick initialisation
		print('   ')
		print('--- Tick initialisation ---')
		print('   ')
		####################################################################################################


		# 1.2.2 Iterating of the tick number
		# [Backbone/Backbone+/3S/ACF]
		self.tick_number +=1

		# Potential external events [Backbone/Backbone+/3S/ACF (can varry)]

		#******
		# print("Specific to the case study")
		#******

		# 1.2.3 Event selection parameters
		# Event 1 - reversal of all Pr-PC relations for all agents at tick 200
		if self.events[1] == True and self.tick_number == 200:
			for agents in self.agent_action_list:
				for cw in range(self.len_Pr*self.len_PC):
					agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + cw][0] = \
						- agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + cw][0]
		# Event 2 - reversal of all Pr-PC relations for policy makers at tick 200
		if self.events[2] == True and self.tick_number == 200:
			for agents in self.agent_action_list:
				if type(agents) == Policymakers:
					for cw in range(self.causalrelation_number):
						agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + cw][0] = \
							- agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + cw][0]
		# Event 3 - reversal of all Pr-PC relations for externalp parties at tick 200
		if self.events[3] == True and self.tick_number == 200:
			for agents in self.agent_action_list:
				if type(agents) == Externalparties:
					for cw in range(self.causalrelation_number):
						agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + cw][0] = \
							- agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + cw][0]
		# Event 3 - reversal of all Pr-PC relations for policy entrepreneurs at tick 200
		if self.events[4] == True and self.tick_number == 200:
			for agents in self.agent_action_list:
				if type(agents) == Policyentres:
					for cw in range(self.causalrelation_number):
						agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + cw][0] = \
							- agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + cw][0]

		# 1.2.4 Technical model simulation (technical_model.py)
		# [Backbone/Backbone+/3S/ACF]
		print('Running the technical model ...')

		# 1.2.4.1 Technical model simulation - calculate the states 
		master_cell = self.technical_model.cells_repository
		self.total_cells = self.height * self.width
		# Perform the step for each tree cell
		for agents in master_cell:
			agents.step(self.thin_burning_probability, self.firefighter_force)

		# 1.2.4.2 Update of the truth belieftree
		self.technical_model.states_update(self.height, self.width, self.belieftree_truth, \
		 self.thin_burning_probability, self.firefighter_force)
		print('... cleared.')
		print('   ')

		# 1.2.4.3 Implementing the policy instruments on the technical model
		prob_update = self.technical_model.measures_implementation(self.agenda_instrument, self.instruments, \
			self.instrument_campSites, self.instrument_planting, self.thin_burning_probability, \
			self.firefighter_force, self.instrument_prevention)
		self.thin_burning_probability = prob_update[0]
		self.firefighter_force = prob_update[1]

		# 1.2.4.4 Update the truth belieftree
		self.technical_model.states_update(self.height, self.width, self.belieftree_truth, \
		 self.thin_burning_probability, self.firefighter_force)
		print('... cleared.')
		print('   ')

		# 1.2.5 Electorate actions on policy makers
		# [Backbone/Backbone+/3S/ACF]
		print('Performing electorate actions on the policy makers ...')
		for agents in self.master_list:
			if type(agents) == Electorate:
				agents.electorate_influence(agents, self.master_list, self.affiliation_number, self.electorate_influence_coefficient)
		print('... cleared.')
		print('   ')

		# 1.2.6 Updating of agents' states
		print('Updating states ...')
		# 1.2.6.1 Update the beliefs of the truth agent
		self.truthagent.belieftree_truth = self.belieftree_truth

		# 1.2.6.2 External parties belief update
		# [Backbone/Backbone+/3S/ACF]
		for agents in self.master_list:
			if type(agents) == Externalparties:
				agents.external_parties_states_update(agents, self.master_list, self.no_interest_states)

		# 1.2.6.3 Electorate belief update
		# [Backbone/Backbone+/3S/ACF]
		for agents in self.master_list:
			if type(agents) == Electorate:
				agents.electorate_states_update(agents, self.master_list, self.affiliation_weights)

		# 1.2.6.4 Policy makers belief update
		# [Backbone/Backbone+/3S/ACF]
		for agents in self.master_list:
			if type(agents) == Policymakers:
				agents.policymakers_states_update(agents, self.master_list, self.affiliation_weights)

		# 1.2.6.5 Policy entrepreneurs belief update
		# [Backbone+/3S/ACF]
		if AS_theory == 1 or PF_theory == 1 or AS_theory == 2 or PF_theory == 2 \
			or AS_theory == 3 or PF_theory == 3:
			for agents in self.master_list:
				if type(agents) == Policyentres:
					agents.policyentres_states_update(agents, self.master_list, self.affiliation_weights)

		# 1.2.7 Principle core partial knowledge share
		for agents1 in self.agent_action_list:
			for agents2 in self.agent_action_list:
				for exchange in range(self.len_Pr):
					agents1.belieftree[1 + agents2.unique_id][exchange][0] = agents2.belieftree[0][exchange][0] + (random.random()/10) - 0.05
					agents1.belieftree[1 + agents2.unique_id][exchange][0] = \
						ActionFunctions.one_minus_one_check(agents1.belieftree[1 + agents2.unique_id][exchange][0])
					agents1.belieftree[1 + agents2.unique_id][exchange][1] = agents2.belieftree[0][exchange][1] + (random.random()/10) - 0.05
					agents1.belieftree[1 + agents2.unique_id][exchange][1] = \
						ActionFunctions.one_minus_one_check(agents1.belieftree[1 + agents2.unique_id][exchange][1])

		# 1.2.8 Initialisation of all partial knowledge values (first tick only)
		if self.tick_number == 1:
			# 1.2.8.1 For the belief tree of all agents
			for agents in self.agent_action_list:
				# For the partial knowledge too
				for who in range(len(self.agent_action_list)):
					for issue in range(self.len_Pr + self.len_PC + self.len_S):
						agents.belieftree[1+who][issue][0] = copy.copy(agents.belieftree[0][issue][0] + random.random() - 0.5)
						agents.belieftree[1+who][issue][1] = copy.copy(agents.belieftree[0][issue][1] + random.random() - 0.5)
						agents.belieftree[1+who][issue][0] = ActionFunctions.one_minus_one_check(agents.belieftree[1+who][issue][0])
						agents.belieftree[1+who][issue][1] = ActionFunctions.one_minus_one_check(agents.belieftree[1+who][issue][1])
					for causalrelations in range(self.causalrelation_number):
						agents.belieftree[1+who][self.len_Pr + self.len_PC + self.len_S + causalrelations][0] = \
							copy.copy(agents.belieftree[0][self.len_Pr + self.len_PC + self.len_S + causalrelations][0] + random.random() - 0.5)
						agents.belieftree[1+who][self.len_Pr + self.len_PC + self.len_S + causalrelations][0] = \
							ActionFunctions.one_minus_one_check(agents.belieftree[1+who][self.len_Pr + self.len_PC + self.len_S + causalrelations][0])
			# 1.2.8.2 For the policy and instrument trees of all agents [3S]
			if AS_theory == 2 and PF_theory == 2:
				for agents in self.agent_action_list:
					for who in range(len(self.agent_action_list)):
						# For the belief tree policy
						# Go through each of the policies defined by the modeller
						for policy_number in range(len(self.policies)):
							# Go through each of the PC issues
							for issue_considered in range(self.len_PC):
								agents.belieftree_policy[1+who][policy_number][issue_considered] = copy.copy(agents.belieftree_policy[0][policy_number][issue_considered] + random.random() - 0.5)
						# For the belief tree instrument
						# Go through each of the policies defined by the modeller
						for policy_number in range(len(self.instruments)):
							# Go through each of the PC issues
							for issue_considered in range(self.len_S):
								agents.belieftree_instrument[1+who][policy_number][issue_considered] = copy.copy(agents.belieftree_instrument[0][policy_number][issue_considered] + random.random() - 0.5)
		
		print('... cleared.')
		print('   ')


		####################################################################################################
		# 1.2.9 - Agenda setting
		print('   ')
		print('--- Agenda setting ---')
		print('   ')
		
		# Structure of the agenda setting
		# 1.2.10. Agent issue classification
		# 1.2.11. The actions
		# 	1.2.11.1. The group actions
		#		1.2.11.1.1. The team actions
		#		1.2.11.1.2. The coalition actions
		#	1.2.11.21. The agent actions
		#		1.2.11.2.1. Distribution of the resources
		#		1.2.11.2.2. Upkeep of the network
		#		1.2.11.2.3. Belief actions of the agents
		# 1.2.12.  The creation of the agenda

		####################################################################################################

		# 1.2.10. Agent issues classification and selection [Backbone/Backbone+]
		print('Issue selection ...')

		# 1.2.10.1 Preference updates for all agents
		# [Backbone/Backbone+/3S/ACF]
		# This update is only performed on the principle core and policy core issues. It includes the 
		# partial knowledge parts of the belief tree.
		for agents in self.master_list:
			if type(agents) == Electorate:
				self.preference_udapte_electorate(agents)
		for agents in self.agent_action_list:
			if type(agents) != Policyentres:
				for who in range(len(self.agent_action_list) + 1):
					self.preference_udapte(agents, who)

		# [Backbone+/3S/ACF]
		if AS_theory != 0 or PF_theory != 0:
			for agents in self.agent_action_list:
				if type(agents) == Policyentres:
					for who in range(len(self.agent_action_list) + 1):
						self.preference_udapte(agents, who)

		# 1.2.10.2. Conflict level update for all links
		# [Backbone/Backbone+/3S/ACF]
		self.conflict_level_update(self.link_list, self.deep_core, self.policy_core, self.secondary, self.conflict_level_coef)

		# 1.2.10.3. Issue selection
		if AS_theory != 2:
			# [Bacbkone/Backbone+/ACF]
			for agents in self.agent_action_list:
				if type(agents) != Policyentres:
					self.issue_selection(agents)
			# [Backbone+/ACF]
			if AS_theory == 1 or AS_theory == 3:
				for agents in self.agent_action_list:
					if type(agents) == Policyentres:
						self.issue_selection(agents)
		else:
			# [3S]
			for agents in self.agent_action_list:
				self.issue_selection_as_3S(agents)

		print('... cleared.')
		print('   ')
		
		# 1.2.11. The actions

		# 1.2.11.1. The group actions [3S/ACF]

		# 1.2.11.1.1. The team actions [3S]
		if AS_theory == 2:
			print('Team actions (AS) for three streams ...')

			# 1.2.11.1.1.1. Assigning the total resources
			for agents in self.agent_action_list:
				agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
				agents.resources[1] = agents.resources[0]

			# 1.2.11.1.1.2. Agent-Team actions
			shuffled_list_agent = self.agent_action_list
			random.shuffle(shuffled_list_agent)
			for agents in shuffled_list_agent:
				agents.agent_team_threeS_as(agents, self.agent_action_list, self.team_list_as, self.team_list_as_total, self.link_list, self.team_number_as, \
					self.tick_number, self.threeS_link_list_as, self.deep_core, self.policy_core, self.secondary, self.team_gap_threshold, self.team_belief_problem_threshold, self.team_belief_policy_threshold)

			# 1.2.11.1.1.3. Belief actions in a team
			conflict_level_option = 1
			shuffled_team_list_as = self.team_list_as
			random.shuffle(shuffled_team_list_as)
			for teams in shuffled_team_list_as:
				teams.team_belief_actions_threeS_as(teams, self.causalrelation_number, self.deep_core, self.policy_core, self.secondary, self.agent_action_list, self.threeS_link_list_as, self.threeS_link_list_as_total, \
					self.threeS_link_id_as, self.link_list, self.affiliation_weights, self.conflict_level_coef, self.resources_weight_action, self.resources_potency, conflict_level_option)
		
			print('... cleared.')
			print('   ')

		# 1.2.11.1.2. The coalition actions [ACF]
		if AS_theory == 3:

			print('Coalition actions (AS) for ACF ...')

			print('WHAT IS THIS?')
			target = 1

			# 1.2.11.1.2.1. Assigning the total resources
			for agents in self.agent_action_list:
				agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
				agents.resources[1] = agents.resources[0]

			# 1.2.11.1.2.2. Agent-coalition actions
			# 1.2.11.1.2.2.1. Reset of coalition assginment
			for agents in self.agent_action_list:
				agents.coalition_as[0] = None
				agents.coalition_as[1] = None
			self.coalitions_list_as = []
			self.ACF_link_list_as = []

			# 1.2.11.1.2.2.2. Creation of the coalitions
			self.coalition_creation_as(self.agent_action_list, self.link_list, self.Pr_ACF_interest, self.coalitions_number_as, self.tick_number, self.coalitions_list_as, self.coalitions_list_as_total, self.coalition_threshold, target)

			# 1.2.11.1.2.3. Belief actions in a team
			shuffled_coalition_list_as = self.coalitions_list_as
			random.shuffle(shuffled_coalition_list_as)
			for coalitions in shuffled_coalition_list_as:
				coalitions.coalition_belief_actions_ACF_as(coalitions, self.causalrelation_number, self.deep_core, self.policy_core, self.secondary, self.agent_action_list, self.ACF_link_list_as, self.ACF_link_list_as_total, \
					self.ACF_link_id_as, self.link_list, self.affiliation_weights, self.conflict_level_coef, self.resources_weight_action, self.resources_potency)
		
			print('... cleared.')
			print('   ')

		# 1.2.11.2 The actions of the actors [Backbone+/3S/ACF]
		if AS_theory == 1 or AS_theory == 2 or AS_theory == 3:

			# 1.2.11.2.1. Distribution of the resources
			# [Backbone+]
			if AS_theory == 1:
				for agents in self.agent_action_list:
					agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
					# Setting up the temporary resources for network upkeep and agent actions
					agents.resources_network = 0.2 * agents.resources[0]
					agents.resources_actions = 0.8 * agents.resources[0]
			
			# [3S] - The difference is related to the belonging level
			if AS_theory == 2:
				for agents in self.agent_action_list:
					# Set the resources equal to the leftover from the belonging values (only if the agent is in a team)
					if agents.team_as[1] != None:
						agents.resources[0] = 1 - agents.team_as[1]
					else:
						agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
					agents.resources_network = 0.2 * agents.resources[0]
					agents.resources_actions = 0.8 * agents.resources[0]
			
			# [ACF] - The difference is related to the belonging level
			if AS_theory == 3:
				for agents in self.agent_action_list:
					# Set the resources equal to the leftover from the belonging values (only if the agent is in a team)
					if agents.coalition_as[1] != None:
						agents.resources[0] = 1 - agents.coalition_as[1]
					else:
						agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
					agents.resources_network = 0.2 * agents.resources[0]
					agents.resources_actions = 0.8 * agents.resources[0]
			
			# 1.2.11.2.2. Upkeep of the network
			print('Running network upkeep actions (AS) ...')
			# Making a shuffled list of agents to have a random selection order
			# The resource weight and resource adequacy parameters will also have to be moved as input parameters			
			shuffled_list_agent = self.agent_action_list
			random.shuffle(shuffled_list_agent)
			for agents in shuffled_list_agent:
				agents.network_upkeep_as(agents, self.link_list, self.affiliation_weights, AS_theory)

			print('... cleared.')
			print('   ')

			# 1.2.11.2.3. Belief actions of the agents
			print('Performing individual agent actions (AS) ...')

			# [Backbone/Backbone+/ACF]
			if AS_theory != 2:
				random.shuffle(shuffled_list_agent)
				for agents in shuffled_list_agent:
					# The external parties
					if type(agents) == Externalparties:
						agents.external_parties_actions_as(agents, self.agent_action_list, self.causalrelation_number, self.affiliation_weights, \
							self.deep_core, self.policy_core, self.secondary, self.electorate_number, self.action_agent_number, self.master_list, self.link_list, self.resources_weight_action, self.resources_potency)

					# The policy makers and policy entrepreneurs
					# Shuffle of the list of links for the actions
					link_list_shuffle = self.link_list
					random.shuffle(link_list_shuffle)
					if type(agents) != Externalparties:
						agents.pm_pe_actions_as(agents, link_list_shuffle, self.deep_core, self.policy_core, self.secondary, \
							self.resources_weight_action, self.resources_potency, self.affiliation_weights)

			# [3S]
			if AS_theory == 2:
				random.shuffle(shuffled_list_agent)
				for agents in shuffled_list_agent:
					# 1 - Looking at the external parties
					if type(agents) == Externalparties:
						agents.external_parties_actions_as_3S(agents, self.agent_action_list, self.causalrelation_number, self.affiliation_weights, self.deep_core, self.policy_core, self.secondary, \
							self.electorate_number, self.action_agent_number, self.master_list, self.link_list, self.conflict_level_coef, self.resources_weight_action, self.resources_potency)

					# 2 - Looking at the policy makers and policy entrepreneurs
					# Shuffle of the list of links for the actions
					link_list_shuffle = self.link_list
					random.shuffle(link_list_shuffle)
					if type(agents) != Externalparties :
						agents.pm_pe_actions_as_3S(agents, link_list_shuffle, self.deep_core, self.policy_core, self.secondary, \
							self.resources_weight_action, self.resources_potency, self.affiliation_weights, self.conflict_level_coef)

			print('... cleared.')
			print('   ')

		else:
			# In case the backbone only is run - display this message
			print('[Backbone only] - No network upkeep actions and belief actions ')
			print('  ')
		
		# 1.2.12. Creation of the agenda
		# [Backbone/Backbone+/3S/ACF]
		print('Issue selection for the agenda ...')
		
		# 1.2.12.1. Agent issue classification
		for agents in self.agent_action_list:
			if AS_theory != 2 and type(agents) == Policymakers:
				# [Backbone/Backbone+/ACF]
				self.issue_selection(agents)
			if AS_theory == 2 and type(agents) == Policymakers:
				# [3S]
				self.issue_selection_as_3S(agents)

		print('... cleared.')
		print('   ')


		# 1.2.12.2. Agenda creation
		if AS_theory != 2:
			# [Backbone/Backbone+/ACF]
			self.agenda_selection()
			print('AGENDA - The issue is: ' + str(self.agenda_as_issue))

		if AS_theory == 2:
			# [3S]
			self.agenda_selection_3S()
			print('AGENDA - The problem is: ' + str(self.agenda_prob_3S_as) + ' and the policy is: ' + str(self.agenda_poli_3S_as))

		print('   ')
		
		# print(' ')
		# print('Checks:')
		# print('Number of teams: ' + str(len(self.team_list_as)))
		# print('Number of teams ever: ' + str(len(self.team_list_as_total)))
		# print('Number of team links: ' + str(len(self.threeS_link_list_as)))
		# print('Number of team links ever: ' + str(len(self.threeS_link_list_as_total)))
		# for teams_check in self.team_list_as:
		# 	list_links_check = []
		# 	print(' ')
		# 	print(teams_check)
		# 	for links in self.threeS_link_list_as:
		# 		if links.agent1 == teams_check:
		# 			# print('' + str(links))
		# 			list_links_check.append(links)
		# 	print(str(teams_check) + ', number of team members: ' + str(len(teams_check.members)) + ' and number of links:' + str(len(list_links_check)))

		####################################################################################################
		# 1.2.13 - Policy formulation
		print('   ')
		print('--- Policy formulation ---')
		print('   ')
		####################################################################################################

		# 1.2.14 Agent instrument classification
		print('Instrument selection ...')

		# 1.2.14.1 Conflict level update
		self.conflict_level_update(self.link_list, self.deep_core, self.policy_core, self.secondary, self.conflict_level_coef)
		
		# 1.2.14.2 Policy instrument selection
		# [Backbone/Backbone+/ACF]
		if PF_theory != 2:
			for agents in self.agent_action_list:
				if type(agents) == Policymakers:
					# Ranking of the instruments
					for who in range(len(self.agent_action_list) + 1):
						self.instrument_preference_update(agents, who, AS_theory)
					# Choose instrument
					self.instrument_selection(agents)
			# [Backbone+/ACF]
			if PF_theory == 1 or PF_theory == 3:
				for agents in self.agent_action_list:
					if type(agents) == Externalparties or type(agents) == Policyentres:
						for who in range(len(self.agent_action_list) + 1):
							self.instrument_preference_update(agents, who, AS_theory)
						self.instrument_selection(agents)
		# [3S]
		else:
			for agents in self.agent_action_list:
				if type(agents) == Policymakers or type(agents) == Externalparties or type(agents) == Policyentres:
					self.issue_selection_pf_3S(agents)

		print('... cleared.')
		print('   ')

		# 1.2.15. The actions

		# 1.2.15.1. The group actions [3S/ACF]

		# 1.2.15.1.1. The team actions [3S]
		if AS_theory == 2:

			print('Team actions (AS) for three streams ...')

			# 1.2.15.1.1.1. Assigning the total resources
			for agents in self.agent_action_list:
				agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
				agents.resources[1] = agents.resources[0]

			# 1.2.15.1.1.2. Agent-Team actions
			shuffled_list_agent = self.agent_action_list
			random.shuffle(shuffled_list_agent)
			for agents in shuffled_list_agent:
				agents.agent_team_threeS_pf(agents, self.agent_action_list, self.team_list_pf, self.team_list_pf_total, self.link_list, self.team_number_pf, \
					self.tick_number, self.threeS_link_list_pf, self.deep_core, self.policy_core, self.secondary, self.agenda_prob_3S_as, self.team_gap_threshold, \
					self.team_belief_problem_threshold, self.team_belief_policy_threshold)

			# 1.2.15.1.1.3. Belief actions in a team
			conflict_level_option = 1
			shuffled_team_list_as = self.team_list_as
			random.shuffle(shuffled_team_list_as)
			for teams in shuffled_team_list_as:
				teams.team_belief_actions_threeS_pf(teams, self.causalrelation_number, self.deep_core, self.policy_core, self.secondary, self.agent_action_list, self.threeS_link_list_pf, self.threeS_link_list_pf_total, \
					self.threeS_link_id_pf, self.link_list, self.affiliation_weights, self.agenda_prob_3S_as, self.conflict_level_coef, self.resources_weight_action, self.resources_potency, conflict_level_option)

			print('... cleared.')
			print('   ')

		# 1.2.15.1.2. The coalition actions [ACF]
		if AS_theory == 3:

			print('Coalition actions (PF) for ACF ...')

			# 1.2.15.1.2.1. Assigning the total resources
			for agents in self.agent_action_list:
				agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
				agents.resources[1] = agents.resources[0]

			# 1.2.15.1.2.2. Agent-coalition actions

			# 1.2.15.1.2.2.1. Reset of coalition assginment
			for agents in self.agent_action_list:
				agents.coalition_pf[0] = None
				agents.coalition_pf[1] = None
			self.coalitions_list_pf = []
			self.ACF_link_list_pf = []

			# 1.2.15.1.2.2.2. Creation of the coalitions
			self.coalition_creation_pf(self.agent_action_list, self.link_list, self.agenda_as_issue, self.tick_number, self.coalitions_number_pf, self.coalitions_list_pf, self.coalitions_list_pf_total, self.coalition_threshold, target)

			# 1.2.16.1.2.2.3. Belief actions in a coalition
			shuffled_coalition_list_pf = self.coalitions_list_pf
			random.shuffle(shuffled_coalition_list_pf)
			for coalitions in shuffled_coalition_list_pf:
				coalitions.coalition_belief_actions_ACF_pf(coalitions, self.causalrelation_number, self.deep_core, self.policy_core, self.secondary, self.agent_action_list, self.ACF_link_list_pf, self.ACF_link_list_pf_total, \
					self.ACF_link_id_pf, self.link_list, self.affiliation_weights, self.agenda_as_issue, self.instruments, self.conflict_level_coef, self.resources_weight_action, self.resources_potency)
		
			print('... cleared.')
			print('   ')

		# 1.2.15.2. The actions of the actors [Backbone+/3S/ACF]
		if PF_theory == 1 or PF_theory == 2 or PF_theory == 3:

			# 1.2.15.2.1. Distribution of the resources
			# [Backbone+]
			for agents in self.agent_action_list:
				agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
				# Setting up the temporary resources for network upkeep and agent actions
				agents.resources_network = 0.2 * agents.resources[0]
				agents.resources_actions = 0.8 * agents.resources[0]

			# [3S] - The difference is related to the belonging level
			if AS_theory == 2:
				for agents in self.agent_action_list:
					# Set the resources equal to the leftover from the belonging values (only if the agent is in a team)
					if agents.team_pf[1] != None:
						agents.resources[0] = 1 - agents.team_pf[1]
					else:
						agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
					agents.resources_network = 0.2 * agents.resources[0]
					agents.resources_actions = 0.8 * agents.resources[0]

			# [ACF] - The difference is related to the belonging level
			if AS_theory == 3:
				for agents in self.agent_action_list:
					# Set the resources equal to the leftover from the belonging values (only if the agent is in a team)
					if agents.coalition_pf[1] != None:
						agents.resources[0] = 1 - agents.coalition_pf[1]
					else:
						agents.resources[0] = 0.5 + self.representation[agents.affiliation]/2
					agents.resources_network = 0.2 * agents.resources[0]
					agents.resources_actions = 0.8 * agents.resources[0]
			
			# 1.2.15.2.2. Upkeep of the network
			print('Running network upkeep action (PF) ...')
			# Making a shuffled list of agents to have a random selection order
			shuffled_list_agent = self.agent_action_list
			random.shuffle(shuffled_list_agent)
			for agents in shuffled_list_agent:
				agents.network_upkeep_pf(agents, self.link_list, self.affiliation_weights, self.agenda_as_issue, self.agenda_prob_3S_as, PF_theory)

			print('... cleared.')
			print('   ')
		
			# 1.2.15.3. Belief actions of the agents
			print('Performing individual agent actions (PF) ...')

			# [Backbone+/Backbone/ACF]
			if PF_theory != 2:
				random.shuffle(shuffled_list_agent)
				for agents in shuffled_list_agent:
					# The external parties
					if type(agents) == Externalparties:
						agents.external_parties_actions_pf(agents, self.agent_action_list, self.causalrelation_number, \
							self.affiliation_weights, self.deep_core, self.policy_core, self.secondary, self.electorate_number, \
							self.action_agent_number, self.agenda_as_issue, self.instruments, self.master_list, self.link_list, self.resources_weight_action, self.resources_potency)

					# The policy makers and policy entrepreneurs
					# Shuffle of the list of links for the actions
					link_list_shuffle = self.link_list
					random.shuffle(link_list_shuffle)
					if type(agents) == Policymakers or type(agents) == Policyentres:
						agents.pm_pe_actions_pf(agents, self.link_list, self.deep_core, self.policy_core, self.secondary, self.causalrelation_number, \
							self.agenda_as_issue, self.instruments, self.resources_weight_action, self.resources_potency, AS_theory, self.affiliation_weights)
			
			# [3S]
			if PF_theory == 2:
				random.shuffle(shuffled_list_agent)
				for agents in shuffled_list_agent:
					# 1 - Looking at the external parties
					if type(agents) == Externalparties:
						agents.external_parties_actions_pf_3S(agents, self.agent_action_list, self.causalrelation_number, self.affiliation_weights, self.deep_core, self.policy_core, self.secondary, \
							self.electorate_number, self.action_agent_number, self.master_list, self.agenda_prob_3S_as, self.link_list, self.conflict_level_coef, self.resources_weight_action, self.resources_potency)

					# 2 - Looking at the policy makers and policy entrepreneurs
					# Shuffle of the list of links for the actions
					link_list_shuffle = self.link_list
					random.shuffle(link_list_shuffle)
					if type(agents) == Policymakers or type(agents) == Policyentres:
						agents.pm_pe_actions_pf_3S(agents, link_list_shuffle, self.deep_core, self.policy_core, self.secondary, \
							self.resources_weight_action, self.resources_potency, self.agenda_prob_3S_as, self.affiliation_weights, self.conflict_level_coef)

			print('... cleared.')
			print('   ')

		else:
			# In case the backbone only is run - display this message
			print('[Backbone only] - No network upkeep actions and normal actions ')
			print('  ')

		# 1.2.16. Agent instrument classification
		# [Backbone/Backbone+/ACF]
		if PF_theory != 2:
			# The policy makers selects and rank the instruments
			for agents in self.agent_action_list:
				if type(agents) == Policymakers:
					self.instrument_preference_update(agents, 0, AS_theory)
					self.instrument_selection(agents)

		# [3S]
		if PF_theory == 2:
			for agents in self.agent_action_list:
				if type(agents) == Policymakers:
					self.issue_selection_pf_3S(agents)


		# 1.2.17. Instrument implementation check
		# [Backbone/Backbone+/ACF]
		if PF_theory != 2:
			self.instrument_implementation_check()

		# [3S]
		if PF_theory == 2:
			self.instrument_implementation_check_3S()


		####################################################################################################
		# 1.2.18 - End of tick procedures
		print('   ')
		print('--- End of tick procedures ---')
		print('   ')
		####################################################################################################

		# 1.2.19 Conflict level update
		self.conflict_level_update(self.link_list, self.deep_core, self.policy_core, self.secondary, self.conflict_level_coef)

		# 1.2.20 Data collection
		print('Data collection process ...')
		self.datacollector.collect(self)
		print('... cleared.')
		print('   ')

		# 1.2.21. Network upkeep procedures
		# Updating the aware and aware decay parameters
		for links in self.link_list:

			# 1.2.21.1. Awareness decay inplementation
			# The aware decays by 1 for every tick
			if links.aware_decay > 0:
				links.aware_decay -= 1

			# 1.2.21.2. Link negative check
			if links.aware_decay < 0:
				links.aware_decay = 0

			# 1.2.21.3 Decay implementation
			if links.aware_decay == 0:
				# Make sure only links with positive aware are considered
				if links.aware > 0:
					links.aware -= self.aware_decay_coefficient
				# Make sure the links that are negative but not -1 are set to 0.
				if links.aware < 0 and links.aware != -1:
					links.aware = 0

	def agenda_selection(self):

		"""
		1.3. The agenda selection function
		===========================

		This function is used to select what will go on the agenda
		based on the issues selected by the policy makers.

		"""	

		# 1.3.1. Selecting the policy maker agents
		agents_policymakers = []
		for agents in self.agent_action_list:
			if type(agents) == Policymakers:
				agents_policymakers.append(agents)

		# 1.3.2. For each PM, selecting the chosen issue
		issue_list = [None for i in range(len(agents_policymakers))]
		for i in range(len(agents_policymakers)):
			issue_list[i] = agents_policymakers[i].select_as_issue

		# 1.3.3. Selecting the most chosen issue
		issue_counter = Counter(issue_list)

		# 1.3.4. Setting the agenda
		self.agenda_as_issue = issue_counter.most_common(1)[0][0]

	def agenda_selection_3S(self):

		"""
		1.4. The agenda selection function - three streams
		===========================

		This function is used to select what will go on the agenda
		based on the issues selected by the policy makers. This is
		the three streams version of the function.

		"""	

		# 1.4.1. Selecting the policy maker agents
		master_list = self.master_list
		agents_policymakers = []
		for agents in self.master_list:
			if type(agents) == Policymakers:
				agents_policymakers.append(agents)

		# 1.4.2. For each PM, selecting the chosen policy and problem
		problems_list = [None for i in range(len(agents_policymakers))]
		policies_list = [None for i in range(len(agents_policymakers))]
		for i in range(len(agents_policymakers)):
			problems_list[i] = agents_policymakers[i].select_problem_3S_as
			policies_list[i] = agents_policymakers[i].select_policy_3S_as
		
		# 1.4.3. Selecting the most chosen policy and problem
		problem_counter = Counter(problems_list)
		policies_counter = Counter(policies_list)

		# 1.4.4. Setting the agenda with policy and problem
		self.agenda_prob_3S_as = problem_counter.most_common(1)[0][0]
		self.agenda_poli_3S_as = policies_counter.most_common(1)[0][0]

	def preference_udapte(self, agent, who):

		"""
		1.5. The preference update function
		===========================

		This function is used to update the preferences of the agents in their
		respective belief trees.

		"""	

		len_Pr = self.len_Pr
		len_PC = self.len_PC
		len_S = self.len_S

		#####
		# 1.5.1. Preference calculation for the principle issues

		# 1.5.1.1. Calculation of the denominator
		Pr_denominator = 0
		for h in range(len_Pr):
			if agent.belieftree[who][h][1] == None or agent.belieftree[who][h][0] == None:
				Pr_denominator = 0
			else:
				Pr_denominator = Pr_denominator + abs(agent.belieftree[who][h][1] - agent.belieftree[who][h][0])
		# print('The denominator is given by: ' + str(Pr_denominator))

		# 1.5.1.2. Selection of the numerator and calculation of the preference
		for i in range(len_Pr):
			# There are rare occasions where the denominator could be 0
			if Pr_denominator != 0:
				agent.belieftree[who][i][2] = abs(agent.belieftree[who][i][1] - agent.belieftree[who][i][0]) / Pr_denominator
			else:
				agent.belieftree[who][i][2] = 0

		#####	
		# 1.5.2 Preference calculation for the policy core issues
		PC_denominator = 0
		# 1.5.2.1. Calculation of the denominator
		for j in range(len_PC):
			PC_denominator = 0
			# print('Selection PC' + str(j+1))
			# print('State of the PC' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + j][0])) # the state printed
			# Selecting the causal relations starting from Pr
			for k in range(len_Pr):
				# Contingency for partial knowledge issues
				if agent.belieftree[who][k][1] == None or agent.belieftree[who][k][0] == None or agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0] == None:
					PC_denominator = 0
				else:
					# print('Causal Relation PC' + str(j+1) + ' - Pr' + str(k+1) + ': ' + str(agent.belieftree[0][len_Pr+len_PC+len_S+j+(k*len_PC)][1]))
					# print('Gap of Pr' + str(k+1) + ': ' + str((agent.belieftree[0][k][1] - agent.belieftree[0][k][0])))
					# Check if causal relation and gap are both positive of both negative
					# print('agent.belieftree[' + str(who) + '][' + str(len_Pr+len_PC+len_S+j+(k*len_PC)) + '][0]: ' + str(agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0]))
					if (agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0] < 0 and (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]) < 0) \
					  or (agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0] > 0 and (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]) > 0):
						PC_denominator = PC_denominator + abs(agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0]*\
						  (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]))
						# print('This is the PC numerator: ' + str(PC_denominator))
					else:
						PC_denominator = PC_denominator	

		# 1.5.2.2. Addition of the gaps of the associated policy core issues
		for i in range(len_PC):
			# Contingency for partial knowledge issues
			if agent.belieftree[who][len_Pr + i][1] == None or agent.belieftree[who][len_Pr + i][0] == None:
				PC_denominator = PC_denominator
			else:
				# print('This is the gap for the PC' + str(i+1) + ': ' + str(agent.belieftree[0][len_Pr + i][1] - agent.belieftree[0][len_Pr + i][0]))
				PC_denominator = PC_denominator + abs(agent.belieftree[who][len_Pr + i][1] - agent.belieftree[who][len_Pr + i][0])
		# print('This is the PC denominator: ' + str(PC_denominator))
		
		# 1.5.2.3 Calculation the numerator and the preference
		# Select one by one the Pr
		for j in range(len_PC):

			# 1.5.2.3.1. Calculation of the right side of the numerator
			PC_numerator = 0
			# print('Selection PC' + str(j+1))
			# print('State of the PC' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + j][0])) # the state printed
			# Selecting the causal relations starting from Pr
			for k in range(len_Pr):
				# Contingency for partial knowledge issues
				if agent.belieftree[who][k][1] == None or agent.belieftree[who][k][0] == None or agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0] == None:
					PC_numerator = 0
				else:
					# print('Causal Relation PC' + str(j+1) + ' - Pr' + str(k+1) + ': ' + str(agent.belieftree[0][len_Pr+len_PC+len_S+j+(k*len_PC)][1]))
					# print('Gap of Pr' + str(k+1) + ': ' + str((agent.belieftree[0][k][1] - agent.belieftree[0][k][0])))
					# Check if causal relation and gap are both positive of both negative
					if (agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0] < 0 and (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]) < 0) \
					  or (agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0] > 0 and (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]) > 0):
						PC_numerator = PC_numerator + abs(agent.belieftree[who][len_Pr+len_PC+len_S+j+(k*len_PC)][0]*\
						  (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]))
						# print('This is the PC numerator: ' + str(PC_numerator))
					else:
						PC_numerator = PC_numerator	

			# 1.5.2.3.2. Addition of the gap to the numerator
			# Contingency for partial knowledge issues
			if agent.belieftree[who][len_Pr + j][1] == None or agent.belieftree[who][len_Pr + j][0] == None:
				PC_numerator = 0
			else:
				# print('This is the gap for the PC' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + j][1] - agent.belieftree[0][len_Pr + j][0]))
				PC_numerator = PC_numerator + abs(agent.belieftree[who][len_Pr + j][1] - agent.belieftree[who][len_Pr + j][0])
			# print('The numerator is equal to: ' + str(PC_numerator))
			# print('The denominator is equal to: ' + str(PC_denominator))

			# 1.5.2.3.3. Calculation of the preference
			if PC_denominator != 0:
				agent.belieftree[who][len_Pr+j][2] = PC_numerator/PC_denominator 
			# print('The new preference of the policy core PC' + str(j+1) + ' is: ' + str(agent.belieftree[0][len_Pr+j][2]))
			else:
				agent.belieftree[who][len_Pr+j][2] = 0

	def preference_udapte_electorate(self, agent):

		"""
		1.6. The electorate preference update function
		===========================

		This function is used to calculate the preferences of the electorate
		agents. It is the similar to the function used to calculate the preferences
		of the other agents. The main difference is the non inclusion of the 
		causal relations (the electorate tree does not have any). Each preference
		is therefore calculated based on the state and aim for each level
		in the tree.

		The calculation of the principle, policy core and secondary issues 
		preferences is performed.

		"""

		len_Pr = self.len_Pr
		len_PC = self.len_PC
		len_S = self.len_S

		#####
		# 1.6.1. Preference calculation for the principle issues
		Pr_denominator = 0
		for h in range(len_Pr):
			Pr_denominator = Pr_denominator + abs(agent.belieftree_electorate[h][1] - agent.belieftree_electorate[h][0])
		for i in range(len_Pr):
			# There are rare occasions where the denominator could be 0
			if Pr_denominator != 0:
				agent.belieftree_electorate[i][2] = abs(agent.belieftree_electorate[i][1] - agent.belieftree_electorate[i][0]) / Pr_denominator
			else:
				agent.belieftree_electorate[i][2] = 0

		#####
		# 1.6.2. Preference calculation for the policy core issues
		PC_denominator = 0
		for h in range(len_PC):
			PC_denominator = PC_denominator + abs(agent.belieftree_electorate[len_Pr + h][1] - agent.belieftree_electorate[len_Pr + h][0])
		for i in range(len_PC):
			# There are rare occasions where the denominator could be 0
			if PC_denominator != 0:
				agent.belieftree_electorate[len_Pr + i][2] = abs(agent.belieftree_electorate[len_Pr + i][1] - agent.belieftree_electorate[len_Pr + i][0]) / PC_denominator
			else:
				agent.belieftree_electorate[len_Pr + i][2] = 0

		#####
		# 1.6.3. Preference calculation for the secondary issues
		S_denominator = 0
		for h in range(len_S):
			S_denominator = S_denominator + abs(agent.belieftree_electorate[len_Pr + len_PC + h][1] - agent.belieftree_electorate[len_Pr + len_PC + h][0])
		for i in range(len_S):
			# There are rare occasions where the denominator could be 0
			if S_denominator != 0:
				agent.belieftree_electorate[len_Pr + len_PC + i][2] = abs(agent.belieftree_electorate[len_Pr + len_PC + i][1] - agent.belieftree_electorate[len_Pr + len_PC + i][0]) / S_denominator
			else:
				agent.belieftree_electorate[len_Pr + len_PC + i][2] = 0

	def issue_selection(self, agent):

		"""
		1.7. The issue selection function
		===========================

		This function is used to select the best preferred issue for 
		each of the agents.

		"""	
		
		len_Pr = self.len_Pr
		len_PC = self.len_PC
		len_S = self.len_S

		# 1.7.1. Assigning all policy core issue preferences into an array
		as_issue = [None for k in range(len_PC)]
		for i in range(len_PC):
			as_issue[i] = agent.belieftree[0][len_Pr + i][2]
		
		# 1.7.2. Selection of the highest preference policy core issue and setting to agent selected issue
		agent.select_as_issue = len_Pr + as_issue.index(max(as_issue))

	def issue_selection_as_3S(self, agent):

		"""
		1.8. Issue selection function (three streams theory) - agenda setting
		===========================

		This function is used to obtain the problem and policy choice of the agents within
		the framework of the three streams theory. The function is split in five parts: the 
		problem grading, the policy grading, the issue selection, the problem selection and
		the policy selection.

		1/ Grading of the problem

		2/ Grading of the policy

		3/ Selection of the issue

		4/ Selection of the problem

		5/ Selection of the policy

		"""

		len_Pr = self.len_Pr
		len_PC = self.len_PC
		len_S = self.len_S

		# print(' ')

		###########################
		# Grading of the problem ##
		###########################
	
		total_list = []
		grade_prob_list = []

		# Calculating the numerator and the preference of all policy core issues:
		# Select one by one the Pr
		for j in range(len_PC):
			grade_prob = 0
			# print('Selection PC' + str(j+1))
			# print('State of the PC' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + j][0])) # the state printed
			# Selecting the causal relations starting from Pr
			for k in range(len_Pr):
				# print(' ')
				# print(len_Pr + len_PC + len_S + j + k*len_PC)
				# Contingency for partial knowledge issues
				if agent.belieftree[0][k][1] != None and agent.belieftree[0][k][0] != None:
					# print('Causal Relation PC' + str(j+1) + ' - Pr' + str(k+1) + ': ' + str(agent.belieftree[0][len_Pr+len_PC+len_S+(j+(k*len_PC))][0]))
					# print('Gap of Pr' + str(k+1) + ': ' + str(agent.belieftree[0][k][1] - agent.belieftree[0][k][0]))
					# Check if causal relation and gap are both positive of both negative
					if (agent.belieftree[0][len_Pr+len_PC+len_S+(j+(k*len_PC))][0] < 0 and (agent.belieftree[0][k][1] - agent.belieftree[0][k][0]) < 0) \
						or (agent.belieftree[0][len_Pr+len_PC+len_S+(j+(k*len_PC))][0] > 0 and (agent.belieftree[0][k][1] - agent.belieftree[0][k][0]) > 0):
						# print('Calculating')
						grade_prob = grade_prob + abs(agent.belieftree[0][len_Pr+len_PC+len_S+(j+(k*len_PC))][0] * \
							(agent.belieftree[0][k][1] - agent.belieftree[0][k][0]))
						# print('This is the PC numerator: ' + str(grade_prob))
					else:
						grade_prob = grade_prob
				else:
					grade_prob = 0
			# Contingency for partial knowledge issues
			if agent.belieftree[0][len_Pr + j][1] == None or agent.belieftree[0][len_Pr + j][0] == None:
				grade_prob = 0
			else:
				# Then adding the gap of the policy core:
				# print('This is the gap for the PC' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + j][1] - agent.belieftree[0][len_Pr + j][0]))
				grade_prob = grade_prob + abs(agent.belieftree[0][len_Pr + j][1] - agent.belieftree[0][len_Pr + j][0])
			grade_prob_list.append(grade_prob)
			total_list.append(grade_prob)
		
		# print(grade_prob_list)

		###########################
		# Grading of the policy ###
		###########################

		grade_poli_list = []

		# Going through each of the policies
		for p in range(len(agent.belieftree_policy[0])):
			# Going through each of the policy core issues
			grade_poli = 0
			for j in range(len_PC):
				# print('Policy number: ' + str(p) + ' with impact: ' + str(policies[p][j]))
				# Calculating the grade of the policy
				grade_poli = grade_poli + (agent.belieftree[0][len_Pr + j][1] - (agent.belieftree[0][len_Pr + j][0] * agent.belieftree_policy[0][p][j]))
			grade_poli_list.append(grade_poli)
			total_list.append(grade_poli)

		# print(grade_poli_list)

		# print(total_list)
		# print(max(total_list))

		############################
		# Selection of the issue ###
		############################

		issue_index = total_list.index(max(total_list))

		# print('Index: ' + str(issue_index))
		# print('Problem length:' + str(len(grade_prob_list)))

		if issue_index < len(grade_prob_list):
			agent.select_issue_3S_as = 'problem'
		else:
			agent.select_issue_3S_as = 'policy'

		# print(agent.select_issue_3S_as)

		############################
		# Selection of the problem #
		############################

		agent.select_problem_3S_as = len_Pr + grade_prob_list.index(max(grade_prob_list))
		# print(agent.select_problem_3S_as)

		############################
		# Selection of the policy ##
		############################

		agent.select_policy_3S_as = grade_poli_list.index(max(grade_poli_list))
		# print(agent.select_policy_3S_as)

	def issue_selection_pf_3S(self, agent):

		"""
		Issue selection function (three streams theory) - policy formulation
		===========================

		This function is used to obtain the problem and policy choice of the agents within
		the framework of the three streams theory. The function is split in five parts: the 
		problem grading, the policy grading, the issue selection, the problem selection and
		the policy selection.

		1/ Grading of the problem

		2/ Grading of the policy

		3/ Selection of the issue

		4/ Selection of the problem

		5/ Selection of the policy

		"""

		len_Pr = self.len_Pr
		len_PC = self.len_PC
		len_S = self.len_S

		# print(' ')

		###########################
		# Grading of the problem ##
		###########################
	
		total_list = []
		grade_prob_list = []

		k = self.agenda_prob_3S_as

		# Calculating the numerator and the preference of all policy core issues:
		# Select one by one the Pr
		for j in range(len_S):
			grade_prob = 0
			# print('Selection S' + str(j+1))
			# print('State of the S' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + j][0])) # the state printed
			# Selecting the causal relations starting from PC
			# print(' ')
			# print(len_Pr + len_PC + len_S + len_Pr*len_PC + j*len_PC + (k-len_Pr))
			# Contingency for partial knowledge issues
			if agent.belieftree[0][k][1] != None and agent.belieftree[0][k][0] != None:
				# print('Causal Relation S' + str(j+1) + ' - PC' + str(k+1) + ': ' + str(agent.belieftree[0][len_Pr+len_PC+len_S+(j+(k*len_PC))][0]))
				# print('Gap of PC' + str(k+1) + ': ' + str(agent.belieftree[0][k][1] - agent.belieftree[0][k][0]))
				# Check if causal relation and gap are both positive of both negative
				if (agent.belieftree[0][len_Pr + len_PC + len_S + len_Pr*len_PC + j*len_PC + (k-len_Pr)][0] < 0 and (agent.belieftree[0][k][1] - agent.belieftree[0][k][0]) < 0) \
					or (agent.belieftree[0][len_Pr + len_PC + len_S + len_Pr*len_PC + j*len_PC + (k-len_Pr)][0] > 0 and (agent.belieftree[0][k][1] - agent.belieftree[0][k][0]) > 0):
					# print('Calculating')
					grade_prob = grade_prob + abs(agent.belieftree[0][len_Pr + len_PC + len_S + len_Pr*len_PC + j*len_PC + (k-len_Pr)][0] * \
						(agent.belieftree[0][k][1] - agent.belieftree[0][k][0]))
					# print('This is the PC numerator: ' + str(grade_prob))
				else:
					grade_prob = grade_prob
			else:
				grade_prob = 0
			# Contingency for partial knowledge issues
			if agent.belieftree[0][len_Pr + j][1] == None or agent.belieftree[0][len_Pr + j][0] == None:
				grade_prob = grade_prob
			else:
				# Then adding the gap of the policy core:
				# print('This is the gap for the S' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + j][1] - agent.belieftree[0][len_Pr + j][0]))
				grade_prob = grade_prob + abs(agent.belieftree[0][len_Pr + j][1] - agent.belieftree[0][len_Pr + j][0])
			grade_prob_list.append(grade_prob)
			total_list.append(grade_prob)
		
		# print(grade_prob_list)

		###########################
		# Grading of the policy ###
		###########################

		grade_poli_list = []

		# Going through each of the policies
		for p in range(len(agent.belieftree_instrument[0])):
			# Going through each of the policy core issues
			grade_poli = 0
			for j in range(len_S):
				# Calculating the grade of the policy
				grade_poli = grade_poli + (agent.belieftree[0][len_Pr + len_PC + j][1] - (agent.belieftree[0][len_Pr + len_PC + j][0] * agent.belieftree_instrument[0][p][j]))
			grade_poli_list.append(grade_poli)
			total_list.append(grade_poli)

		# print(grade_poli_list)

		# print(total_list)
		# print(max(total_list))

		############################
		# Selection of the issue ###
		############################

		issue_index = total_list.index(max(total_list))

		# print('Index: ' + str(issue_index))
		# print('Problem length:' + str(len(grade_prob_list)))

		if issue_index < len(grade_prob_list):
			agent.select_issue_3S_pf = 'problem'
		else:
			agent.select_issue_3S_pf = 'policy'

		# print(agent.select_issue_3S_as)

		############################
		# Selection of the problem #
		############################

		agent.select_problem_3S_pf = len_Pr + len_PC + grade_prob_list.index(max(grade_prob_list))
		# print(agent.select_problem_3S_pf)

		############################
		# Selection of the policy ##
		############################

		agent.select_policy_3S_pf = grade_poli_list.index(max(grade_poli_list))
		# print(agent.select_policy_3S_pf)

	def instrument_preference_update(self, agent, who, AS_theory):

		"""
		Instrument preference update function
		===========================

		This function is used to calculate the ranking of each of the instrument from 
		which the agents can choose from. This is done in two parts.

		1/ The first part consists of calculating the preference level for the different
		secondary issues (layer 3 in the belief tree). In this part, the preferences of
		the agents are updated similarly to the function where the preferences are calculated.
		The main difference is that this time, it is based on the agenda which means that
		only the secondary issues affecting the problem on the agenda are considered.

		2/ The second part consists of obtaining the grade for the policy instruments.
		This is calculated as shown in the formalisation with the equation given by:
		G = sum(impact * (Aim - State) * Preference_secondary)
		We make sure that the instruments impact are only taken into account if the
		impact is of the same sign as the gap between the state and the aim for the
		specific secondary issues. If this is not the case, the impact is not considered
		for that specific part of the instrument.

		Notes:
		1/ The secondary issues for which the agent is not interested (this applies to 
		the external parties only) are not taken into account in the calculation. They
		are marked as the 'No' values.

		"""

		len_Pr = self.len_Pr
		len_PC = self.len_PC
		len_S = self.len_S
		instruments = self.instruments

		######################################################################################################
		# 1/ Calculation of the preference level for the secondary issues based on the problem on the agenda #
		######################################################################################################

		# Calculation of the denominator first
		S_denominator = 0
		# Select one by one the Pr
		## We dont want to go through all PC but just one PC - the question is then to change all the j for what number it should be
		if AS_theory != 2:
			j = agent.select_as_issue
		if AS_theory == 2:
			j = agent.select_problem_3S_as
		# for j in range(len_PC):
		# print('Selection PC' + str(j+1))
		# print('State of the PC' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr+j][0])) # the state printed
		# Selecting the causal relations starting from Pr
		for k in range(len_S):
			# print('Causal Relation PC' + str(j+1) + ' - S' + str(k+1) + ': ' + str(agent.belieftree[0][len_Pr+len_PC+len_S+(len_Pr*len_PC)+(j+k)][1]))
			# print('Gap of PC' + str(j+1) + ': ' + str((agent.belieftree[0][j][1] - agent.belieftree[0][j][0])))
			# Check is gap of PC and causal relation to PC are both negative or both positive!
			# Check if causal relation and gap are both positive of both negative
			# Pass the issues - pass the first layer of links - pass the unrelated links
			if agent.belieftree[who][j][1] != None and agent.belieftree[who][j][0] != None and agent.belieftree[who][len_Pr + len_PC + len_S + (len_Pr*len_PC) + (j - len_Pr)*len_S + k][0] != None:
				if (agent.belieftree[who][len_Pr + len_PC + len_S + (len_Pr*len_PC) + (j - len_Pr)*len_S + k][0] < 0 and (agent.belieftree[who][j][1] - agent.belieftree[who][j][0]) < 0) \
					or (agent.belieftree[who][len_Pr + len_PC + len_S + (len_Pr*len_PC) + (j - len_Pr)*len_S + k][0] > 0 and (agent.belieftree[who][j][1] - agent.belieftree[who][j][0]) > 0):
					S_denominator = S_denominator + abs(agent.belieftree[who][len_Pr + len_PC + len_S + (len_Pr*len_PC) + (j - len_Pr)*len_S + k][0]*\
					  (agent.belieftree[who][j][1] - agent.belieftree[who][j][0]))
					# print('This is the S denominator: ' + str(S_denominator))
				else:
					S_denominator = S_denominator
			else:
				S_denominator = S_denominator
			# Then adding the gap of the policy core:

		for i in range(len_S):
			if agent.belieftree[who][len_Pr + len_PC + i][0] != 'No':
				# print('This is the gap for the S' + str(i+1) + ': ' + str(agent.belieftree[0][len_Pr + len_PC + i][1] - agent.belieftree[0][len_Pr + len_PC + i][0]))
				if agent.belieftree[who][len_Pr + len_PC + i][1] != None and agent.belieftree[who][len_Pr + len_PC + i][0] != None:
					S_denominator = S_denominator + abs(agent.belieftree[who][len_Pr + len_PC + i][1] - agent.belieftree[who][len_Pr + len_PC + i][0])
				else:
					S_denominator = 0
		# print('This is the S denominator: ' + str(S_denominator))

		# Calculating the numerator and the preference of all policy core issues:
		S_numerator = 0
		# Select one by one the Pr
		
		for j in range(len_S):
			S_numerator = 0
			# print('Selection S' + str(j+1))
			# print('State of the S' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + len_PC + j][0])) # the state printed
			# Selecting the causal relations starting from PC
			if AS_theory != 2:
				k = agent.select_as_issue
			if AS_theory == 2:
				k = agent.select_problem_3S_as
			# print("This is the index of the selected policy: " + str(k))
			# print('Causal Relation S' + str(j+1) + ' - PC' + str(k+1) + ': ' + str(agent.belieftree[0][len_Pr+len_PC+len_S+(len_Pr*len_PC)+(j+(k*2))][1]))
			# print('Gap of PC' + str(k+1) + ': ' + str((agent.belieftree[0][len_Pr + k][1] - agent.belieftree[0][len_Pr + k][0])))
			# Check if causal relation and gap are both positive of both negative
			if agent.belieftree[who][k][1] != None and agent.belieftree[who][k][0] != None and agent.belieftree[who][len_Pr + len_PC + len_S + (len_Pr*len_PC) + (k - len_Pr)*len_S + j][0] != None:
				if (agent.belieftree[who][len_Pr + len_PC + len_S + (len_Pr*len_PC) + (k - len_Pr)*len_S + j][0] < 0 and (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]) < 0) \
					or (agent.belieftree[who][len_Pr + len_PC + len_S + (len_Pr*len_PC) + (k - len_Pr)*len_S + j][0] > 0 and (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]) > 0):
					S_numerator = S_numerator + abs(agent.belieftree[who][len_Pr + len_PC + len_S + (len_Pr*len_PC) + (k - len_Pr)*len_S + j][0]*\
						  (agent.belieftree[who][k][1] - agent.belieftree[who][k][0]))
					# print('This is the S numerator: ' + str(S_numerator))
				else:
					S_numerator = S_numerator
			else:
				S_numerator = S_numerator
			# Then adding the gap of the policy core:
			# print('This is the gap for the S' + str(j+1) + ': ' + str(agent.belieftree[0][len_Pr + len_PC + j][1] - agent.belieftree[0][len_Pr + len_PC + j][0]))
			if agent.belieftree[who][len_Pr + len_PC + j][0] != 'No':
				if agent.belieftree[who][len_Pr + len_PC + j][1] != None and agent.belieftree[who][len_Pr + len_PC + j][0] != None:
					S_numerator = S_numerator + abs(agent.belieftree[who][len_Pr + len_PC + j][1] - agent.belieftree[who][len_Pr + len_PC + j][0])
				else:
					S_numerator = 0

			# print('The numerator is equal to: ' + str(S_numerator))
			# print('The denominator is equal to: ' + str(S_denominator))
			# print('agent.belieftree[' + str(who) + '][len_Pr + len_PC + ' + str(j) + '][1]: ' + str(agent.belieftree[who][len_Pr + len_PC + j][1]))
			# print('agent.belieftree[' + str(who) + '][len_Pr + len_PC + ' + str(j) + '][0]: ' + str(agent.belieftree[who][len_Pr + len_PC + j][0]))
			
			if S_denominator != 0:
				agent.belieftree[who][len_Pr+len_PC+j][2] = S_numerator/S_denominator 
			else:
				agent.belieftree[who][len_Pr+len_PC+j][2] = 0
			# print('The new preference of the policy core S' + str(j+1) + ' according to the agenda is: ' + str(agent.belieftree[who][len_Pr+len_PC+j][2]))

		# print(agent)
		# print('Secondary preferences: ' + str(agent.belieftree[who][len_Pr+len_PC+j][2]))
		# print(' ')

		##################################################################################################
		# 2/ Calculation of the grade of each of the instruments based on impact on the secondary issues #
		##################################################################################################

		# This is the part where the grade of the policy instruments is actually calculated
		agent.instrument_preferences[who] = [0 for h in range(len(instruments))]
		# go through all instruments
		for i in range(len(instruments)):
			# print('This is the number of instruments considered: ' + str(len(instruments)))
			# go through each of the issues
			for j in range(len_S):
				# print('This is the number of secondary belief affected: ' + str(len_S))
				# print('This is instrument number: ' + str(i) + ' with value ' + str(instruments[i][j]) \
				# 	+ ' with secondary belief: ' + str(j))
				# print('Aim of the actor: ' + str(agent.belieftree[0][len_Pr + len_PC + j][1]))
				# print('State of the actor: ' + str(agent.belieftree[0][len_Pr + len_PC + j][0]))
				# print('This is the gap: ' + str(agent.belieftree[0][len_Pr + len_PC + j][1] - agent.belieftree[0][len_Pr + len_PC + j][0]))
				if agent.belieftree[who][len_Pr + len_PC + j][0] != 'No':
					if agent.belieftree[who][len_Pr + len_PC + j][1] != None and agent.belieftree[who][len_Pr + len_PC + j][0] != None:
						if (instruments[i][j] > 0 and (agent.belieftree[who][len_Pr + len_PC + j][1] - agent.belieftree[who][len_Pr + len_PC + j][0]) > 0 ) \
						  or (instruments[i][j] < 0 and (agent.belieftree[who][len_Pr + len_PC + j][1] - agent.belieftree[who][len_Pr + len_PC + j][0]) < 0 ):
							# print('*************************************************')
							# print('I will now calculate the grade of the instrument:')
							# print('This is the gap in the secondary belief: ' + str(agent.belieftree[0][len_Pr + len_PC + j][1] - agent.belieftree[0][len_Pr + len_PC + j][0]))
							# print('This is the corresponding preference for this secondary belief: ' + str(agent.belieftree[who][len_Pr+len_PC+j][2]))
							# print('And this is the instrument corresponding value: ' + str(instruments[i][j]))

							agent.instrument_preferences[who][i] = agent.instrument_preferences[who][i] + \
								(instruments[i][j] * (agent.belieftree[who][len_Pr + len_PC + j][1] - agent.belieftree[who][len_Pr + len_PC + j][0]) * \
								(agent.belieftree[who][len_Pr+len_PC+j][2]))

							# print(' ')
							# print('agent.instrument_preferences' + str(agent.instrument_preferences[who][i]))
							# print('Who: ' + str(who))
							# print('instruments[' + str(i) + '][' + str(j) + ']: ' + str(instruments[i][j]))
							# print('agent.belieftree[' + str(who) + '][len_Pr + len_PC + ' + str(j) + '][1]: ' + str(agent.belieftree[who][len_Pr + len_PC + j][1]))
							# print('agent.belieftree[' + str(who) + '][len_Pr + len_PC + ' + str(j) + '][0]: ' + str(agent.belieftree[who][len_Pr + len_PC + j][0]))
							# print('instrument_secondary_preferences[' + str(who) + '][' + str(j) + ']: ' + str(agent.belieftree[who][len_Pr+len_PC+j][2]))

							# print('This is the grade of the instrument at this point: ' + str(agent.instrument_preferences[i]))
							# print('*************************************************')
					else:
						agent.instrument_preferences[who][i] = 0
			# print('And this is the preference grade for instrument ' + str(i) + ' with grade: ' + str(agent.instrument_preferences[i]))
			# print('-------------------------------------------------')
		# print(agent)
		# print('Instrument preferences' + str(agent.instrument_preferences))

	def instrument_selection(self, agent):

		"""
		Instrument selection function
		===========================

		This function is used to determine what instrument the agent will choose as a
		prefered option. This is based on the preferences calculated in the ionstrument
		preference update function - instrument_preference_update(). The highest grade
		is simply selected as the instrument of choice for the agent. Note that if no
		instrument has a grade that is higher than one, then the agent does not select
		any of the policy instruments.

		"""
		
		if max(agent.instrument_preferences[0]) != 0.0:
			agent.select_pinstrument = agent.instrument_preferences[0].index(max(agent.instrument_preferences[0]))
		# If all preferences are 0, then set the instrument to -1 to make sure it is not selected
		else:
			agent.select_pinstrument = -1

	def instrument_implementation_check(self):

		"""
		The instrument implementation check function
		===========================

		This function is used at the end of the policy formulation to 
		check whether the instrument selected by the policy makers passes
		the minimum threshold for implementation. If it does, it is implemented.
		If not, nothing will happen and no instrument will be implemented.

		"""

		agents_policymakers = []
		for agents in self.agent_action_list:
			if type(agents) == Policymakers:
				agents_policymakers.append(agents)

		instrument_pref_list = []
		for i in range(len(agents_policymakers)):
			instrument_pref_list.append(agents_policymakers[i].select_pinstrument)
			# print('This loop has ran ' +  str(i+1) + ' times so far and this is the current list: ' + str(instrument_pref_list))
		print(instrument_pref_list)
		count_instrument_pref_list = Counter(instrument_pref_list)
		most_common_instrument = count_instrument_pref_list.most_common(1)
		if most_common_instrument[0][1] > (len(agents_policymakers)/2):
			
			self.agenda_instrument = most_common_instrument[0][0]	
			print('Instrument ' + str(self.agenda_instrument) + ' has been chosen and can be implemented!')
		else:
			print('No instrument meets the majority condition!')

	def instrument_implementation_check_3S(self):

		"""
		The instrument implementation check function - three streams
		===========================

		This function is used at the end of the policy formulation to 
		check whether the instrument selected by the policy makers passes
		the minimum threshold for implementation. If it does, it is implemented.
		If not, nothing will happen and no instrument will be implemented.
		This is the three streams version of the function.

		"""

		agents_policymakers = []
		for agents in self.agent_action_list:
			if type(agents) == Policymakers:
				agents_policymakers.append(agents)

		policy_pref_list = []
		for i in range(len(agents_policymakers)):
			policy_pref_list.append(agents_policymakers[i].select_policy_3S_pf)
			# print('This loop has ran ' +  str(i+1) + ' times so far and this is the current list: ' + str(policy_pref_list))
		print(policy_pref_list)
		count_policy_pref_list = Counter(policy_pref_list)
		most_common_policy = count_policy_pref_list.most_common(1)
		if most_common_policy[0][1] > (len(agents_policymakers)/2):
			
			self.agenda_instrument = most_common_policy[0][0]	
			print('Instrument ' + str(self.agenda_instrument) + ' has been chosen and can be implemented!')
		else:
			print('No instrument meets the majority condition!')

	agent_action_dict = defaultdict(list)

	def add(self, agent):
		agent_class = type(agent)
		self.agent_action_dict[agent_class].append(agent)

	link_list = defaultdict(list)

	def conflict_level_update(self, link_list, deep_core, policy_core, secondary, conflict_level_coef):

		"""
		The conflict level update function
		===========================

		This function is used to calculate the conflict level in the links
		between the agents. It is calculated for the aims and states of the
		issues and for the causal relations.

		"""

		for links in self.link_list:
			# print(links)
			conflict_level_temp = copy.copy(links.conflict_level)
			for issues in range(self.len_Pr + self.len_PC + self.len_S):
				# This is all based on partial knowledge
				# AGENT 1 - based on the partial knowledge he has of 
				# For the calculation of the state conflict level:

				# If one of the beliefs is known to be 'No' then assign 'No' to the conflict level
				if links.agent1.belieftree[1 + links.agent2.unique_id][issues][0] == 'No' or links.agent1.belieftree[0][issues][0] == 'No':
					links.conflict_level[0][issues][0] = 'No'
				# If there is no knowledge of the other agent's beliefs, the conflict level is set to 0.85 by default
				elif links.agent1.belieftree[1 + links.agent2.unique_id][issues][0] == None:
					links.conflict_level[0][issues][0] = conflict_level_coef[1]
				# If all beliefs are known, calculate the conflict level
				else:
					conflict_level_temp[0][issues][0] = abs(links.agent1.belieftree[0][issues][0] - links.agent1.belieftree[1 + links.agent2.unique_id][issues][0])
					if conflict_level_temp[0][issues][0] <= 0.25:
						links.conflict_level[0][issues][0] = conflict_level_coef[0]
					if conflict_level_temp[0][issues][0] > 0.25 and conflict_level_temp[0][issues][0] <= 1.75:
						links.conflict_level[0][issues][0] = conflict_level_coef[2]
					if conflict_level_temp[0][issues][0] > 1.75:
						links.conflict_level[0][issues][0] = conflict_level_coef[1]

				# For the calculation of the aim conflict level:
				if links.agent1.belieftree[1 + links.agent2.unique_id][issues][1] == 'No' or links.agent1.belieftree[0][issues][1] == 'No':
					links.conflict_level[0][issues][1] = 'No'
				elif links.agent1.belieftree[1 + links.agent2.unique_id][issues][1] == None:
					links.conflict_level[0][issues][1] = conflict_level_coef[1]
				else:
					conflict_level_temp[0][issues][1] = abs(links.agent1.belieftree[0][issues][1] - links.agent1.belieftree[1 + links.agent2.unique_id][issues][1])
					if conflict_level_temp[0][issues][1] <= 0.25:
						links.conflict_level[0][issues][1] = conflict_level_coef[0]
					if conflict_level_temp[0][issues][1] > 0.25 and conflict_level_temp[0][issues][1] <= 1.75:
						links.conflict_level[0][issues][1] = conflict_level_coef[2]
					if conflict_level_temp[0][issues][1] > 1.75:
						links.conflict_level[0][issues][1] = conflict_level_coef[1]
				
				# AGENT 2
				# For the calculation of the state conflict level:
				if links.agent2.belieftree[1 + links.agent1.unique_id][issues][0] == 'No' or links.agent2.belieftree[0][issues][0] == 'No':
					links.conflict_level[1][issues][0] = 'No'
				elif links.agent2.belieftree[1 + links.agent1.unique_id][issues][0] == None:
					links.conflict_level[1][issues][0] = conflict_level_coef[1]
				else:
					conflict_level_temp[1][issues][0] = abs(links.agent2.belieftree[0][issues][0] - links.agent2.belieftree[1 + links.agent1.unique_id][issues][0])
					if conflict_level_temp[1][issues][0] <= 0.25:
						links.conflict_level[1][issues][0] = conflict_level_coef[0]
					if conflict_level_temp[1][issues][0] > 0.25 and conflict_level_temp[1][issues][0] <= 1.75:
						links.conflict_level[1][issues][0] = conflict_level_coef[2]
					if conflict_level_temp[1][issues][0] > 1.75:
						links.conflict_level[1][issues][0] = conflict_level_coef[1]
				# For the calculation of the aim conflict level:
				if links.agent2.belieftree[1 + links.agent1.unique_id][issues][1] == 'No' or links.agent2.belieftree[0][issues][1] == 'No':
					links.conflict_level[1][issues][1] = 'No'
				elif links.agent2.belieftree[1 + links.agent1.unique_id][issues][1] == None:
					links.conflict_level[1][issues][1] = conflict_level_coef[1]
				else:
					conflict_level_temp[1][issues][1] = abs(links.agent2.belieftree[0][issues][1] - links.agent2.belieftree[1 + links.agent1.unique_id][issues][1])
					if conflict_level_temp[1][issues][1] <= 0.25:
						links.conflict_level[1][issues][1] = conflict_level_coef[0]
					if conflict_level_temp[1][issues][1] > 0.25 and conflict_level_temp[1][issues][1] <= 1.75:
						links.conflict_level[1][issues][1] = conflict_level_coef[2]
					if conflict_level_temp[1][issues][1] > 1.75:
						links.conflict_level[1][issues][1] = conflict_level_coef[1]

			# Addition of the causal relations conflict level:
			for issues in range(self.causalrelation_number):

				# AGENT 1
				# If one of the beliefs is known to be 'No' then assign 'No' to the conflict level
				if links.agent1.belieftree[1 + links.agent2.unique_id][self.issues_number + issues][0] == 'No' or links.agent1.belieftree[0][self.issues_number + issues][0] == 'No':
					links.conflict_level[0][self.issues_number + issues][0] = 'No'
				# If there is no knowledge of the other agent's beliefs, the conflict level is set to 0.85 by default
				elif links.agent1.belieftree[1 + links.agent2.unique_id][self.issues_number + issues][0] == None:
					links.conflict_level[0][self.issues_number + issues][0] = conflict_level_coef[1]
				# If all beliefs are known, calculate the conflict level
				else:
					conflict_level_temp[0][self.issues_number + issues][0] = abs(links.agent1.belieftree[0][self.issues_number + issues][0] - links.agent1.belieftree[1 + links.agent2.unique_id][self.issues_number + issues][0])
					if conflict_level_temp[0][self.issues_number + issues][0] <= 0.25:
						links.conflict_level[0][self.issues_number + issues][0] = conflict_level_coef[0]
					if conflict_level_temp[0][self.issues_number + issues][0] > 0.25 and conflict_level_temp[0][self.issues_number + issues][0] <= 1.75:
						links.conflict_level[0][self.issues_number + issues][0] = conflict_level_coef[2]
					if conflict_level_temp[0][self.issues_number + issues][0] > 1.75:
						links.conflict_level[0][self.issues_number + issues][0] = conflict_level_coef[1]
				
				# AGENT 2
				# For the calculation of the state conflict level:
				if links.agent2.belieftree[1 + links.agent1.unique_id][self.issues_number + issues][0] == 'No' or links.agent2.belieftree[0][self.issues_number + issues][0] == 'No':
					links.conflict_level[1][self.issues_number + issues][0] = 'No'
				elif links.agent2.belieftree[1 + links.agent1.unique_id][self.issues_number + issues][0] == None:
					links.conflict_level[1][self.issues_number + issues][0] = conflict_level_coef[1]
				else:
					conflict_level_temp[1][self.issues_number + issues][0] = abs(links.agent2.belieftree[0][self.issues_number + issues][0] - links.agent2.belieftree[1 + links.agent1.unique_id][self.issues_number + issues][0])
					if conflict_level_temp[1][self.issues_number + issues][0] <= 0.25:
						links.conflict_level[1][self.issues_number + issues][0] = conflict_level_coef[0]
					if conflict_level_temp[1][self.issues_number + issues][0] > 0.25 and conflict_level_temp[1][self.issues_number + issues][0] <= 1.75:
						links.conflict_level[1][self.issues_number + issues][0] = conflict_level_coef[2]
					if conflict_level_temp[1][self.issues_number + issues][0] > 1.75:
						links.conflict_level[1][self.issues_number + issues][0] = conflict_level_coef[1]

			# print(links.conflict_level)

	def coalition_creation_as(self, agent_action_list, link_list, Pr_ACF_interest, coalitions_number_as, tick_number, coalitions_list_as, coalitions_list_as_total, coalition_threshold, target):

		"""
		The coalition creation function (agenda setting)
		===========================

		This function is used to create the coalitions in the agenda setting.
		The first step is to choose an agent that will be the leader, then
		the coalition is created around that agent based on the network of 
		the leader agent and the belief of the different agents. The criteria
		are detailed in the formalisation.

		"""

		coalition_agent_list = copy.copy(agent_action_list)
		# Run this loop until less than 10\% of the actors are left coalition-less
		while len(coalition_agent_list) > round(0.1 * len(agent_action_list), 0):
			# Finding the agent with the most aware
			agent_aware_sum_list = []
			for agents in coalition_agent_list:
				# print(' ')
				# print(agents.unique_id)
				agent_aware_sum = 0
				for links in link_list:
					if agents == links.agent1 or agents == links.agent2:
						agent_aware_sum += links.aware
				agent_aware_sum_list.append(agent_aware_sum)
			# Deciding the leader
			max_aware_agent = agent_aware_sum_list.index(max(agent_aware_sum_list))
			leader = coalition_agent_list[max_aware_agent]
			coalition_agent_list.remove(leader)
			# print(leader)
			# Choosing the coalition members
			coalition_members = [leader]
			for links in link_list:
				# Only selecting agents in the network
				if leader == links.agent1 and links.agent2 in coalition_agent_list:
					# Threshold check
					# print(' ')
					# print('Leader beliefs: ' + str(leader.belieftree[0][Pr_ACF_interest][1]))
					# print('Agent considered: ' + str(links.agent2.unique_id))
					# print('Agents perceived beliefs: ' + str(leader.belieftree[1 + links.agent2.unique_id][Pr_ACF_interest][1]))
					if leader.belieftree[1 + links.agent2.unique_id][Pr_ACF_interest][target] < leader.belieftree[0][Pr_ACF_interest][target] + coalition_threshold and \
						leader.belieftree[1 + links.agent2.unique_id][Pr_ACF_interest][target] > leader.belieftree[0][Pr_ACF_interest][target] - coalition_threshold:			
						coalition_members.append(links.agent2)
						coalition_agent_list.remove(links.agent2)

				elif leader == links.agent2 and links.agent1 in coalition_agent_list:
					# print(' ')
					# print('Leader beliefs: ' + str(leader.belieftree[0][Pr_ACF_interest][1]))
					# print('Agent considered: ' + str(links.agent1.unique_id))
					# print('Agents perceived beliefs: ' + str(leader.belieftree[1 + links.agent1.unique_id][Pr_ACF_interest][1]))
					if leader.belieftree[1 + links.agent1.unique_id][Pr_ACF_interest][target] < leader.belieftree[0][Pr_ACF_interest][target] + coalition_threshold and \
						leader.belieftree[1 + links.agent1.unique_id][Pr_ACF_interest][target] > leader.belieftree[0][Pr_ACF_interest][target] - coalition_threshold:
						coalition_members.append(links.agent1)
						coalition_agent_list.remove(links.agent1)
			# print('Number of members: ' + str(len(coalition_members)))

			# Creation of a coalition
			# Coalition resources = [Initial resources, calculation resources]
			coalition_resources = [0, 0]
			members_id = []
			for members_for_id in coalition_members:
				members_id.append(members_for_id.unique_id)
			coalition = Coalition(coalitions_number_as[0], leader, coalition_members, members_id, leader.select_as_issue, tick_number, coalition_resources)
			coalitions_number_as[0] += 1
			coalitions_list_as.append(coalition)
			coalitions_list_as_total.append(coalition)

			# Update of different member parameters
			for agents_member in coalition.members:
				# Partial knowledge exchange - aim and state for the issue of the coalition
				for agents_exchange in coalition.members:
					agents_member.belieftree[1 + agents_exchange.unique_id][coalition.issue][0] = agents_exchange.belieftree[0][coalition.issue][0] + (random.random()/5) - 0.1
					agents_member.belieftree[1 + agents_exchange.unique_id][coalition.issue][0] = \
						ActionFunctions.one_minus_one_check(agents_member.belieftree[1 + agents_exchange.unique_id][coalition.issue][0])
					agents_member.belieftree[1 + agents_exchange.unique_id][coalition.issue][1] = agents_exchange.belieftree[0][coalition.issue][1] + (random.random()/5) - 0.1
					agents_member.belieftree[1 + agents_exchange.unique_id][coalition.issue][1] = \
						ActionFunctions.one_minus_one_check(agents_member.belieftree[1 + agents_exchange.unique_id][coalition.issue][1])
				# Assignint coalition number and belonging value
				if agents_member != leader:
					# Assigning the coalition
					agents_member.coalition_as[0] = coalition
					# Belinging value update
					agents_member.coalition_as[1] = 1 - abs(agents_member.belieftree[0][coalition.issue][0] - agents_member.belieftree[1 + leader.unique_id][coalition.issue][0])
				# If the agent is the leader, his belonging level is 1
				if agents_member == leader:
					agents_member.coalition_as[0] = coalition
					agents_member.coalition_as[1] = 1
				# Assigning the team resources - sum of the belonging levels
				coalition.resources[0] += agents_member.coalition_as[1]
				coalition.resources[1] = coalition.resources[0]

	def coalition_creation_pf(self, agent_action_list, link_list, agenda_as_issue, tick_number, coalitions_number_pf, coalitions_list_pf, coalitions_list_pf_total, coalition_threshold, target):

		"""
		The coalition creation function (policy formulation)
		===========================

		This function is used to create the coalitions in the policy formulation.
		The first step is to choose an agent that will be the leader, then
		the coalition is created around that agent based on the network of 
		the leader agent and the belief of the different agents. The criteria
		are detailed in the formalisation.

		"""

		coalition_agent_list = copy.copy(agent_action_list)
		# Run this loop until less than 10\% of the actors are left coalition-less
		while len(coalition_agent_list) > round(0.1 * len(agent_action_list), 0):
			# Finding the agent with the most aware
			agent_aware_sum_list = []
			for agents in coalition_agent_list:
				# print(' ')
				# print(agents.unique_id)
				agent_aware_sum = 0
				for links in link_list:
					if agents == links.agent1 or agents == links.agent2:
						agent_aware_sum += links.aware
				agent_aware_sum_list.append(agent_aware_sum)
			# Deciding the leader
			max_aware_agent = agent_aware_sum_list.index(max(agent_aware_sum_list))
			leader = coalition_agent_list[max_aware_agent]
			coalition_agent_list.remove(leader)
			# print(leader)
			# Choosing the coalition members
			coalition_members = [leader]
			for links in link_list:
				# Only selecting agents in the network
				if leader == links.agent1 and links.agent2 in coalition_agent_list:
					# Threshold check
					check_none = 0
					if leader.belieftree[1 + links.agent2.unique_id][agenda_as_issue][target] == None:
						leader.belieftree[1 + links.agent2.unique_id][agenda_as_issue][target] = 0
						check_none = 1
					# print(' ')
					# print('Leader beliefs: ' + str(leader.belieftree[0][self.agenda_as_issue][1]))
					# print('Agent considered: ' + str(links.agent2.unique_id))
					# print('Agents perceived beliefs: ' + str(leader.belieftree[1 + links.agent2.unique_id][self.agenda_as_issue][1]))
					if leader.belieftree[1 + links.agent2.unique_id][agenda_as_issue][target] < leader.belieftree[0][agenda_as_issue][target] + coalition_threshold and \
						leader.belieftree[1 + links.agent2.unique_id][agenda_as_issue][target] > leader.belieftree[0][agenda_as_issue][target] - coalition_threshold:			
						coalition_members.append(links.agent2)
						coalition_agent_list.remove(links.agent2)
					if check_none == 1:
						leader.belieftree[1 + links.agent2.unique_id][agenda_as_issue][target] = None

				elif leader == links.agent2 and links.agent1 in coalition_agent_list:
					check_none = 0
					if leader.belieftree[1 + links.agent1.unique_id][agenda_as_issue][target] == None:
						leader.belieftree[1 + links.agent1.unique_id][agenda_as_issue][target] = 0
						check_none = 1
					# print(' ')
					# print('Leader beliefs: ' + str(leader.belieftree[0][agenda_as_issue][1]))
					# print('Agent considered: ' + str(links.agent1.unique_id))
					# print('Agents perceived beliefs: ' + str(leader.belieftree[1 + links.agent1.unique_id][agenda_as_issue][1]))
					if leader.belieftree[1 + links.agent1.unique_id][agenda_as_issue][target] < leader.belieftree[0][agenda_as_issue][target] + coalition_threshold and \
						leader.belieftree[1 + links.agent1.unique_id][agenda_as_issue][target] > leader.belieftree[0][agenda_as_issue][target] - coalition_threshold:
						coalition_members.append(links.agent1)
						coalition_agent_list.remove(links.agent1)
					if check_none == 1:
						leader.belieftree[1 + links.agent1.unique_id][agenda_as_issue][target] = None
			# print('Number of members: ' + str(len(coalition_members)))

			# Creation of a coalition
			# Coalition resources = [Initial resources, calculation resources]
			coalition_resources = [0, 0]
			members_id = []
			for members_for_id in coalition_members:
				members_id.append(members_for_id.unique_id)
			coalition = Coalition(coalitions_number_pf[0], leader, coalition_members, members_id, leader.select_pinstrument, tick_number, coalition_resources)
			coalitions_number_pf[0] += 1
			coalitions_list_pf.append(coalition)
			coalitions_list_pf_total.append(coalition)

			# Selecting the issues of interest
			issue_of_interest = []
			for issue_choice in range(self.len_S):
				if self.instruments[coalition.issue][issue_choice] != 0:
					issue_of_interest.append(self.len_Pr + self.len_PC + issue_choice)

			# Update of different member parameters
			for agents_member in coalition.members:
				# Partial knowledge exchange - aim and state for the secondary issues related to the instrument of the coalition
				for agents_exchange in coalition.members:
					for issue_num in issue_of_interest:
						agents_member.belieftree[1 + agents_exchange.unique_id][issue_num][0] = agents_exchange.belieftree[0][issue_num][0] + (random.random()/5) - 0.1
						agents_member.belieftree[1 + agents_exchange.unique_id][issue_num][0] = \
							ActionFunctions.one_minus_one_check(agents_member.belieftree[1 + agents_exchange.unique_id][issue_num][0])
						agents_member.belieftree[1 + agents_exchange.unique_id][issue_num][1] = agents_exchange.belieftree[0][issue_num][1] + (random.random()/5) - 0.1
						agents_member.belieftree[1 + agents_exchange.unique_id][issue_num][1] = \
							ActionFunctions.one_minus_one_check(agents_member.belieftree[1 + agents_exchange.unique_id][issue_num][1])
						# print(agents_member.belieftree[1 + agents_exchange.unique_id][issue_num])

					# Update the agent on the other's instrument preferences
					self.instrument_preference_update(agents_member, 1 + agents_exchange.unique_id, 3)

				# Assignint coalition number and belonging value
				if agents_member != leader:
					# Assigning the coalition
					agents_member.coalition_pf[0] = coalition
					# Belinging value update
					agents_member.coalition_pf[1] = 1 - abs(agents_member.instrument_preferences[0][coalition.issue] - agents_member.instrument_preferences[1 + leader.unique_id][coalition.issue])
				# If the agent is the leader, his belonging level is 1
				if agents_member == leader:
					agents_member.coalition_pf[0] = coalition
					agents_member.coalition_pf[1] = 1
				# Assigning the team resources - sum of the belonging levels
				coalition.resources[0] += agents_member.coalition_pf[1]
				coalition.resources[1] = coalition.resources[0]

	def __str__(self):
		return str(self.grid)