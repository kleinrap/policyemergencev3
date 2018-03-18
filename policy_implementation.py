'''''
'''''

def policy_package_implementation(policy_selected, AT_value, OT_value, DT_value, FPT_value, ERC_value, RT_value, AdT_value, PH_value, RS_value, CT_value):

	'''''
	This function is used to implement the policy package chosen by the policy makers through the changing of the exogenous parameters from the technical model.
	'''''

	AT_value = AT_value * (1 + policy_selected[0])
	OT_value = OT_value * (1 + policy_selected[1])
	DT_value = DT_value * (1 + policy_selected[2])
	FPT_value = FPT_value * (1 + policy_selected[3])
	ERC_value = ERC_value * (1 + policy_selected[4])
	# CHANGE THIS! - This will need to be changed once the real belief tree has been added
	RT_value = RT_value * (1 + policy_selected[5])
	AdT_value = AdT_value * (1 + policy_selected[6])
	PH_value = PH_value * (1 + policy_selected[7])
	RS_value = RS_value * (1 + policy_selected[8])
	CT_value = CT_value * (1 + policy_selected[9])

	return AT_value, OT_value, DT_value, FPT_value, ERC_value, RT_value, AdT_value, PH_value, RS_value, CT_value

