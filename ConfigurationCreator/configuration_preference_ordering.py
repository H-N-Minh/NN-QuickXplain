import random
import xml.etree.ElementTree as et

# Shuffle the order of constraints in the config file
# configuration_file_path: file with the original order
# variable_order_file_path: the new order will be saved here
# irrelevant_features: all the constraints that should be ignored in this config
# @return: the new order of constraints
def configuration_preference_ordering(configuration_file_path, variable_order_file_path, irrelevant_features):
    property_ordering = []
    tree = et.parse(configuration_file_path + "/conf_0.xml")
    root = tree.getroot()
    irrelevant = False

    # Get the order of constraints, ignoring the irrelevant ones, save them in property_ordering
    for items in root:
        for item in items:
            with open(irrelevant_features, "r") as irr_Features:
                for lines in irr_Features:
                    if lines.strip() == item.attrib['key']:
                        irrelevant = True
            if not irrelevant:
                property_ordering.append(item.attrib['key'])
            irrelevant = False

    # Shuffle the order of constraints, and write it to variable_order_file_path
    random.shuffle(property_ordering)

    with open(variable_order_file_path, 'w') as f:
        f.writelines('\n'.join(property_ordering))
    return property_ordering

