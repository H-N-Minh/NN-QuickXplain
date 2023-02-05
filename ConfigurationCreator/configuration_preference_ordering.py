import random
import xml.etree.ElementTree as et


def configuration_preference_ordering(configuration_file_path, variable_order_file_path, irrelevant_features):
    property_ordering = []
    tree = et.parse(configuration_file_path + "/conf_0.xml")
    root = tree.getroot()
    irrelevant = False

    for items in root:
        for item in items:
            with open(irrelevant_features, "r") as irr_Features:
                for lines in irr_Features:
                    if lines.strip() == item.attrib['key']:
                        irrelevant = True
            if not irrelevant:
                property_ordering.append(item.attrib['key'])
            irrelevant = False

    random.shuffle(property_ordering)

    with open(variable_order_file_path, 'w') as f:
        f.writelines('\n'.join(property_ordering))
    return property_ordering

