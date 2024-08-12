import xml.etree.ElementTree as et


def configuration_xml_write(configuration_values, progress_xml, output_xml):
    tree = et.parse(progress_xml)
    root = tree.getroot()

    for items in root:
        for item in items:
            if item.attrib['key'] in configuration_values:
                item.attrib['value'] = configuration_values[item.attrib['key']]
                item.attrib['valid'] = "1"
                continue

    tree.write(output_xml)

    return
