import re

def parse_configurations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    configs = []
    current_config = {}

    for line in lines:
        # Match configuration number
        config_match = re.match(r"Configuration (\d+)/\d+", line)
        if config_match:
            if current_config:  # Save the previous config
                configs.append(current_config)
            current_config = {"config_number": config_match.group(1)}

        # Match Exact Match and F1 scores
        performance_match = re.search(r"Exact Match = ([\d.]+)%, F1 = ([\d.]+)", line)
        if performance_match:
            current_config["exact_match"] = float(performance_match.group(1))
            current_config["f1"] = float(performance_match.group(2))

        # Match configuration name
        if ":" in line:
            current_config["name"] = line.strip()

    # Append the last configuration
    if current_config:
        configs.append(current_config)

    return configs

def find_best_configs(configs):
    # Find the configuration with the highest Exact Match
    best_exact_match = max(configs, key=lambda x: x["exact_match"])
    
    # Find the configuration with the highest F1 score
    best_f1 = max(configs, key=lambda x: x["f1"])
    
    # Find the configuration with the highest combined score (Exact Match + F1)
    best_combined = max(configs, key=lambda x: x["exact_match"] + x["f1"])
    
    return best_exact_match, best_f1, best_combined

def main():
    file_path = "temp.txt"  # Replace with your file path
    configs = parse_configurations(file_path)
    best_exact_match, best_f1, best_combined = find_best_configs(configs)

    print("Configuration with Highest Exact Match:")
    print(f"Name: {best_exact_match['name']}")
    print(f"Exact Match: {best_exact_match['exact_match']}%, F1: {best_exact_match['f1']}\n")

    print("Configuration with Highest F1:")
    print(f"Name: {best_f1['name']}")
    print(f"Exact Match: {best_f1['exact_match']}%, F1: {best_f1['f1']}\n")

    print("Configuration with Highest Combined Score (Exact Match + F1):")
    print(f"Name: {best_combined['name']}")
    print(f"Exact Match: {best_combined['exact_match']}%, F1: {best_combined['f1']}")

if __name__ == "__main__":
    main()