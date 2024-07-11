#!/bin/bash

# Fixed paths for the configuration and temporary files
config_file="/home/ubuntu/zkevm-prover-build/config/config_prover.json"
temp_file="/home/ubuntu/zkevm-prover-build/config/temp.json"
# config_file="../temp/config_prover.json"
# temp_file="../temp/temp.json"

config_file_reading="../config/config_prover.json"

# Output script file
output_script="set_config.sh"

# Start the output script with shebang and default environment variable settings
echo "#!/bin/bash" > $output_script
echo "" >> $output_script

# Read each key from the JSON configuration file
jq 'keys[]' -r $config_file_reading | while read key; do
    # Extract the default value for the current key from the JSON file
    default_value=$(jq -r ".$key" $config_file_reading)

    # Determine the type of the default value to correctly handle boolean and null in Bash
    if [[ "$default_value" == "true" || "$default_value" == "false" ]]; then
        echo "${key^^}=\${${key^^}:-$default_value}" >> $output_script
        echo "jq --argjson key_value \${${key^^}} '.[\"$key\"] = \$key_value' '$config_file' > '$temp_file' && mv '$temp_file' '$config_file'" >> $output_script
    elif [[ "$default_value" == "null" ]]; then
        echo "${key^^}=\${${key^^}:-}" >> $output_script
        echo "jq --argjson key_value null '.[\"$key\"] = \$key_value' '$config_file' > '$temp_file' && mv '$temp_file' '$config_file'" >> $output_script
    elif [[ "$default_value" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "${key^^}=\${${key^^}:-$default_value}" >> $output_script
        echo "jq --argjson key_value \${${key^^}} '.[\"$key\"] = \$key_value' '$config_file' > '$temp_file' && mv '$temp_file' '$config_file'" >> $output_script
    else
        echo "${key^^}=\${${key^^}:-'$default_value'}" >> $output_script
        echo "jq --arg key_value \"\${${key^^}}\" '.[\"$key\"] = \$key_value' '$config_file' > '$temp_file' && mv '$temp_file' '$config_file'" >> $output_script
    fi
done

echo "Configuration script is ready: $output_script"
