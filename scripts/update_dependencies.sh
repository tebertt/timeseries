#!/bin/bash

SCRIPTS_PATH="./"
PROJECT_PATH="../"
REQUIREMENTS_NAME="requirements.txt"
VENV_NAME="venv"
VENV_PATH=$PROJECT_PATH$VENV_NAME"/"

# Import the logging functions
. $SCRIPTS_PATH"logging.sh"

# Usage function
# Prints the help message when the help flag is set
function usage () {
    echo "Usage: $0 [-v] [-h]"
    echo "  -v  Verbose mode"
    echo "  -h  Display this help message"
    exit 0
}

# Main function
function main() {

    # Handle flags
    local OPTIND
    while getopts "vh" opt; do
        case $opt in
            v) # Handle the -v verbose flag
                VERBOSE=1
                log "Running in verbose mode"
            ;;
            h) # Handle the -h help flag
                usage
            ;;
            \?) # Handle invalid options
                error "Invalid option: $OPTARG"
            ;;
        esac
    done
    shift $((OPTIND -1))

    # Check if the virtual pip is installed
    if test $VENV_PATH"bin/pip"; then
        # Update python packages
        log "Updating python packages"
        {
            $VENV_PATH"bin/pip" --disable-pip-version-check list --outdated --format=json |
            $VENV_PATH"bin/python" -c "import json, sys; print('\n'.join([x['name'] for x in json.load(sys.stdin)]))" |
            xargs -r -n1 $VENV_PATH"bin/pip" install -U
        } || {
            error "Failed to update packages"
        }
    else
        error "Virtual environment pip does not exist"
    fi

    # Update requirements.txt
    log "Updating requirements.txt"
    {
        $VENV_PATH"bin/pip" freeze > $PROJECT_PATH$REQUIREMENTS_NAME
    } || {
        error "Failed to update requirements.txt"
    }

    log "Packages updated"

}

main $@
