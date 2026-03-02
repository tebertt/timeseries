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

    # Check if Python3 is installed
    if command -v python3 >/dev/null 2>&1; then
        log "Python 3 is installed"

        # Check if the python virtual environment module is installed
        if python3 -c "import ensurepip" > /dev/null 2>&1; then
            log "python3-venv is installed"
        else
            log "python3-venv is not installed, attempting to install..."
            echo "Need permission to install python3 virtual environment"
            {
                sudo apt-get install python3-venv
            } || {
                error "Could not install python3-venv"
            }
        fi

        # Delete the virtual environment if it already exists
        if [ -d $VENV_PATH ]; then
            {
                log "Removing existing virtual environment"
                rm -rf $VENV_PATH
            } || {
                error "Could not remove virtual environment"
            }
        fi

        # Create the virtual environment
        log "Creating virtual environment $VENV_NAME"
        {
            python3 -m venv $VENV_PATH
        } || {
            error "Could not create virtual environment"
        }

        # Add environment variable to not generate .pyc files
        log "Adding environment variables"
        echo >> $VENV_PATH"bin/activate"
        echo "# Do not generate .pyc files" >> $VENV_PATH"bin/activate"
        echo "export PYTHONDONTWRITEBYTECODE=1" >> $VENV_PATH"bin/activate"

        # Install the required packages
        log "Installing required packages"
        {
            $VENV_PATH"bin/pip" install -r $PROJECT_PATH$REQUIREMENTS_NAME
        } || {
            error "Could not install required packages"
        }

    else
        error "Python 3 is not installed"
    fi

    log "Environment created"

}

main $@
