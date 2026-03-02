#!/bin/bash

VERBOSE=0

# Verbose logging function
# Will only print to stdout if the verbose flag is set
function log () {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "$@"
    fi
}

# Error function
# Prints the error message to stderr and exits
function error () {
    echo "$@" >&2
    exit 1
}
