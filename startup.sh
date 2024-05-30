#!/bin/bash

# Set FSL environment variables
FSLDIR=/usr/local/fsl
PATH=${FSLDIR}/bin:${PATH}
FSLOUTPUTTYPE=NIFTI_GZ

# Source the FSL configuration script
source ${FSLDIR}/etc/fslconf/fsl.sh

case "$1" in 
    terminal)
        bash  
    ;;
    
    *)
        eval python3 /app/Pipeline.py "$@"
        cmd_exit="$?"
        if [ "$cmd_exit" -ne 0 ]; then
            exit "$cmd_exit"
        fi
    ;;
esac