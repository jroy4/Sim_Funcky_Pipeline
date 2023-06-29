 #!/bin/bash
## Set up FSL environment 

FSLDIR=/usr/share/fsl/5.0
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH 

case "$1" in
  workflow)
  eval python3 /app/Pipeline.py
  ;; 
  debug) 
    bash
  ;; 
      *)
      echo "
        Starting Jupyter Lab 

        Exit with CTRL+D
        "
    
    #exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    exec jupyter lab --ip=0.0.0.0
  ;;
esac




