#!/bin/bash
OPTSTRING=":c:"

while getopts ${OPTSTRING} opt; do
  case ${opt} in
    c)
      python -m run_training -c ${OPTARG} &> /dev/null &
      streamlit run demonstration.py ${OPTARG}
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
    python -m run_training &> /dev/null &
    streamlit run demonstration.py
fi

trap 'kill 0' EXIT