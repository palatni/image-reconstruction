#!/bin/bash
OPTSTRING=":c:"

while getopts ${OPTSTRING} opt; do
  case ${opt} in
    c)
      echo ${OPTARG}
      python -m run_training -c ${OPTARG} &> /dev/null &
      streamlit run demonstration.py ${OPTARG}
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      echo "default path ./config.yaml has been chosen"
      exit 1
      ;;
  esac
done

trap 'kill 0' EXIT