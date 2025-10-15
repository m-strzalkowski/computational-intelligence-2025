#!/usr/bin/env bash
#run with source ./venv
PLATFORM="$(uname | tr '[:upper:]' '[:lower:]')"
if [[ $PLATFORM =~ nt|win ]]; then
  echo "WINDOWS"
  ACTIVATE_SCRIPT=.venv/Scripts/activate
else
  echo "UNIXLIKE"
  ACTIVATE_SCRIPT=.venv/bin/activate
fi
source $ACTIVATE_SCRIPT