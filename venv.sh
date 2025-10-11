#!/usr/bin/env bash
#run with source ./hydrovenv
PLATFORM="$(uname | tr '[:upper:]' '[:lower:]')"
if [[ $PLATFORM =~ nt|win ]]; then
  echo "WINDOWS"
  ACTIVATE_SCRIPT=.hydrovenv/Scripts/activate
else
  echo "UNIXLIKE"
  ACTIVATE_SCRIPT=.hydrovenv/bin/activate
fi
source $ACTIVATE_SCRIPT