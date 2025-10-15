#!/usr/bin/env bash
#1. git pull
#2. Czy istnieją installed_requirements.txt?
#a) - nie ma - do zainstalowania
#b) - są - porównaj z aktualnymi requirements.txt
#b  a) takie same - nic nie rób
#b  b) różne - reinstall_venv() - zrób venv .hydroenv, zainstaluj requirements.txt
#            Zrób kopię requirements.txt - used_requirements.txt (ma być w gitignore)

#git pull origin

# Function to recreate virtual environment and install dependencies
reinstall_venv() {
  rm -rf .venv
  python -m venv .venv
  source venv.sh
  python -c "import sys; print('sys.prefix:', sys.prefix)"
  echo "Ensure, that srcipt really is in the virtual environment. Press newline to continue"; read
  python -m pip install --upgrade pip
  python -m pip --version
  pip install -r requirements.txt
  cp requirements.txt installed_requirements.txt
  deactivate
  echo "reinstalled .venv"
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
  echo "no venv, installing"
  reinstall_venv
else
  # Check if used_requirements.txt exists
  if [ ! -f "installed_requirements.txt" ]; then
    echo "no installed_requirements.txt file, reinstalling"
    reinstall_venv
  else
    # Compare requirements.txt with installed_requirements.txt
    if ! cmp -s requirements.txt installed_requirements.txt; then
      echo "requirements lists not equal, reinstalling"
      reinstall_venv
    else
        echo "requirements lists equal, no need to reinstall"
    fi
  fi
fi
