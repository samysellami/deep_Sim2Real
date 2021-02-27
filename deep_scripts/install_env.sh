#!/usr/bin/env bash
pyrep_src="../PyRep"
env_name="deep_env"

if [ ! -d $env_name ]
then
  echo "Have not found existing $env_name, creating a new one"
  # pip install virtualenv
  virtualenv --python=python3.6 $env_name
  . ./$env_name/bin/activate && echo "successfully activated $env_name python is $(which python)"
  cd $pyrep_src
  pip install -r requirements.txt
  pip install .
  cd -
  pip install -e .
fi
