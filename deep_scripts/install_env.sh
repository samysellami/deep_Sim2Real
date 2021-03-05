#!/usr/bin/env bash
pyrep_src="../PyRep"
spinningup_src = "../spinningup"
baselines_src = "../baselines"
RL_src =  baselines_src # choose which RL implementations to use
env_name="deep_env" # virtual environment name

if [ ! -d $env_name ]
then
  echo "Have not found existing $env_name, creating a new one"
  # pip install virtualenv
  virtualenv --python=python3.6 $env_name
  . ./$env_name/bin/activate && echo "successfully activated $env_name python is $(which python)"
  cd $pyrep_src
  pip install -r requirements.txt
  pip install .
  # cd -
  # cd $RL_src
  # pip install .
  cd -  
  pip install -e .
fi
