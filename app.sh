#!/usr/bin/env bash
 
# This is a bash script for Domino's App publishing feature
# Learn more at http://support.dominodatalab.com/hc/en-us/articles/209150326
 
## R/Shiny Example
## This is an example of the code you would need in this bash script for a R/Shiny app
R -e 'shiny::runApp("./apps/wine.R", port=8888, host="0.0.0.0")'
 
## Flask example
## This is an example of the code you would need in this bash script for a Python/Flask app
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8
#export FLASK_APP=app-flask.py
#export FLASK_DEBUG=1
#python -m flask run --host=0.0.0.0 --port=8888
 
## Dash Example
## This is an example of the code you would need in this bash script for a Dash app
#python app-dash.py