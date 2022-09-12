#!/bin/bash

pip install pip-tools
pip-compile requirements.in
pip-sync