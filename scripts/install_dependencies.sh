#!/bin/bash

conda install --yes pytorch=0.4.0 cuda91 -c pytorch
conda install --yes matplotlib

pip install -r requirements.txt
