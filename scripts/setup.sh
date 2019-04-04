#!/bin/bash

scripts/install_dependencies.sh
scripts/get_data.sh
scripts/preprocess.sh

cd SentEval/data/
./get_transfer_data.bash
