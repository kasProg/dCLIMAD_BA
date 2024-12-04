#!/bin/bash

python ibicus_eval.py access_cm2 0 &
python ibicus_eval.py ipsl_cm6a_lr 1 &
python ibicus_eval.py cmcc_esm2 2 &
python ibicus_eval.py miroc6 3 &
python ibicus_eval.py mpi_esm1_2_lr 4 &
python ibicus_eval.py mri_esm2_0 5 &
python ibicus_eval.py noresm2_mm 6 &

wait
