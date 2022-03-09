#!/bin/bash
python make_hdf5.py --dataset I128 --batch_size 64 --data_root "/zfs/laogroup/Bingyin_BigGan"
python calculate_inception_moments.py --dataset I128_hdf5 --data_root "/zfs/laogroup/Bingyin_BigGan"
