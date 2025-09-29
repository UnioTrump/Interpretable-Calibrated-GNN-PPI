echo 'Pretrain on Train7596 dataset'
python demo.py

echo 'Test on Test60 dataset'
python Val.py --data config.VAL_DATA_PATH --Dset_name Test60

echo 'Test on Test346 dataset'
python Val.py --data config.TUNING_DATA_PATH --Dset_name Test346