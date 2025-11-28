echo 'Pretrain on Train2066 dataset'
python demo.py

echo 'Finetune T'
python Fit_T.py

echo 'Test on Test60 dataset'
python Val.py --data config.VAL1 --Dset_name Test60

echo 'Test on Test315 dataset'
python Val.py --data config.VAL2 --Dset_name Test315

echo 'Test on Train362 dataset'
python Val.py --data config.VAL3 --Dset_name Train362