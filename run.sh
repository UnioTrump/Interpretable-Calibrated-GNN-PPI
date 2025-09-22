echo 'Pretrain on Train7596 dataset'
python demo.py

echo 'Tune the model on Train346 dataset'
python Tuning.py

echo 'Test on Test60 dataset'
python Val.py