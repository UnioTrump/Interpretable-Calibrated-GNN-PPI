# Running This shell file

echo 'Pretrain on Train1958 dataset'
python demo.py

echo 'Test on Test60 dataset'
python Val.py --data_path ../gz-data/Test_60

echo 'Test on Test315 dataset'
python Val.py --data_path ../gz-data/Test_315

# echo 'Test on DSet_72 dataset'
# python Val.py --data_path ../gz-data/Test_72

# echo 'Test on Test164 dataset'
# python Val.py --data_path ../gz-data/Test_164

# echo 'Test on Test186 dataset'
# python Val.py --data_path ../gz-data/Test_186