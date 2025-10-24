echo 'Pretrain on Train7596 dataset'
python demo.py

echo 'Test on Test60 dataset'
python Val.py --data r'/../gz-data/Test/Test_60' --Dset_name Test60

echo 'Test on Test346 dataset'
python Val.py --data r'/../gz-data/Test/Test_315' --Dset_name Test315

echo 'Test one UBtest31 dataset'
python Val.py --data r'/../gz-data/Test/UBtest_31.pkl' --Dset_name UBtest31