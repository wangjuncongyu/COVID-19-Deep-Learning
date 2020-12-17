python test_network.py -train_subsets=1;2;3;4 -eval_subsets=subset_5.csv -batch_size=72 -gt_treatment=1
python test_network.py -train_subsets=1;2;3;5 -eval_subsets=subset_4.csv -batch_size=72 -gt_treatment=1
python test_network.py -train_subsets=1;2;4;5 -eval_subsets=subset_3.csv -batch_size=72 -gt_treatment=1 
python test_network.py -train_subsets=1;3;4;5 -eval_subsets=subset_2.csv -batch_size=72 -gt_treatment=1  
python test_network.py -train_subsets=2;3;4;5 -eval_subsets=subset_1.csv -batch_size=72 -gt_treatment=1

python test_network.py -train_subsets=1;2;3;4 -eval_subsets=subset_5.csv -batch_size=72 -gt_treatment=0
python test_network.py -train_subsets=1;2;3;5 -eval_subsets=subset_4.csv -batch_size=72 -gt_treatment=0
python test_network.py -train_subsets=1;2;4;5 -eval_subsets=subset_3.csv -batch_size=72 -gt_treatment=0 
python test_network.py -train_subsets=1;3;4;5 -eval_subsets=subset_2.csv -batch_size=72 -gt_treatment=0  
python test_network.py -train_subsets=2;3;4;5 -eval_subsets=subset_1.csv -batch_size=72 -gt_treatment=0


