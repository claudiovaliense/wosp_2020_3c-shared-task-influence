escores () {	
	for index in $(seq 0 $3); do
		nohup python3.6 classifiers_score.py $1 dataset/representations/$1/train$index $2 > escores/$1_nohup_$2_grid_train$index.txt &		
	done
}

escores 3c-shared-task-influence passive_aggressive 0
#escores 3c-shared-task-influence sgd 0
#escores 3c-shared-task-influence svm 0
