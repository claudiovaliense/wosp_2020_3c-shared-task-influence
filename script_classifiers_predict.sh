predict(){
	for index in $(seq 0 $4); do			
		if [ $index -eq 40 ]; then			
			python3.7 classifiers_predict.py $1 dataset/representations/$1/train$index dataset/representations/$1/test$index $2 $3 > y_pred/$1_nohup_$2_$3_predict_test$index.txt
		else						
			python3.6 classifiers_predict.py $1 dataset/representations/$1/train$index dataset/representations/$1/test$index $2 $3 > y_pred/$1_nohup_$2_$3_predict_test$index.txt &
		fi
	done
}

predict 3c-shared-task-influence passive_aggressive f1_macro 0
#predict 3c-shared-task-influence sgd f1_macro 0
#predict 3c-shared-task-influence svm f1_macro 0