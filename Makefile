all:
	make run

run:
	python3 knn.py &
	python3 naivebayes.py &
	python3 dt3.py

clean:
	find . -name '*.pyc' -exec rm --force {} +
