all:
	make run

run:
	python3 knn.py
	python3 naivebayes.py

clean:
	find . -name '*.pyc' -exec rm --force {} +