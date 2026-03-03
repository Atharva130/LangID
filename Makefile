install:
	pip install -r requirements.txt

train:
	python -m src.train

evaluate:
	python -m src.evaluate

predict:
	python -m src.predict

run:
	uvicorn src.api:app --reload

test:
	python -m tests.test_edge_cases