# Cricket Fixtures

### Setup
- I installed in a virtual environment using `pip install -e ./`
- Not many packages are required for this to run pandas, numpy, tqdm etc.

### Entry points
- `src/cricket_fixtures/cricket_fixtures.py`
- Unit tests will show you how the implementation works

### Data input
- See `data fixtures/raw/cricket_fixtures`

### Output
- See the files produced in `data/processed/cricket_fixtures/outputs`
- The files are updated live during a run 

### How to make changes
- Define a new mutation function in `src/cricket_fixtures/mutation.py`
- Define a new fitness function in `src/cricket_fixtures/fitness.py`
- Reconfigure the run in `src/cricket_fixtures/cricket_fixtures.py`