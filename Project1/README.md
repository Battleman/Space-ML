# Higgs Boson Challenge

The Kaggle competition on predicting if a particle is a Higgs boson based on the decay signature.

## Content
Structure of the repository:
```
.
├── data/
├── misc/
├── README.md
├── report/
└── src/
    ├── cache/
    ├── cross_validation.py
    ├── exploration.ipynb
    ├── features_engineering.py
    ├── implementations.py
    ├── parameters.yaml
    ├── preprocessing.py
    ├── proj1_helpers.py
    ├── project1.ipynb
    └── run.py
```
Interesting directories:
* `data`: Where to store training and testing csv
* `src`: All of the source code
  * `cross_validation.py`: Methods related to cross-validation/kfold
  * `features_engineeriny.py` Methods related to features engineering/augmentation
  * `parameters.yaml` (Hyper-)parameters of the model, such as lambdas, powers,... but also if the model should cache intermediate results,..
  * `implementations.py` The building blocks of ML: least_squares, ridge_regression,...
  * `run.py`: Run this to obtain predictions. Should work blackbox.


## HOW-TO: obtain predictions:
* Retrieve training and testing data (in csv format) and place them as `train.csv` and `test.csv` in `data/`. If you wish to name differently or place them elsewhere, adapt consequently in `parameters.yaml`.
* Move to `src`, and run `run.py`. E.g. `python3 run.py`

### Limitations
The powers labelled COMBINED_DEGREES designates the maximum degree polynomial combination. This number will increase _exponentially_ the amount of engineered variables, and thus of required memory. This will create $\frac{( r+n-1)!}{r!(n-1)!}$ variables (with $n$ the number of features, $r$ the selected power). A usually acceptable number for $r$ is usually 2 or 3. Setting it to 4, already requires about 40GB of memory, and 5 around 150GB. SWAP is a good alternative, but being quite slower than RAM, this will take considerable time.

## Authors:
* Alexandre Reynaud
* Othmane Squalli
* Olivier Cloux
