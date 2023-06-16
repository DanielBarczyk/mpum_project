# mpum_project

## Running
Just run `python model_comparison.py` to start evaluating the data.

## Change run parameters
Modify `distributions` array to test models for specific difstributions. The program will run for all distributions in the array. Allowed values are `"full", "partial", "even"`.

Modify `models` array to change which models will be tested.

Change parameters of `Data` class instance to modify how the data is loaded:
1. `filename` will change what file the data is loaded from. Set appropriate `feature_idx` and `label_idx` corresponding to proper columns if you want to change this.
2. `vectorization` will change feature extraction method. Allowed values are `"count", "tfidf"`.
3. `use_single` when used with a MBTI label will cast labels to a binary. E.g. `Data(use_single=0)` will change all "Exxx" labels to 1 and "Ixxx" labels to 0, `Data(use_single=1)` will change all "xNxx" labels to 1 and "xSxx" labels to 0. Allowed values are `None, 0, 1, 2, 3`.
