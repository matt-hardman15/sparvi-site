"""Make sure both fitted sets are in the model store before the build: load each from the store, or train it when its
model code or training data changed (artifacts.py). The refresh runs this as its own step, so a run that had to train
shows it there and the build that follows only scores.

    python baseball/pitch_quality/ensure_models.py
"""
from train import fitted

fitted("production")
fitted("oos", verify=True)
