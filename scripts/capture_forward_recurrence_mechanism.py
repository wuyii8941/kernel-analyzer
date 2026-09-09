"""Reuse the model/input/snapshot runner with the forward recurrence adapter."""
from scripts.capture_softmax_mechanism import main


if __name__ == '__main__':
    main(family='forward_recurrence')
