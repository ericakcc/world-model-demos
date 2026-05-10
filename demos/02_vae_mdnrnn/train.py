"""
Demo 02: VAE + MDN-RNN + Controller (Ha & Schmidhuber 2018).

Not implemented yet — see README.md for the implementation checklist.

The classic World Models paper trains in three separate stages:
    1. VAE on raw frames           → train_vae.py
    2. MDN-RNN on latent sequences → train_mdnrnn.py
    3. Linear controller via CMA-ES → train_controller.py

This file is intentionally a stub so the directory has a runnable entry point.
"""
import sys


def main():
    print(__doc__)
    print("Not implemented. Read demos/02_vae_mdnrnn/README.md for the plan.")
    sys.exit(2)


if __name__ == "__main__":
    main()
