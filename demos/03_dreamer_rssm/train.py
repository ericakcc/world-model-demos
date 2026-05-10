"""
Demo 03: Dreamer / RSSM.

Not implemented yet — see README.md for the implementation checklist.

Build order suggestion:
    1. RSSM cell (prior / posterior / recurrent)    → rssm.py
    2. World model (RSSM + obs/reward/continue)     → world_model.py
    3. Sequence replay buffer                        → replay.py
    4. Actor-critic on imagined latent rollouts      → actor_critic.py
    5. Main training loop                            → train.py
"""
import sys


def main():
    print(__doc__)
    print("Not implemented. Read demos/03_dreamer_rssm/README.md for the plan.")
    sys.exit(2)


if __name__ == "__main__":
    main()
