# Problem-Set-2-Econ_5200_PS2

#  The Underpowered Portfolio

A simulation of 200 A/B tests with known true effects. It audits a client's "ship if p < 0.05" rule and shows how the winner's curse inflates reported gains. It then tests a stricter rule and an always-valid sequential test.

## Files
- `Econ_5200_PS2.ipynb`: the full analysis (Phases 1 to 4), top to bottom.
- `ai-appendix.md`: both AI prompts, the model's raw replies and its code (Claude Sonnet 5.5, 27 October 2026).
- `experiment_portfolio.py`: the simulator and the two decision rules.
- `msprt.py`: the always-valid sequential test (mixture SPRT), AI-generated and checked in the notebook.

## How to run
Open the notebook in Colab and choose Runtime › Restart and run all. The seed is fixed (42), so every number reruns identically. The last cell prints every planted parameter and
