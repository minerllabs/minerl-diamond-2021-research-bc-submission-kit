# NeurIPS 2021: MineRL Competition. Research track baseline submission kit.

[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/BT9uegr)

This repository contains the baseline solution for the research track of NeurIPS MineRL 2021 competition,
placed inside the submission kit. This means it is ready for a submit with few changes!

**Other Resources**
- [Repository for the baseline solution](https://github.com/KarolisRam/MineRL2021-Research-baselines) - Original baseline solution in a cleaner format. **Go here if you want to study the code without overhead from the competition kit!**
- [Submission template](https://github.com/minerllabs/competition_submission_template/) - The original submission template with a random agent. **Go here if you want full details on the submission process**.

# How to submit.

1. Clone this repository.
2. Update `aicrowd.json` file with your list of authors.
3. Follow the instructions [here](https://github.com/minerllabs/competition_submission_template/#how-to-submit) to submit to AICrowd.

# Contents

This kit contains the ["Research track Behavioural cloning"](https://github.com/KarolisRam/MineRL2021-Research-baselines/blob/main/standalone/Behavioural_cloning.py) baseline
solution, modified to fit into the submission baseline.

Here is a list things that were modified over the [submission template](https://github.com/minerllabs/competition_submission_template/) to get things working.

1) Updated `aicrowd.json` to specify research track with `"tags": "research"`. Also set `"gpu": true` so that GPU is used for running the model.
2) Updated `environment.yml` with the required libraries and the correct Python and PyTorch versions (note: it is important that you make sure these versions match your local setup, otherwise the agent may not work!).
3) Added the behavioural cloning model `research_potato.pth` and the KMeans centroids `centroids_for_research_potato.npy`  to `./train` directory.
4) Updated `test_submission_code.py` by placing functions from the [baseline code](https://github.com/KarolisRam/MineRL2021-Research-baselines/blob/main/standalone/Behavioural_cloning.py) into the file, and updating the main entry point inside `run_agent_on_episode` (at the end of the code file).
5) Updated `train_submission_code.py` by placing parameters and functions from the [baseline code](https://github.com/KarolisRam/MineRL2021-Research-baselines/blob/main/standalone/Behavioural_cloning.py) into the file, and updating `main()` function.
