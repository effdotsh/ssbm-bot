<a href="https://aispawn.com/support" target="_blank"><img src="https://aispawn.com/support/readme-image.png" alt="Support Me!" height="41" width="174"></a>
# SmashBot

A psuedo imitation learning AI to play super smash bros



**Disclaimer:** I am making the code public, however I am not able to dedicate the time to help anybody troubleshoot anything, or to make the tool more user friendly. Utilize and build upon at your own risk. I'm very sorry but I'm too lazy to make most things work through arguments, so you will need edit variables in the code to change stuff.

## Usage

**Step 0:** Follow the [libmelee](https://github.com/altf4/libmelee) setup instruction. Set the dolphin path in `Args.py`

**Step 1**: [Get a melee ISO](https://dolphin-emu.org/docs/guides/ripping-games/), name it `SSBM.iso` and put it in the main folder.

**Step 2:** Get *alot* of slippi replays. For my project, I used [this](https://drive.google.com/file/d/1ab6ovA46tfiPZ2Y3a_yS1J3k3656yQ8f/edit) dataset, however even this was more limited then I would like. Best case scenario, is alot of replay by a single player against a bunch of different opponents. Your mileage may vary.

**Step 3:** Change the `replay_folder` variable to the path to your dataset, and run `organize_replays.py`

**Step 4:** Run `generate_data.py` . Depending on the size of your dataset, this may take a very long time.

**Step 5:**  Set  `player_character`, `opponent_character`, and `stage` to your desired targets and run `train.py`. You will need to tune the optimizer, learning rate, network structure with different targets. 

**Step 6:** Set the same targets in `duel.py` and run it. You can very the models "attack weighting" by changing the denominator in `Datahandler.py` line 231




















