# ERMiner
Repo - Paper "Discovering Editing Rules by Deep Reinforcement Learning"

## File Structure

* `doc/er.pdf`: Implementation Details of EnuMiner.

* `src/`: Source code of algorithms
	- `src/enuminer.py`: the Enuminer algorithm.
	- `src/train.py`: training algorithm of RLMiner.
	- `src/discovery.py`: discovery algorithm of RLMiner.
	- `src/env.py`:  environment of RLMiner.
	- `src/net.py`: network of RL algorith - Rainbow.
	- `src/tree.py`: structure of rule tree in RLMiner.
	- `src/utils.py`: utility class functions including couting, etc.
	- `src/eval.py`: evaluation for EnuMiner and RLMiner.
	- `src/data.py`: generator of dirty data.
	
* `data/`: Dataset source files. 
	- `data/Covid`: dataset Covid-19	
	  - `data/Covid/master_data.csv`: master data of Covid
	  - `data/Covid/input_data.csv`: input data of Covid
	  - `data/Covid/input_data_clean.csv`: ground truths w.r.t input data of Covid
	
	- `data/Adult`: dataset Adult
	
	  - `data/Adult/master_data.csv`: master data of Adult
	
	  - `data/Adult/input_data.csv`: input data of Adult
	
	  - `data/Adult/input_data_clean.csv`: ground truths w.r.t input data of Adult
	
	- `data/Nursery`: dataset Nursery	
	  - `data/Nursery/master_data.csv`: master data of Nursery
	  - `data/Nursery/input_data.csv`: input data of Nursery
	  - `data/Nursery/input_data_clean.csv`: ground truths w.r.t input data of Nursery

## Requirements

- torch>=1.1.0
- pandas>=0.25.3
- numpy>=1.19.5
- tianshou >= 0.4.5

## Usage

- Test on Covid-19
```
bash covid.sh
```
- Test on Adult
```
bash adult.sh
```
- Test on Nursery
```
bash nursery.sh
```

