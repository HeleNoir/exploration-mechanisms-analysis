# exploration-mechanisms-analysis
Evaluation and analysis of experiments on mechanisms to enhance exploration in metaheuristics 
(see https://github.com/HeleNoir/exploration-mechanisms for experiment setup).

The following algorithms are analysed and compared:

1) PSO
2) SHADE
3) PSO with Random Restarts
4) PSO-NPGM (New Population Generation Mechanism)
5) PSO-GPGM (Gbest-guided Population Generation Mechanism)
6) PSO-SRM (Solution Replacement Mechanism)
7) PSO-PDM (Population Dispersion Mechanism)

### Run analysis

First, convert the .cbor logs from the experiments into dataframes (if both repositories are in the same folder, there is
no need to change the file paths), using `convert_log_files.py` for each algorithm and dimension setting.

Next, run `analysis.py` to generate data summaries and some plots for a general overview.

For the comparison, run `comparison.py` (has to be applied last as it uses dataframes generated in the analysis).