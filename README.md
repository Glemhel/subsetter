![](media/logo.gif)

Always wanted to create a subset of metrics for repositores analysis? **Say no more!**

This project contains scripts for repositories analysis, and determining minimal metrics subset.
We also store metrics collected for Rust language repostories, currently over 100 metrics from over 250 repositores, collected using [ifcount tool](https://github.com/DCNick3/ifcount). Repositories were selected to satisfy certain quality and popularity requirements, filtered using [very-large-repos](https://github.com/DCNick3/very-large-repos).

## Quick Start

The main analysis involves running optimization algorithms for selecting optimal subset of metrics. You can try to run the following:

```cd src && python main.py --config=sa_sammon_example.yaml```

where .yaml is config file, setting algorithm and its hyperparameters, metrics used and analysis size (in terms of number of repositories).

## Project Structure

Our repository includes collected data, analysis jupyter notebooks, optimization analysis scripts and hypothesis testing scripts. The folder structure is the following:

- `analysis` stores results of optimization experiments. Folders with name {object}-{algorithm}-{loss function} store analysis results with metrics for {object}, conducted with {algorithm} and {loss function}. Each folder has the following files, each prefixed with folder name: the results yaml itself in `results-{object}-{algorithm}-{loss function}.yaml`, optimization results output file `optimization_report.csv`, summary of loss function values for each subset size `stats.dat`, and list of best metrics `best_metrics.dat`.\
Moreover, *analysis* store correlation analysis results in tabular and graphical formats.
- `configs` contains config files for each conducted experiment, with name corresponding to ones from `analysis` folder.
- `data` folder has metrics data and names for the analysis.
- `src` stores Python scripts for the optimization and hypothesis testing. The entry point is `main.py` configured by `.yaml` config files. It also has scripts for metrics preprocessing `metrics_preprocessor.py` and results postprocessing `results_converter.py`.
