![](media/logo.gif)

Always wanted to create a subset of metrics for repositores analysis? **Say no more!**

This project contains scripts for repositories analysis, and determining minimal metrics subset.
We also store metrics collected for Rust language repostories, currently over 75 metrics from over 170 repositores, collected using [ifcount tool](https://github.com/DCNick3/ifcount). Repositories were selected to satisfy certain quality and popularity requirements, filtered using [very-large-repos](https://github.com/DCNick3/very-large-repos).

The main analysis involves running optimisation algorithms for selecting optimal subset of metrics. You can try to run the following:

```cd src && python main.py --config=sa_sammon_example.yaml```

where .yaml is config file, setting algorithm and its hyperparameters, metrics used and anlaysis size (in terms of number of repositories).
