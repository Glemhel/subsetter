![](media/logo.gif)

Always wanted to create a subset of metrics for repositores analysis? **Say no more!**

This project contains scripts for repositories analysis, and determining minimal metrics subset.
We also store metrics collected for Rust language repostories, currently over 75 metrics from over 170 repositores, collected using [ifcount tool](https://github.com/DCNick3/ifcount). Repositories were selected to satisfy certain quality and popularity requirements, filtered using [very-large-repos](https://github.com/DCNick3/very-large-repos).

The main analysis happes in [subsetter jupyter notebook](Subsetter.ipynb), which can be run using google colab. Instruction on local reproduction will follow in next project stages.

Correlations plot for dataset features:
![](correlations.png)
