## Usage

### Build a clustering
To build a hierarchical clustering, use the following

```sh
./clustering_driver.py build \-i \[input\] \-p \[params\] \-k \[cluster_count\] \-C \[cutoff\] \-o \[output\]
```
Input is in the form of a `.npy` or a `.fbs` file. params is in the form of a `.fbs` file generate by the param generator in [ingest-pipeline](https://github.com/DoYouEvenStackSmash/ingest-pipeline) Output filename is a prefix, e.g. `output`. This will produce 3 files: `output_tree_hierarchy.json`, `output_tree_data_list.npy`, and `output_tree_node_vals.npy`.

### Load a clustering for visualization
To construct a graphml of an existing hierarchical clustering, use the following

```sh
./clustering_driver.py load \-t tree_hierarchy.json \-G
```

This will produce a `tree_representation.graphml` file which can be loaded by many graph viewers.

### Evaluate likelihoods

To evaluate likelihoods of an input, use the following

```sh
./clustering likelihood \-\-model input_tree_hierarchy.json \-\-images input_images.npy \-\-ctfs input_ctfs.npy \-test
```

This will calculate likelihoods via the algorithms chosen in `likelihood_scratch.py` and output the results to a CSV file which can be further processed by `analytics.py` or the MATLAB in [postprocessing](https://github.com/DoYouEvenStackSmash/data-analysis/tree/cluster-radius-patch/src/postprocessing).  

Current likelihood algorithms:

|  Search Type         | Defining Characteristic     |   Performance | 
|------------------|------------------|------------------|
| Greedy | Best First Search | Very Fast but inaccurate | 
| Patient | KD tree style query | Fast but somewhat inaccurate |
| Impatient | An attempt at a different bound | Seems to be sorta fast but not accurate |
| Level Patient | Lexicographical BFS | Slow but generally accurate |
| Naive | All pairs comparison | Very slow but accurate |
