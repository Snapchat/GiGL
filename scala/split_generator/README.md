## Split Generator

______________________________________________________________________

### Steps:

```
1. Read the assigner and split strategy classpaths from the config.
2. Read the SGS output from GCS using the TFRecord Spark Connector (more details on that below)
3. Coalesce the read dataframe. This is to increase the CPU utilization of the Spark job (no shuffle cost incurred)
4. Cache this dataframe as we will require this to compute all the 3 (train/test/val) splits. If we do not cache this Spark tries to read the input from GCS 3 times. We can skip caching for inference only jobs as we only want to compute the test split.
5. Convert the Array of Bytes Dataset to the proto Dataset.
6. Using dataset.map() we map every sample to the output sample of the respective split using the SplitStrategy class(which does not take any Spark dependency). We do this for every split.
7. The output dataset is converted back to a dataset of array of bytes which is written to GCS.
```

### SplitGen Arguments:

- **train/test/val ratio**

  Ratios of the respective split. We use the `scala.util.hashing.MurmurHash3` library to hash the edges/nodes to the
  splits. Murmur hash is independent of the machine it runs on and hence produces the same split irrespective of the
  underlying machine.

- **should_split_edges_symmetrically**

  flag to indicate whether bidirectional edges should be assigned to the same split. That is if set to true a->b and
  b->a will belong to the same split. Typically set false for directed graphs.

- **is_disjoint_mode** flag to indicate if the training is done in disjoint mode or not. In non-disjoint mode, both
  supervision and message passing edges are treated the same for the train split. In disjoint mode, they are treated
  separately.

- **assignerClsPth** class path to the assginer that is used for assigning the nodes/edges to a respective split. New
  assiners must inplement the abstract class `Assigner`

- **splitStrategyClsPth** class path to the Split Strategy being used for the job. New implmentations must inplement the
  abstract class `SplitStrategy`

### Implementation:

- Leverages Dataset.map() and uses custom Scala code to convert an input sample to 3 output samples for each
  train/test/val split.

- Relies on caching and partition coalesce for performance optimization.

- Can be run on mac, gCloud VMs and Dataproc VMs. (cmds are available in Makefile)

## Spark Optimizations / Important points to consider

- The current spark plan only contains one Deserialization and Serialization step. It is important that we keep this
  behaviour as it directly affects the performance of the job.
- We use a coalesce factor to decrease the input partitions and in turn increase the CPU utilization. Currently this is
  hardcoded in the code to 12 (for main samples) and 4 (for rooted node neighborhood samples) as this was fastest. If
  the number of input partiotions is changed by SGS job, we can reconsider these numbers.
