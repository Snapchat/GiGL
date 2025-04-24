package libs.task

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.{functions => F}

object SamplingStrategy {

  object PermutationStrategy extends Enumeration {
    type PermutationStrategy = String
    val Deterministic    = "deterministic"
    val NonDeterministic = "non-deterministic"
  }

  private var _counter = 1

  def hashBasedUniformPermutation(
    sortedArrayDF: DataFrame,
    arrayColName: String,
    seed: Integer,
  ): DataFrame = {

    /**
          * Permutes node ids using hash strategy satisfying 3 conditions for down stream uniform sampling:
          * 1. sampling is deterministic (reproducible) given the same seed
          * 2. sampling is diverse, meaning for multiple appearance of a node for which we want to sample it's neighbors, the subtrees of that node are different
          * 3. sampling is diverse any time this function is called, given that, `seed` is different; i.e., for 1-hop sampling and 2-hop seed must be different, otherwise the subtrees of each hop will be the same!
          * @param sortedArrayDF: a DF with an array column, from which we will later take sample. This DF should have the array column sorted. Columns of this DF are:
          * col 1, col 2, ... col n, arrayColName, where all col 1, ... col n are integer cols and arrayColName is a sorted array of integers
          * @param arrayColName: name of the column including sorted array of Ints from which we later take samples
          * @param seed: a seed for shifting hash values, this seed MUST be different any time this func is called, to satisfy condition 3
          * @return shuffledArrayDF, with cols:
          * col 1, col 2, ... col n, _shuffled_${arrayColName}
          * down stream sampling queries, can use this DF to take first N elements as sampled node ids
          */

    var currentSeed: Integer = seed * _counter
    println(f"Permuting node ids using hash based method with seed =${currentSeed}")
    // use sum of all integer columns to create a unique internal seed, to satisfy condition 2
    val colsToCreateInternalSeedFrom =
      sortedArrayDF.columns.filter(c => !c.contains(arrayColName)).mkString("+")
    val sortedArrayWithSizeDF = sortedArrayDF.select(
      F.col("*"),
      F.size(F.col(arrayColName)).alias("_size"),
      F.expr(colsToCreateInternalSeedFrom).alias("_internal_seed"),
    )
    // generate an array col of indices starting from 1
    val indicesDF = sortedArrayWithSizeDF.select(
      F.col("*"),
      F.sequence(F.lit(1), F.col("_size")).alias("_indices"),
    )

    val hashDF = indicesDF
      .select(
        F.col("*"),
        F.expr(f"transform(_indices, x -> xxhash64(x+_internal_seed+ ${currentSeed})) as _hash"),
      )
      .drop("_size")
      .drop("_internal_seed")

    val permutedIndicesDF = hashDF
      .select(
        F.col("*"),
        F.array_sort(F.arrays_zip(F.col("_hash"), F.col("_indices"))).alias("_permuted"),
      )
      .drop("_hash")
      .drop("_indices")

    val shuffledArrayDF = permutedIndicesDF
      .select(
        F.col("*"),
        F.expr(
          f"transform(_permuted._indices, i-> element_at(${arrayColName}, i)) as _shuffled${arrayColName}",
        ),
      )
      .drop("_permuted")
      .drop(f"${arrayColName}")

    // increment counter any time this function is called to ensure we use a different seed for each call
    _counter += 1
    shuffledArrayDF

  }

  def shuffleBasedUniformPermutation(
    sortedArrayDF: DataFrame,
    arrayColName: String,
  ): DataFrame = {

    /**
          * This function permutes node ids usnig `shuffle` function, which is non-deterministic
          */
    println("permuting node ids using Sparks native function shuffle, which is not deterministic")
    val shuffledArrayCols = sortedArrayDF.columns.map(c => {
      if (c.contains(arrayColName)) F.shuffle(F.col(c)).alias(f"_shuffled${arrayColName}")
      else F.col(c)
    })

    val shuffledArrayDF = sortedArrayDF.select(shuffledArrayCols: _*)

    shuffledArrayDF
  }
}
