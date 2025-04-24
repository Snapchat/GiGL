package splitgenerator.lib.assigners

import common.types.pb_wrappers.GraphMetadataPbWrapper

import java.lang.Math.floorMod
import scala.collection.immutable.Seq
import scala.util.hashing.MurmurHash3

object AbstractAssigners {
  abstract class Assigner[T, S] extends Serializable {

    /**
      * Some assigners will want to access GraphMetadata to handle assignment logic.
      * This is because determining how a certain entity should should be assigned may depend on
      * certain (readable) node or edge types, field values (e.g. timestamps), etc.
      */
    val graphMetadataPbWrapper: GraphMetadataPbWrapper

    /**
        * Assigns an object (type T) into a fixed category (type S).
        * Used for assigning graph entities like nodes / edges based on custom logic.
        * e.g. could be assigning a NodePb (T) to some Enum (S).
        *
        * @param obj the object to hash
        * @return 
        */
    def assign(obj: T): S
  }

  abstract class HashingAssigner[T, S] extends Assigner[T, S] {

    /**
      * Designates assignment of objects (of type T) to buckets (of type S) via a hashing protocol.
      * This is an abstract class, where the user must further implement:
      * - "coder" which can encode objects of type T to bytes.
      * - "bucket_weights" which indicates how much of the hash space each bucket should cover.
      */

    val HashSpaceGranularity = 10000

    /**
        * Returns a byte representation of the given object, which will be used to hash it.
        * Note that if two objects produce the same byte representation, they will be hashed in the same way.
        * @param obj
        * @return
        */
    def coder(obj: T): Array[Byte]

    /**
      * Weights of each bucket; determines partitioning of the hash space.
      * @return
      */
    val bucketWeights: Map[S, Float]

    /**
      * Unique buckets which an object can be hashed to.
      */
    lazy val buckets: Seq[S] = bucketWeights.keys.toList

    /**
      * Relative width of each bucket in the hash space.  e.g. [0.2, 0.4, 0.4] would indicate 3 buckets, where
      * the second and third bucket are twice as prominent as the first bucket. 
      */
    lazy val weights: Seq[Float] = bucketWeights.values.toList

    /**
      * Return indices marking the transition points in the hash space.  e.g. [0, 5000, 10000] would indicate
      * partitioning the hash space into 2 equal-sized buckets.
      */
    private lazy val hashBucketIndices: Seq[Int] = {
      val normedWeights: Seq[Float]           = weights.map(_ / weights.sum)
      val cumulativeNormedWeights: Seq[Float] = normedWeights.scanLeft(0.0f)(_ + _)
      cumulativeNormedWeights.map(cw => math.round(cw * HashSpaceGranularity))
    }

    /**
        * For a given hash value, assign it to a bucket based on which span of the hash space it falls in.
        *
        * @param hashValue
        * @return
        */
    private def getBucketAssignmentForHashValue(hashValue: Int): S = {
      assert(
        0 <= hashValue && hashValue < HashSpaceGranularity,
        s"Encountered hash value $hashValue outside hash space, which is bounded on [0, $HashSpaceGranularity).",
      )

      buckets
        .zip(hashBucketIndices.zip(hashBucketIndices.tail))
        .find { case (_, (hashSlotLow, hashSlotHigh)) =>
          hashValue >= hashSlotLow && hashValue < hashSlotHigh
        }
        .get // we are certain here that a bucket exists
        ._1
    }

    /**
        * Hash an object into a hash space (partitioned by buckets), and return the relevant bucket.
        *
        * @param obj
        * @return
        */
    override def assign(obj: T): S = {
      val byteString = coder(obj)
      val hashSlotForObj = floorMod(
        MurmurHash3.bytesHash(data = byteString),
        HashSpaceGranularity,
      ) // floormod to keep the modulo positive
      getBucketAssignmentForHashValue(hashValue = hashSlotForObj)
    }
  }
}
