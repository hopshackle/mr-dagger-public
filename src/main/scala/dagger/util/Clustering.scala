package coref.util

import collection.immutable.Map
/**
 * Created by narad on 4/8/15.
 */
case class Clustering[T](elemToCluster: Map[T, Int], clusterToElems: Map[Int, List[T]]) {

  def add(t: T): Clustering[T] = {
    val n = elemToCluster.size
    Clustering(elemToCluster + (t -> n), clusterToElems + (n -> List(t)))
  }

  def addToCluster(t: T, i: Int): Clustering[T] = {
    Clustering(elemToCluster + (t -> i), clusterToElems + (i -> (t +: clusterToElems(i))))
  }

  def cluster(i: Int): Option[List[T]] = clusterToElems.get(i)

  def clusterOf(t: T): Option[Int] = {
    elemToCluster.get(t)
  }

  def hasCluster(t: T): Boolean = clusterOf(t).isDefined

  def isEmpty = elemToCluster.isEmpty

  def numClusters = clusterToElems.size

  def numElems = elemToCluster.size

  def toClusters = {
    clusterToElems // elemToCluster.keys.groupBy(elemToCluster(_))
  }

  def sharesCluster(t1: T, t2: T): Boolean = {
    elemToCluster(t1) == elemToCluster(t2)
  }
}

object Clustering {

  def empty[T]: Clustering[T] = {
    new Clustering[T](Map[T, Int](), Map[Int, List[T]]())
  }
}
