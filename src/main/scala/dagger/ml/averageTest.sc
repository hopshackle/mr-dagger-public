package dagger.ml

import collection.mutable.HashMap

object averageTest {

val weights1 = new HashMap[Int, Float]            //> weights1  : scala.collection.mutable.HashMap[Int,Float] = Map()
weights1.put(3, 0.5f)                             //> res0: Option[Float] = None
weights1.put(4, 0.5f)                             //> res1: Option[Float] = None

val weights2 = new HashMap[Int, Float]            //> weights2  : scala.collection.mutable.HashMap[Int,Float] = Map()
weights2.put(3, 1.0f)                             //> res2: Option[Float] = None
weights2.put(10, 0.5f)                            //> res3: Option[Float] = None

weights1                                          //> res4: scala.collection.mutable.HashMap[Int,Float] = Map(4 -> 0.5, 3 -> 0.5)
                                                  //| 
weights2                                          //> res5: scala.collection.mutable.HashMap[Int,Float] = Map(10 -> 0.5, 3 -> 1.0)
                                                  //| 
val aw1 = new HashMap[String, HashMap[Int, Float]]//> aw1  : scala.collection.mutable.HashMap[String,scala.collection.mutable.Hash
                                                  //| Map[Int,Float]] = Map()
aw1.put("one", weights1)                          //> res6: Option[scala.collection.mutable.HashMap[Int,Float]] = None
aw1.put("two", weights2)                          //> res7: Option[scala.collection.mutable.HashMap[Int,Float]] = None
aw1                                               //> res8: scala.collection.mutable.HashMap[String,scala.collection.mutable.HashM
                                                  //| ap[Int,Float]] = Map(one -> Map(4 -> 0.5, 3 -> 0.5), two -> Map(10 -> 0.5, 3
                                                  //|  -> 1.0))
val c1 = new AROWClassifier(aw1)                  //> c1  : dagger.ml.AROWClassifier[String] = AROWClassifier(Map(one -> Map(4 -> 
                                                  //| 0.5, 3 -> 0.5), two -> Map(10 -> 0.5, 3 -> 1.0)),Map())
val aw2 = new HashMap[String, HashMap[Int, Float]]//> aw2  : scala.collection.mutable.HashMap[String,scala.collection.mutable.Hash
                                                  //| Map[Int,Float]] = Map()
aw2.put("one", weights2)                          //> res9: Option[scala.collection.mutable.HashMap[Int,Float]] = None
aw2.put("three", weights1)                        //> res10: Option[scala.collection.mutable.HashMap[Int,Float]] = None
val c2 = new AROWClassifier(aw2)                  //> c2  : dagger.ml.AROWClassifier[String] = AROWClassifier(Map(one -> Map(10 ->
                                                  //|  0.5, 3 -> 1.0), three -> Map(4 -> 0.5, 3 -> 0.5)),Map())
                                                  
val c3 = AROW.average(c1, c2, 3)                  //> c3  : dagger.ml.AROWClassifier[String] = AROWClassifier(Map(one -> Map(4 -> 
                                                  //| 0.375, 10 -> 0.125, 3 -> 0.625), three -> Map(4 -> 0.125, 3 -> 0.125), two -
                                                  //| > Map(10 -> 0.375, 3 -> 0.75)),Map())
AROW.average(c3, c1, 4)                           //> res11: dagger.ml.AROWClassifier[String] = AROWClassifier(Map(one -> Map(4 ->
                                                  //|  0.4, 10 -> 0.1, 3 -> 0.6), three -> Map(4 -> 0.1, 3 -> 0.1), two -> Map(10 
                                                  //| -> 0.4, 3 -> 0.8)),Map())
}