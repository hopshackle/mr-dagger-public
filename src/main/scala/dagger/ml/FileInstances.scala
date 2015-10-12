package dagger.ml

import scala.collection.Iterable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class FileInstances[T: ClassTag](files: List[String], stringToAction: (String => T)) extends Iterable[Instance[T]] {

  def iterator: Iterator[Instance[T]] = {
    new InstanceIterator[T](files, stringToAction)
  }
}

class InstanceIterator[T: ClassTag](files: List[String], stringToAction: (String => T)) extends Iterator[Instance[T]] {

  var fileCursor = 0
  var currentFile = nextFileContents(files(0))
  var nextFileIndex = 1

  def hasNext: Boolean = {
    fileCursor < currentFile.length || nextFileIndex < files.size
  }

  def next: Instance[T] = {
    val output = currentFile(fileCursor)
    fileCursor += 1
    if (fileCursor >= currentFile.length && nextFileIndex < files.size) {
      currentFile = nextFileContents(files(nextFileIndex))
      nextFileIndex += 1
      fileCursor = 0
    }
    output
  }

  def nextFileContents(fileName: String): Array[Instance[T]] = {
    val lines = io.Source.fromFile(fileName).getLines
    var currentString = ""
    val instances = new ArrayBuffer[Instance[T]]
    var input = ""
    for (l <- lines) {
      input += l + "\n"
      if (l == "END") {
        instances += Instance.construct(input, stringToAction)
        input = ""
      }
    }
    instances.toArray
  }

}
  