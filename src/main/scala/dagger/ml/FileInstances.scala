package dagger.ml

import scala.collection.Iterable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import java.io._

class FileInstances[T: ClassTag](files: List[String], stringToAction: (String => T), actionToString: (T => String), errorLimit: Int) extends Iterable[Instance[T]] {

  def iterator: Iterator[Instance[T]] = {
    new InstanceIterator[T](files, stringToAction, actionToString, errorLimit)
  }
}

class InstanceIterator[T: ClassTag](files: List[String], stringToAction: (String => T), actionToString: (T => String), errorLimit: Int) extends Iterator[Instance[T]] {

  var fileCursor = 0
  var currentFile = nextFileContents(files(0))
  var nextFileIndex = 1
  var lastOutput: Instance[T] = null
  var lastFileWrittenTo: String = null
  var currentFileBeingWrittenTo: FileWriter = null

  def hasNext: Boolean = {
    currentFile.hasNext match {
      case true => true
      case false => {
        writeBackToFile(lastOutput)
        currentFileBeingWrittenTo.close
        false
      }
    }
  }

  def next: Instance[T] = {
    val output = currentFile.next
    writeBackToFile(lastOutput)
    fileCursor += 1
    if (!currentFile.hasNext && nextFileIndex < files.size) {
      currentFile = nextFileContents(files(nextFileIndex))
      nextFileIndex += 1
      fileCursor = 0
    }
    lastOutput = output
    output
  }

  def nextFileContents(fileName: String): Iterator[Instance[T]] = {
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
    instances.toIterator
  }

  def writeBackToFile(i: Instance[T]) {
    if (currentFileBeingWrittenTo != null && i != null && i.getErrorCount < errorLimit)
      currentFileBeingWrittenTo.write(i.fileFormat(actionToString))
    val fileName = files(nextFileIndex - 1)

    if (lastFileWrittenTo != fileName) {
      if (currentFileBeingWrittenTo != null) currentFileBeingWrittenTo.close
      currentFileBeingWrittenTo = new FileWriter(fileName, false)
      lastFileWrittenTo = fileName
    }

  }
}
  