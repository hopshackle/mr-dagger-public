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

import java.io.{ File, FileInputStream, FileOutputStream }

object FileInstances {
  def copyFile(source: String, destination: String) {
    val src = new File(source)
    val dest = new File(destination)
    new FileOutputStream(dest).getChannel.transferFrom(new FileInputStream(src).getChannel, 0, Long.MaxValue)
  }
}
class InstanceIterator[T: ClassTag](files: List[String], stringToAction: (String => T), actionToString: (T => String), errorLimit: Int) extends Iterator[Instance[T]] {

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
    if (!currentFile.hasNext && nextFileIndex < files.size) {
      currentFile = nextFileContents(files(nextFileIndex))
      nextFileIndex += 1
    }
    lastOutput = output
    output
  }

  def nextFileContents(fileName: String): Iterator[Instance[T]] = {
    FileInstances.copyFile(fileName, fileName + "_tmp")
    val lines = io.Source.fromFile(fileName + "_tmp").getLines

    new Iterator[Instance[T]] {
      def hasNext: Boolean = lines.hasNext

      def next: Instance[T] = {
        val input = new StringBuffer
        var endOfInstance = false
        while (!endOfInstance) {
          val l = lines.next
          input.append(l + "\n")
          if (l == "END") endOfInstance = true
        }
        Instance.construct(input.toString, stringToAction)
      }
    }
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
  