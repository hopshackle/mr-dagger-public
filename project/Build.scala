import java.io.IOException
import sbt._
import sbt.Keys._
import sbtrelease.ReleasePlugin._


object ShellPrompt {
  object devnull extends ProcessLogger {
    def info(s: => String) {}
    def error(s: => String) {}
    def buffer[T](f: => T): T = f
  }
  def currBranch = {
    try {
      (
      ("git status -sb" lines_! devnull headOption)
      getOrElse "-" stripPrefix "## "
      )
    } catch {
      case ex: IOException => "?"
    }
  }

  val buildShellPrompt = {
    (state: State) => {
      val extracted = Project.extract(state)
      val currProject = extracted.currentProject.id
      "%s:%s:%s> ".format(
        currProject, currBranch, extracted.get(version)
      )
    }
  }
}

object BuildSettings {
  val buildName         = "mr-dagger"
  val buildOrganization = "uclmr"
  val buildScalaVersion = "2.11.4"
  val buildSbtVersion = "0.13.7"

  val buildSettings = Defaults.defaultSettings ++ Seq(
    organization := buildOrganization,
    scalaVersion := buildScalaVersion,
    scalacOptions := Seq("-unchecked", "-deprecation", "-feature"),
    shellPrompt := ShellPrompt.buildShellPrompt,
    fork in run := true //use a fresh JVM for sbt run
  )

  val globalDependencies = libraryDependencies ++= Seq(
    "org.scalautils" % "scalautils_2.11" % "2.1.5",
    "org.scalatest" % "scalatest_2.11" % "2.2.1",
    "org.apache.commons" % "commons-compress" % "1.8",
    "com.typesafe.scala-logging" % "scala-logging-slf4j_2.11" % "2.1.2",
    "net.sf.trove4j" % "trove4j" % "3.0.3"
  )

  val publishSettings = Seq(
    publishTo <<= version {
      version: String =>
        val homeniscient = "http://homeniscient.cs.ucl.ac.uk:8081/nexus/content/repositories/"
        if (version.trim.endsWith("SNAPSHOT")) Some("snapshots" at homeniscient + "snapshots/")
        else Some("releases" at homeniscient + "releases/")
    },
    credentials += Credentials(Path.userHome / ".ivy2" / ".credentials-homeniscient")
  )

  def vmargs = Command.args("vmargs", "<name>") {
    (state, args) =>
      val javaRunOptions = args.mkString(" ")
      println("Applying JVM arguments: " + javaRunOptions)
      Project.extract(state).append(javaOptions := Seq(javaRunOptions), state)
  }


  val globalSettings =
    Seq(
      commands ++= Seq(vmargs),
      scalacOptions ++= Seq("-feature"),
	  ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true) },
      resolvers ++= Seq(
        "IESL Release" at "https://dev-iesl.cs.umass.edu/nexus/content/groups/public",
     	"Scala-Tools Maven2 Snapshots Repository" at "http://scala-tools.org/repo-snapshots",
        Resolver.sonatypeRepo("snapshots"),
        Resolver.sonatypeRepo("releases")
      ),
      globalDependencies
    ) ++ buildSettings ++ releaseSettings ++ publishSettings

}


object Build extends Build {

  import BuildSettings._

  lazy val root = Project(
    id = "mr-dagger",
    base = file("."),
    settings = Project.defaultSettings ++ globalSettings
  ) 
}
