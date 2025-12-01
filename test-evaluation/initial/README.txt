Instructions to run the Java files:

1. Ensure Java Development Kit (JDK) is installed and 'javac'/'java' are in your PATH.
2. Ensure 'weka.jar' is located in the '../lib' directory relative to this folder.
3. Open a terminal/command prompt in this folder ('implementation-03').
4. Run the 'compile_and_run.bat' script to compile and run all algorithms.
   > compile_and_run.bat

Alternatively, to run manually:

1. Compile:
   javac -cp ../lib/weka.jar *.java

2. Run each file (example for ZeroR):
   java -cp ".;../lib/weka.jar" ZeroR_Classification

The results will be saved in the 'result' folder.
