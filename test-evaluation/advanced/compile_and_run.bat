@echo off
if not exist result mkdir result

echo Compiling Java files...
javac -cp "../lib/weka.jar" *.java
if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Running ZeroR...
java -cp "../lib/weka.jar;." ZeroR_Classification > result/ZeroR_Classification.txt

echo Running OneR...
java -cp "../lib/weka.jar;." OneR_Classification > result/OneR_Classification.txt

echo Running J48...
java -cp "../lib/weka.jar;." J48_Classification > result/J48_Classification.txt

echo Running Naive Bayes...
java -cp "../lib/weka.jar;." Naive_Bayes_Classification > result/Naive_Bayes_Classification.txt

echo Running K-Means...
java -cp "../lib/weka.jar;." K_Means_Classification > result/K_Means_Classification.txt

echo Running Apriori...
java -cp "../lib/weka.jar;." Apriori_Classification > result/Apriori_Classification.txt

echo Done. Check the result folder.
pause
