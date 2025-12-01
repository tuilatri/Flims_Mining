import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class ZeroR_Classification {
    public static void main(String[] args) {
        try {
            // Load data
            DataSource source = new DataSource("../pre-processing-21/combined-cleaned.arff");
            Instances data = source.getDataSet();

            // Set class index to the last attribute (members_encoded)
            data.setClassIndex(data.numAttributes() - 1);

            // 1. Build model on full training set
            Classifier classifier = new ZeroR();

            long startTime = System.currentTimeMillis();
            classifier.buildClassifier(data);
            long endTime = System.currentTimeMillis();
            double timeTaken = (endTime - startTime) / 1000.0;

            // Print Run Information
            System.out.println("=== Run information ===");
            System.out.println();
            System.out.println("Scheme:       " + classifier.getClass().getName());
            System.out.println("Relation:     " + data.relationName());
            System.out.println("Instances:    " + data.numInstances());
            System.out.println("Attributes:   " + data.numAttributes());
            for (int i = 0; i < data.numAttributes(); i++) {
                System.out.println("              " + data.attribute(i).name());
            }
            System.out.println("Test mode:    10-fold cross-validation");
            System.out.println();

            // Print Classifier model
            System.out.println("=== Classifier model (full training set) ===");
            System.out.println();
            System.out.println(classifier.toString());
            System.out.println("Time taken to build model: " + String.format("%.2f", timeTaken) + " seconds");
            System.out.println();

            // 2. Cross-validation
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(1));

            System.out.println("=== Stratified cross-validation ===");
            System.out.println(evaluation.toSummaryString("=== Summary ===", false));
            System.out.println(evaluation.toClassDetailsString("=== Detailed Accuracy By Class ==="));
            System.out.println(evaluation.toMatrixString("=== Confusion Matrix ==="));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
