import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class ZeroR_Classification {
    public static void main(String[] args) {
        try {
            // Load data
            DataSource source = new DataSource("../pre-processing-20/combined-cleaned.arff");
            Instances data = source.getDataSet();

            // Set class index to the last attribute
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

            // Create classifier
            ZeroR classifier = new ZeroR();

            // Print Run information
            System.out.println("=== Run information ===");
            System.out.println();
            System.out.println("Scheme:       weka.classifiers.rules.ZeroR ");
            System.out.println("Relation:     " + data.relationName());
            System.out.println("Instances:    " + data.numInstances());
            System.out.println("Attributes:   " + data.numAttributes());
            for (int i = 0; i < data.numAttributes(); i++) {
                System.out.println("              " + data.attribute(i).name());
            }
            System.out.println("Test mode:    10-fold cross-validation");
            System.out.println();

            // Build model on full training set
            System.out.println("=== Classifier model (full training set) ===");
            System.out.println();

            long startTime = System.currentTimeMillis();
            classifier.buildClassifier(data);
            long endTime = System.currentTimeMillis();

            System.out.println(classifier.toString());
            System.out.println();
            System.out.println("Time taken to build model: "
                    + String.format("%.2f", (endTime - startTime) / 1000.0).replace(",", ".") + " seconds");
            System.out.println();

            // Cross-validation
            System.out.println("=== Stratified cross-validation ===");

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new Random(1));

            System.out.println(eval.toSummaryString(false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
