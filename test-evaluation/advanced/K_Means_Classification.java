import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class K_Means_Classification {
    public static void main(String[] args) {
        try {
            // Load data
            DataSource source = new DataSource("../pre-processing-20/combined-cleaned.arff");
            Instances data = source.getDataSet();

            // Unset class index for clustering (use all attributes)
            data.setClassIndex(-1);

            // Create clusterer
            SimpleKMeans clusterer = new SimpleKMeans();
            String[] options = Utils.splitOptions(
                    "-init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -N 2 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -num-slots 1 -S 10");
            clusterer.setOptions(options);

            // Print Run information
            System.out.println("=== Run information ===");
            System.out.println();
            System.out.println(
                    "Scheme:       weka.clusterers.SimpleKMeans -init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -N 2 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -num-slots 1 -S 10");
            System.out.println("Relation:     " + data.relationName());
            System.out.println("Instances:    " + data.numInstances());
            System.out.println("Attributes:   " + data.numAttributes());
            for (int i = 0; i < data.numAttributes(); i++) {
                System.out.println("              " + data.attribute(i).name());
            }
            System.out.println("Test mode:    evaluate on training data");
            System.out.println();

            // Build model on full training set
            System.out.println("=== Clustering model (full training set) ===");
            System.out.println();

            long startTime = System.currentTimeMillis();
            clusterer.buildClusterer(data);
            long endTime = System.currentTimeMillis();

            System.out.println(clusterer.toString());
            System.out.println();
            System.out.println("Time taken to build model (full training data) : "
                    + String.format("%.2f", (endTime - startTime) / 1000.0).replace(",", ".") + " seconds");
            System.out.println();

            // Evaluation on training set
            System.out.println("=== Model and evaluation on training set ===");
            System.out.println();

            ClusterEvaluation eval = new ClusterEvaluation();
            eval.setClusterer(clusterer);
            eval.evaluateClusterer(data);

            System.out.println(eval.clusterResultsToString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
