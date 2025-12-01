import weka.clusterers.SimpleKMeans;
import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;

public class K_Means_Classification {
    public static void main(String[] args) {
        try {
            // Load data
            DataSource source = new DataSource("../pre-processing-21/combined-cleaned.arff");
            Instances data = source.getDataSet();

            // Do NOT set class index for clustering as per output analysis
            // data.setClassIndex(0);

            // Initialize SimpleKMeans with options
            SimpleKMeans clusterer = new SimpleKMeans();
            String options = "-init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -N 2 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -num-slots 1 -S 10";
            clusterer.setOptions(Utils.splitOptions(options));

            // 1. Build clusterer on full training set
            long startTime = System.currentTimeMillis();
            clusterer.buildClusterer(data);
            long endTime = System.currentTimeMillis();
            double timeTaken = (endTime - startTime) / 1000.0;

            // Print Run Information
            System.out.println("=== Run information ===");
            System.out.println();
            String scheme = clusterer.getClass().getName() + " " + Utils.joinOptions(clusterer.getOptions());
            System.out.println("Scheme:       " + scheme);
            System.out.println("Relation:     " + data.relationName());
            System.out.println("Instances:    " + data.numInstances());
            System.out.println("Attributes:   " + data.numAttributes());
            for (int i = 0; i < data.numAttributes(); i++) {
                System.out.println("              " + data.attribute(i).name());
            }
            System.out.println("Test mode:    evaluate on training data");
            System.out.println();

            // Print Clustering model
            System.out.println("=== Clustering model (full training set) ===");
            System.out.println();
            System.out.println(clusterer.toString());
            System.out.println("Time taken to build model (full training data) : " + String.format("%.2f", timeTaken)
                    + " seconds");
            System.out.println();

            // 2. Evaluation
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
