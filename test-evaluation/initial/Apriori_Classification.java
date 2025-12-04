import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;

public class Apriori_Classification {
    public static void main(String[] args) {
        try {
            // Load data
            DataSource source = new DataSource("../../dataset/initial/combined-cleaned.arff");
            Instances data = source.getDataSet();

            // No class index for Apriori (or -c -1 handles it)
            // data.setClassIndex(data.numAttributes() - 1); // Do not set

            // Initialize Apriori with options
            Apriori associator = new Apriori();
            String options = "-N 10 -T 0 -C 0.9 -D 0.05 -U 1.0 -M 0.1 -S -1.0 -c -1";
            associator.setOptions(Utils.splitOptions(options));

            // 1. Build associator on full training set
            associator.buildAssociations(data);

            // Print Run Information
            System.out.println("=== Run information ===");
            System.out.println();
            String scheme = associator.getClass().getName() + " " + Utils.joinOptions(associator.getOptions());
            System.out.println("Scheme:       " + scheme);
            System.out.println("Relation:     " + data.relationName());
            System.out.println("Instances:    " + data.numInstances());
            System.out.println("Attributes:   " + data.numAttributes());
            for (int i = 0; i < data.numAttributes(); i++) {
                System.out.println("              " + data.attribute(i).name());
            }
            System.out.println("=== Associator model (full training set) ===");
            System.out.println();
            System.out.println(associator.toString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
