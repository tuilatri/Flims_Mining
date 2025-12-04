import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class Apriori_Classification {
    public static void main(String[] args) {
        try {
            // Load data
            DataSource source = new DataSource("../../dataset/advanced/combined-cleaned.arff");
            Instances data = source.getDataSet();

            // Unset class index (Apriori doesn't use class index in the same way, usually
            // -c -1 means no class)
            data.setClassIndex(-1);

            // Create associator
            Apriori associator = new Apriori();
            String[] options = Utils.splitOptions("-N 10 -T 0 -C 0.9 -D 0.05 -U 1.0 -M 0.1 -S -1.0 -c -1");
            associator.setOptions(options);

            // Print Run information
            System.out.println("=== Run information ===");
            System.out.println();
            System.out.println(
                    "Scheme:       weka.associations.Apriori -N 10 -T 0 -C 0.9 -D 0.05 -U 1.0 -M 0.1 -S -1.0 -c -1");
            System.out.println("Relation:     " + data.relationName());
            System.out.println("Instances:    " + data.numInstances());
            System.out.println("Attributes:   " + data.numAttributes());
            for (int i = 0; i < data.numAttributes(); i++) {
                System.out.println("              " + data.attribute(i).name());
            }
            System.out.println("=== Associator model (full training set) ===");
            System.out.println();

            associator.buildAssociations(data);

            System.out.println(associator.toString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
