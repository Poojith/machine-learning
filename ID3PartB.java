/**
 * Decision tree implementation for the prediction of products.
 * Team 2 - MSIT eBusiness Technology, Carnegie Mellon University
 */

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.Map.Entry;

/**
 * Class that represents an instance of a product from the data set.
 */

class ProductInfo {
    String service_type;
    String customer;
    double monthly_fee;
    double advertisement_budget;
    String size;
    String promotion;
    double interest_rate;
    double period;
    String label;

    public ProductInfo(String service_type, String customer, double monthly_fee, double advertisement_budget,
                       String size, String promotion, double interest_rate, double period, String label) {
        this.service_type = service_type;
        this.customer = customer;
        this.monthly_fee = monthly_fee;
        this.advertisement_budget = advertisement_budget;
        this.size = size;
        this.promotion = promotion;
        this.interest_rate = interest_rate;
        this.period = period;
        this.label = label;
    }
}

/**
 * Class that represents the structure of a node in the decision tree.
 */

class TreeNode {
    Map<Double, TreeNode> children = new HashMap<>();
    String attribute;
    boolean isLeaf;
    String outputLabel;
    public TreeNode(String attribute) {
        this.attribute = attribute;
    }
}


/**
 * Class for the ID3 Decision Tree (Part B - prediction of products).
 */

public class ID3PartB {

    private static final String SERVICE_TYPE = "type";
    private static final String CUSTOMER = "customer";
    private static final String MONTHLY_FEE = "monthly_fee";
    private static final String ADVERTISEMENT_BUDGET = "advertisement_budget";
    private static final String SIZE = "size";
    private static final String PROMOTION = "promotion";
    private static final String INTEREST_RATE = "interest_rate";
    private static final String PERIOD = "period";

    private static List<String> attributes = new ArrayList<>(); // list of attributes from the data set

    /**
     * Method to calculate the entropy for a given set of data.
     * @param tempInstanceList for which the entropy has to be calculated
     * @return value of entropy in double
     */

    private static double calculateEntropy(List<ProductInfo> tempInstanceList) {
        int numInstances = tempInstanceList.size();

        if (numInstances == 0) {
            // Return an arbitrary high entropy value when number of instances is zero.
            return 9999;
        }

        double[] labelCounts = new double[2];
        for (ProductInfo productInfo : tempInstanceList) {
            // Calculate the number of occurrences of each label
            switch (productInfo.label) {
                case "1":
                    labelCounts[0]++;
                    break;
                case "0":
                    labelCounts[1]++;
                    break;
                default:
                    System.out.println("Invalid class label");
                    break;
            }
        }

        double entropy = 0;

        for (double labelCount : labelCounts) {
            double probability = labelCount / numInstances;
            entropy += (probability) * Math.log(1 / probability); // Compute entropy
        }
        return entropy;
    }

    /**
     * Method to retrieve the list of possible values for a certain attribute.
     * @param attribute for which the values have to be obtained.
     * @return List of values for the attribute. For non-numeric values, each double value is mapped to its
     * corresponding String literal.
     */

    public List<Double> getPossibleValues(String attribute) {
        List<Double> list = new ArrayList<>();
        if (attribute.equals(SERVICE_TYPE)) {
            list.add(0.0);
            list.add(1.0);
            list.add(2.0);
            list.add(3.0);
            list.add(4.0);
        } else if (attribute.equals(CUSTOMER)) {
            list.add(0.0);
            list.add(1.0);
            list.add(2.0);
            list.add(3.0);
            list.add(4.0);
        } else if (attribute.equals(SIZE)) {
            list.add(0.0);
            list.add(1.0);
            list.add(2.0);
        } else if (attribute.equals(PROMOTION)) {
            list.add(0.0);
            list.add(1.0);
            list.add(2.0);
            list.add(3.0);
        } else {
            list.add(0.25);
            list.add(0.5);
            list.add(0.75);
            list.add(1.0);
        }
        return list;
    }

    /**
     * Method to retrieve the attribute with the highest information gain.
     * @param data for which the information gain is computed
     * @param remainingAttributes - list of attributes for which the information gain is computed.
     *                            Note : Whenever an attribute is chosen as a node of the tree, it is removed from the list.
     *
     * @return the attribute with the highest information gain
     */

    public String getAttributeWithHighestGain(List<ProductInfo> data, List<String> remainingAttributes) {
        HashMap<String, Double> gainMap = new HashMap<>();

        for (String attribute : remainingAttributes) {
            gainMap.put(attribute, computeInformationGain(data, attribute));
        }

        List<Entry<String, Double>> gainList = new ArrayList<>(gainMap.entrySet());
        Collections.sort(gainList, (o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        return gainList.size() == 0 ? null : gainList.get(0).getKey();
    }

    /**
     * Method to check if a node is pure when the tree traversal has reached a leaf.
     * @param data among which the label is determined
     * @param majority - boolean flag that is used when all attributes are exhausted. (Majority label is returned)
     * @return
     */

    public String checkPureNode(List<ProductInfo> data, boolean majority) {
        Map<String, Integer> labelCounts = new HashMap<>();

        for (ProductInfo product : data) {
            if (!labelCounts.containsKey(product.label)) {
                labelCounts.put(product.label, 1);
            }
            labelCounts.put(product.label, labelCounts.get(product.label) + 1);
        }

        if (majority) { // Return the attribute that has the highest information gain among the remaining attributes
            List<Entry<String, Integer>> labelList = new ArrayList<>(labelCounts.entrySet());

            Collections.sort(labelList, (o1, o2) -> o2.getValue().compareTo(o1.getValue()));
            return labelList.size() == 0 ? null : labelList.get(0).getKey();
        }

        int size = data.size();
        for (String key : labelCounts.keySet()) {
            int count = labelCounts.get(key);
            if (count > 0.7 * size) // Prune to 70%
                return key;
        }
        return null;
    }

    /**
     * Method to compute the information gain for an attribute from the list of data.
     * @param data - superset of data among which the information gain must be computed
     * @param attribute for which the information gain is computed
     * @return the information gain value for the attribute in double
     */

    public double computeInformationGain(List<ProductInfo> data, String attribute) {
        double entropy = calculateEntropy(data); // Calculate entropy for the list of data

        List<List<ProductInfo>> chunkedData = getFilteredData(data, attribute); // Get filtered data for this particular attribute

        List<Double> possibleValues = getPossibleValues(attribute); // Get all possible values for this attribute
        double S = data.size();

        for (int i = 0; i < possibleValues.size(); i++) {
            double S_v = chunkedData.get(i).size();
            entropy -= ((S_v / S) * calculateEntropy(chunkedData.get(i))); // Compute entropy for the attribute
        }

        return entropy;
    }

    /**
     * Method to extract the set of data for a given an attribute
     * @param data - the set in which data is filtered for a given attribute
     * @param attribute for which the data is filtered
     * @return A list that contains lists. Each of the contained lists are chunks of data for the attribute.
     * Note : For non-numeric values, there are four chunks, equally divided from 0.0 to 1.0.
     */

    private List<List<ProductInfo>> getFilteredData(List<ProductInfo> data, String attribute) {
        List<List<ProductInfo>> chunked = new ArrayList<>();
        if (attribute.equals(SIZE)) {
            for (int i = 0; i < 3; i++) {
                chunked.add(new ArrayList<>());
            }
        } else if (attribute.equals(CUSTOMER) || attribute.equals(SERVICE_TYPE)) {
            for (int i = 0; i < 5; i++) {
                chunked.add(new ArrayList<>());
            }
        } else {
            for (int i = 0; i < 4; i++) {
                chunked.add(new ArrayList<>());
            }
        }

        /**
         * Get the mapped value of the index(for the list) to which the chunk of data is added.
         * The arguments to getValueMap correspond to whether the attribute has numeric or non-numeric values.
         * For numeric valued attributes, the actual value of the instance is passed with the String parameter set as null.
         * For non-numeric valued attributes, the String value of the instance is passed with the double parameter set to 0.0.
         */
        switch (attribute) {
            case SERVICE_TYPE:
                for (ProductInfo product : data) {
                    int index = getValueMap(SERVICE_TYPE, product.service_type, 0.0);
                    chunked.get(index).add(product);
                }
                break;
            case CUSTOMER:
                for (ProductInfo product : data) {
                    int index = getValueMap(CUSTOMER, product.customer, 0.0);
                    chunked.get(index).add(product);
                }
                break;
            case MONTHLY_FEE:
                for (ProductInfo product : data) {
                    int index = getValueMap(MONTHLY_FEE, null, product.monthly_fee);
                    chunked.get(index).add(product);
                }
                break;
            case ADVERTISEMENT_BUDGET:
                for (ProductInfo product : data) {
                    int index = getValueMap(ADVERTISEMENT_BUDGET, null, product.advertisement_budget);
                    chunked.get(index).add(product);
                }
                break;
            case SIZE:
                for (ProductInfo product : data) {
                    int index = getValueMap(SIZE, product.size, 0.0);
                    chunked.get(index).add(product);
                }
                break;
            case PROMOTION:
                for (ProductInfo product : data) {
                    int index = getValueMap(PROMOTION, product.promotion, 0.0);
                    chunked.get(index).add(product);
                }
                break;
            case INTEREST_RATE:
                for (ProductInfo product : data) {
                    int index = getValueMap(INTEREST_RATE, null, product.interest_rate);
                    chunked.get(index).add(product);
                }
                break;
            case PERIOD:
                for (ProductInfo product : data) {
                    int index = getValueMap(PERIOD, null, product.period);
                    chunked.get(index).add(product);
                }
                break;
        }
        return chunked;
    }

    /**
     * Method to map a particular attribute and its values in a pre-defined order. This order is utilized when dividing the data
     * into chunks of lists.
     * @param attribute for which the mapped value has to be obtained
     * @param stringValue - For non-numeric valued attributes, the String value of the instance is
     *                    passed, while the double parameter is set to 0.0.
     * @param doubleValue -  For numeric valued attributes, the actual value of the instance is
     *                    passed, while  the String parameter is set as null.

     * @return index of the list to which the value should be added
     */

    private int getValueMap(String attribute, String stringValue, double doubleValue) {
        switch (attribute) {
            case SERVICE_TYPE:
                switch (stringValue) {
                    case "Fund":
                        return 0;
                    case "Loan":
                        return 1;
                    case "Mortgage":
                        return 2;
                    case "CD":
                        return 3;
                    case "Bank_Account":
                        return 4;
                }
            case CUSTOMER:
                switch (stringValue) {
                    case "Student":
                        return 0;
                    case "Business":
                        return 1;
                    case "Professional":
                        return 2;
                    case "Doctor":
                        return 3;
                    case "Other":
                        return 4;
                }
            case SIZE:
                switch (stringValue) {
                    case "Small":
                        return 0;
                    case "Medium":
                        return 1;
                    case "Large":
                        return 2;
                }
            case PROMOTION:
                switch (stringValue) {
                    case "Full":
                        return 0;
                    case "Web":
                        return 1;
                    case "Web&Email":
                        return 2;
                    case "None":
                        return 3;
                }
            case MONTHLY_FEE:
            case ADVERTISEMENT_BUDGET:
            case INTEREST_RATE:
            case PERIOD:
                if (doubleValue >= 0 && doubleValue <= 0.25)
                    return 0;
                else if (doubleValue > 0.25 && doubleValue <= 0.5)
                    return 1;
                else if (doubleValue > 0.5 && doubleValue <= 0.75)
                    return 2;
                else
                    return 3;
            default:
                return -1;
        }
    }

    /**
     * Method to retrieve the mapped arrow label of a particular attribute value.
     * @param attribute for which the mapped value has to be obtained
     * @param stringValue - For non-numeric valued attributes, the String value of the instance is
     *                    passed, while the double parameter is set to 0.0.
     * @param doubleValue -  For numeric valued attributes, the actual value of the instance is
     *                    passed, while  the String parameter is set as null.

     * @return index of the list to which the value should be added
     */
    private double getArrowLabel(String attribute, String stringValue, double doubleValue) {
        switch (attribute) {
            case SERVICE_TYPE:
                switch (stringValue) {
                    case "Fund":
                        return 0;
                    case "Loan":
                        return 1;
                    case "Mortgage":
                        return 2;
                    case "CD":
                        return 3;
                    case "Bank_Account":
                        return 4;
                }
            case CUSTOMER:
                switch (stringValue) {
                    case "Student":
                        return 0;
                    case "Business":
                        return 1;
                    case "Professional":
                        return 2;
                    case "Doctor":
                        return 3;
                    case "Other":
                        return 4;
                }
            case SIZE:
                switch (stringValue) {
                    case "Small":
                        return 0;
                    case "Medium":
                        return 1;
                    case "Large":
                        return 2;
                }
            case PROMOTION:
                switch (stringValue) {
                    case "Full":
                        return 0;
                    case "Web":
                        return 1;
                    case "Web&Email":
                        return 2;
                    case "None":
                        return 3;
                }
            case MONTHLY_FEE:
            case ADVERTISEMENT_BUDGET:
            case INTEREST_RATE:
            case PERIOD:
                if (doubleValue >= 0 && doubleValue <= 0.25)
                    return 0.25;
                else if (doubleValue > 0.25 && doubleValue <= 0.5)
                    return 0.5;
                else if (doubleValue > 0.5 && doubleValue <= 0.75)
                    return 0.75;
                else
                    return 1.0;
            default:
                return -1;
        }
    }

    /**
     * Method to construct the decision tree.
     * @param data - data set that is considered on every recursive call of the tree
     * @param remainingAttributes - list of attributes that are considered while constructing the tree.
     *                            Note: When an attribute is added as a node, it is removed from the list of remainingAttributes
     * @return - ode of the decision tree which has the output label
     */

    public TreeNode train(List<ProductInfo> data, List<String> remainingAttributes) {
        if (data.size() == 0) {
            return null;
        }

        String opLabel = checkPureNode(data, false); // Check if a node is a pure node. If it is a pure node,
                                                             // opLabel will not be null. Majority is a flag that is used for
                                                             // the scenario when we have exhausted all the attributes.
                                                             // Hence, we keep a track of the attribute with highest gain on each level.
        if (opLabel != null) {
            TreeNode node = new TreeNode("");
            node.isLeaf = true;
            node.outputLabel = opLabel;
            return node;
        }

        String splittingAttribute = getAttributeWithHighestGain(data, remainingAttributes); // Get attribute with highest gain
        String majorityLabel = checkPureNode(data, true);

        if (splittingAttribute == null) {
            if (majorityLabel != null) {
                TreeNode node = new TreeNode("");
                node.isLeaf = true;
                node.outputLabel = majorityLabel;
                return node;
            }
        }

        TreeNode node = new TreeNode(splittingAttribute);
        node.outputLabel = majorityLabel;
        remainingAttributes.remove(splittingAttribute);

        List<List<ProductInfo>> chunkedData = getFilteredData(data, splittingAttribute);
        List<Double> possibleValues = getPossibleValues(splittingAttribute);

        for (int i = 0; i < possibleValues.size(); i++) {
            TreeNode child = train(chunkedData.get(i), remainingAttributes);
            node.children.put(possibleValues.get(i), child);
        }

        return node;
    }

    /**
     * Method to predict the class label for a given instance from the data set.
     * @param product instance for which the prediction should be done
     * @param node of the tree that is recursively utilized during traversal
     * @return
     */
    public String predict(ProductInfo product, TreeNode node) {
        if (node == null) {
            return null;
        }

        if (node.isLeaf) {
            return node.outputLabel;
        }

        double key;

        switch (node.attribute) {
            case SERVICE_TYPE:
                key = getArrowLabel(SERVICE_TYPE, product.service_type, 0);
                String op = predict(product, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case CUSTOMER:
                key = getArrowLabel(CUSTOMER, product.customer, 0);
                op = predict(product, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case MONTHLY_FEE:
                key = getArrowLabel(MONTHLY_FEE, null, product.monthly_fee);
                op = predict(product, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case ADVERTISEMENT_BUDGET:
                key = getArrowLabel(ADVERTISEMENT_BUDGET, null, product.advertisement_budget);
                op = predict(product, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case SIZE:
                key = getArrowLabel(SIZE, product.size, 0);
                op = predict(product, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case PROMOTION:
                key = getArrowLabel(PROMOTION, product.promotion, 0);
                op = predict(product, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }

            case INTEREST_RATE:
                key = getArrowLabel(INTEREST_RATE, null, product.interest_rate);
                op = predict(product, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }

            case PERIOD:
                key = getArrowLabel(PERIOD, null, product.period);
                op = predict(product, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }

            default:
                return node.outputLabel;
        }
    }

    /**
     * Main method of the class where arguments of the file paths are specified.
     * Note : First argument is for the input file path of the train data set.
     *      : Second argument is for the input file path of the test data set.
     * @param args - array of arguments (of the file paths)
     */

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Please enter the file paths for train and test data sets.");
            return;
        }
        String trainDataPath = args[0];
        String testDataPath = args[1];
        ID3PartB id3 = new ID3PartB();

        List<ProductInfo> trainingData = id3.readData(trainDataPath);
        TreeNode root = id3.train(trainingData, attributes);

        System.out.println("Training successfully completed");

        List<ProductInfo> validationList = new ArrayList<>(trainingData);

        int folds = 5;
        int validationSize = trainingData.size() / folds, foldCount = 1;
        double sum = 0;

        while (foldCount <= folds) {
            Collections.shuffle(validationList);
            List<ProductInfo> testValidationSet = validationList.subList(0, validationSize);
            int count = 0;

            for (ProductInfo product : testValidationSet) {
                String validateLabel = id3.predict(product, root);
                if (validateLabel.equals(product.label))
                    count++;
            }

            double accuracy = (double) count / testValidationSet.size() * 100;
            System.out.println("Accuracy for fold " + foldCount + " : " + String.format("%.2f", (double) accuracy));
            sum += accuracy;
            foldCount++;
        }

        System.out.println("Cross-validation accuracy: " + String.format("%.2f", (double) sum / folds) + "\n");

        List<ProductInfo> testData = id3.readData(testDataPath);

        System.out.println("Successfully loaded test data");
        System.out.println("Output class labels for the test set:");

        for (int i = 0; i < testData.size(); i++) {
            ProductInfo product = testData.get(i);
            System.out.println(id3.predict(product, root));
        }
    }

    /**
     * Method to read instances from the data set.
     * @param filePath - path location from where the data is read
     * @return - list of instances that are read from the specified file path.
     */
    private List<ProductInfo> readData(String filePath) {
        List<ProductInfo> instanceList = new ArrayList<>();
        BufferedReader bufferedReader;
        String inputFile = filePath;
        try {
            bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(inputFile),
                    StandardCharsets.UTF_8));
            String line;
            line = bufferedReader.readLine();
            line = line.toLowerCase(); // to maintain consistency of header names in CSV file
            attributes = new LinkedList<String>(Arrays.asList(line.split(",")));
            attributes.remove(attributes.size() - 1);

            while ((line = bufferedReader.readLine()) != null) {
                String[] temp = line.split(",");
                if (temp.length == 9) {
                    try {
                        ProductInfo testInfo = new ProductInfo(temp[0], temp[1],
                                Double.parseDouble(temp[2]), Double.parseDouble(temp[3]), temp[4], temp[5],
                                Double.parseDouble(temp[6]), Double.parseDouble(temp[7]), temp[8]);
                        instanceList.add(testInfo);
                    } catch (NumberFormatException e) {
                        e.printStackTrace();
                    }
                }
            }

            return instanceList;

        } catch (IOException e) {
            e.printStackTrace();
        }

        return instanceList;
    }
}
