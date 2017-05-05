/**
 * Decision tree implementation for the classification of customers.
 * Team 2 - MSIT eBusiness Technology, Carnegie Mellon University
 */

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.Map.Entry;

/**
 * Class that represents an instance of a customer from the data set.
 */

class CustomerInfo {
    String type;
    String lifeStyle;
    double vacation;
    double eCredit;
    double salary;
    double property;
    String label;

    public CustomerInfo(String type, String lifeStyle, double vacation, double eCredit, double salary,
                        double property, String label) {
        this.type = type;
        this.lifeStyle = lifeStyle;
        this.vacation = vacation;
        this.eCredit = eCredit;
        this.salary = salary;
        this.property = property;
        this.label = label;
    }
}

/**
 * Class that represents the structure of a node in the decision tree.
 */

class Node {
    Map<Double, Node> children = new HashMap<>();
    String attribute;
    boolean isLeaf;
    String outputLabel;

    public Node(String attribute) {
        this.attribute = attribute;
    }
}


/**
 * Class for the ID3 Decision Tree (Part A).
 */

public class ID3 {

    private static final String TYPE = "type";
    private static final String LIFESTYLE = "lifestyle";
    private static final String VACATION = "vacation";
    private static final String ECREDIT = "ecredit";
    private static final String SALARY = "salary";
    private static final String PROPERTY = "property";

    private static List<String> attributes = new ArrayList<>(); // list of attributes from the data set

    /**
     * Method to calculate the entropy for a given set of data.
     *
     * @param tempInstanceList for which the entropy has to be calculated
     * @return value of entropy in double
     */
    private static double calculateEntropy(List<CustomerInfo> tempInstanceList) {
        int numInstances = tempInstanceList.size();

        if (numInstances == 0) {
            return 9999;             // Return an arbitrary high entropy value when number of instances is zero.

        }

        double[] labelCounts = new double[5];
        for (CustomerInfo customerInfo : tempInstanceList) {
            // Calculate the number of occurrences of each label
            switch (customerInfo.label) {
                case "C1":
                    labelCounts[0]++;
                    break;
                case "C2":
                    labelCounts[1]++;
                    break;
                case "C3":
                    labelCounts[2]++;
                    break;
                case "C4":
                    labelCounts[3]++;
                    break;
                case "C5":
                    labelCounts[4]++;
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
     *
     * @param attribute for which the values have to be obtained.
     * @return List of values for the attribute. For non-numeric values, each double value is mapped to its
     * corresponding String literal.
     */

    public List<Double> getPossibleValues(String attribute) {
        List<Double> list = new ArrayList<>();
        if (attribute.equals(LIFESTYLE)) {
            list.add(0.0);
            list.add(1.0);
            list.add(2.0);
            list.add(3.0);
        } else if (attribute.equals(TYPE)) {
            list.add(0.0);
            list.add(1.0);
            list.add(2.0);
            list.add(3.0);
            list.add(4.0);
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
     *
     * @param data                for which the information gain is computed
     * @param remainingAttributes - list of attributes for which the information gain is computed.
     *                            Note : Whenever an attribute is chosen as a node of the tree, it is removed from the list.
     * @return the attribute with the highest information gain
     */

    public String getAttributeWithHighestGain(List<CustomerInfo> data, List<String> remainingAttributes) {
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
     *
     * @param data     among which the label is determined
     * @param majority - boolean flag that is used when all attributes are exhausted. (Majority label is returned)
     * @return
     */

    public String checkPureNode(List<CustomerInfo> data, boolean majority) {

        Map<String, Integer> labelCounts = new HashMap<>();

        for (CustomerInfo customer : data) {
            if (!labelCounts.containsKey(customer.label)) {
                labelCounts.put(customer.label, 1);
            }
            labelCounts.put(customer.label, labelCounts.get(customer.label) + 1);
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
     *
     * @param data      - superset of data among which the information gain must be computed
     * @param attribute for which the information gain is computed
     * @return the information gain value for the attribute in double
     */

    public Double computeInformationGain(List<CustomerInfo> data, String attribute) {
        double entropy = calculateEntropy(data);

        List<List<CustomerInfo>> chunkedData = getFilteredData(data, attribute);

        List<Double> possibleValues = getPossibleValues(attribute);
        double S = data.size();

        for (int i = 0; i < possibleValues.size(); i++) {
            double S_v = chunkedData.get(i).size();
            entropy -= ((S_v / S) * calculateEntropy(chunkedData.get(i)));
        }

        return entropy;
    }

    /**
     * Method to extract the set of data for a given an attribute
     *
     * @param data      - the set in which data is filtered for a given attribute
     * @param attribute for which the data is filtered
     * @return A list that contains lists. Each of the contained lists are chunks of data for the attribute.
     * Note : For non-numeric values, there are four chunks, equally divided from 0.0 to 1.0.
     */

    private List<List<CustomerInfo>> getFilteredData(List<CustomerInfo> data, String attribute) {
        List<List<CustomerInfo>> chunked = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            chunked.add(new ArrayList<>());
        }
        if (attribute.equals(TYPE)) {
            chunked.add(new ArrayList<>());
        }

        /**
         * Get the mapped value of the index(for the list) to which the chunk of data is added.
         * The arguments to getValueMap correspond to whether the attribute has numeric or non-numeric values.
         * For numeric valued attributes, the actual value of the instance is passed with the String parameter set as null.
         * For non-numeric valued attributes, the String value of the instance is passed with the double parameter set to 0.0.
         */

        switch (attribute) {
            case LIFESTYLE:
                for (CustomerInfo customer : data) {
                    int index = getValueMap(LIFESTYLE, customer.lifeStyle, 0.0);
                    chunked.get(index).add(customer);
                }
                break;
            case TYPE:
                for (CustomerInfo customer : data) {
                    int index = getValueMap(TYPE, customer.type, 0.0);
                    chunked.get(index).add(customer);
                }
                break;
            case ECREDIT:
                for (CustomerInfo customer : data) {
                    int index = getValueMap(ECREDIT, null, customer.eCredit);
                    chunked.get(index).add(customer);
                }
                break;
            case SALARY:
                for (CustomerInfo customer : data) {
                    int index = getValueMap(SALARY, null, customer.salary);
                    chunked.get(index).add(customer);
                }
                break;
            case PROPERTY:
                for (CustomerInfo customer : data) {
                    int index = getValueMap(PROPERTY, null, customer.property);
                    chunked.get(index).add(customer);
                }
                break;
            case VACATION:
                for (CustomerInfo customer : data) {
                    int index = getValueMap(VACATION, null, customer.vacation);
                    chunked.get(index).add(customer);
                }
                break;
        }
        return chunked;
    }


    /**
     * Method to map a particular attribute and its values in a pre-defined order. This order is utilized when dividing the data
     * into chunks of lists.
     *
     * @param attribute   for which the mapped value has to be obtained
     * @param stringValue - For non-numeric valued attributes, the String value of the instance is
     *                    passed, while the double parameter is set to 0.0.
     * @param doubleValue -  For numeric valued attributes, the actual value of the instance is
     *                    passed, while  the String parameter is set as null.
     * @return index of the list to which the value should be added
     */

    private int getValueMap(String attribute, String stringValue, double doubleValue) {
        switch (attribute) {
            case LIFESTYLE:
                switch (stringValue) {
                    case "spend>saving":
                        return 0;
                    case "spend<saving":
                        return 1;
                    case "spend>>saving":
                        return 2;
                    case "spend<<saving":
                        return 3;
                }
            case TYPE:
                switch (stringValue) {
                    case "student":
                        return 0;
                    case "engineer":
                        return 1;
                    case "librarian":
                        return 2;
                    case "professor":
                        return 3;
                    case "doctor":
                        return 4;
                }
            case VACATION:
            case SALARY:
            case PROPERTY:
            case ECREDIT:
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
     *
     * @param attribute   for which the mapped value has to be obtained
     * @param stringValue - For non-numeric valued attributes, the String value of the instance is
     *                    passed, while the double parameter is set to 0.0.
     * @param doubleValue -  For numeric valued attributes, the actual value of the instance is
     *                    passed, while  the String parameter is set as null.
     * @return index of the list to which the value should be added
     */

    private double getArrowLabel(String attribute, String stringValue, double doubleValue) {
        switch (attribute) {
            case LIFESTYLE:
                switch (stringValue) {
                    case "spend>saving":
                        return 0;
                    case "spend<saving":
                        return 1;
                    case "spend>>saving":
                        return 2;
                    case "spend<<saving":
                        return 3;
                }
            case TYPE:
                switch (stringValue) {
                    case "student":
                        return 0;
                    case "engineer":
                        return 1;
                    case "librarian":
                        return 2;
                    case "professor":
                        return 3;
                    case "doctor":
                        return 4;
                }
            case VACATION:
            case SALARY:
            case PROPERTY:
            case ECREDIT:
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
     *
     * @param data                - data set that is considered on every recursive call of the tree
     * @param remainingAttributes - list of attributes that are considered while constructing the tree.
     *                            Note: When an attribute is added as a node, it is removed from the list of remainingAttributes
     * @return - ode of the decision tree which has the output label
     */

    public Node train(List<CustomerInfo> data, List<String> remainingAttributes) {
        if (data.size() == 0) {
            return null;
        }

        // Check if a node is a pure node. If it is a pure node,
        // opLabel will not be null. Majority is a flag that is used for
        // the scenario when we have exhausted all the attributes.
        // Hence, we keep a track of the attribute with highest gain on each level.
        String opLabel = checkPureNode(data, false);
        if (opLabel != null) {
            Node node = new Node("");
            node.isLeaf = true;
            node.outputLabel = opLabel;
            return node;
        }

        String splittingAttribute = getAttributeWithHighestGain(data, remainingAttributes); // Get attribute with highest gain
        String majorityLabel = checkPureNode(data, true);

        if (splittingAttribute == null) {
            if (majorityLabel != null) {
                Node node = new Node("");
                node.isLeaf = true;
                node.outputLabel = majorityLabel;
                return node;
            }
        }

        Node node = new Node(splittingAttribute);
        node.outputLabel = majorityLabel;
        remainingAttributes.remove(splittingAttribute);

        List<List<CustomerInfo>> chunkedData = getFilteredData(data, splittingAttribute);
        List<Double> possibleValues = getPossibleValues(splittingAttribute);

        for (int i = 0; i < possibleValues.size(); i++) {
            Node child = train(chunkedData.get(i), remainingAttributes);
            node.children.put(possibleValues.get(i), child);
        }

        return node;
    }

    /**
     * Method to predict the class label for a given instance from the data set.
     * @param customer instance for which the prediction should be done
     * @param node of the tree that is recursively utilized during traversal
     * @return
     */

    public String predict(CustomerInfo customer, Node node) {
        if (node == null) {
            return null;
        }

        if (node.isLeaf) {
            return node.outputLabel;
        }

        double key;

        switch (node.attribute) {
            case LIFESTYLE:
                key = getArrowLabel(LIFESTYLE, customer.lifeStyle, 0);
                String op = predict(customer, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case TYPE:
                key = getArrowLabel(TYPE, customer.type, 0);
                op = predict(customer, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case VACATION:
                key = getArrowLabel(VACATION, null, customer.vacation);
                op = predict(customer, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case SALARY:
                key = getArrowLabel(SALARY, null, customer.salary);
                op = predict(customer, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case PROPERTY:
                key = getArrowLabel(PROPERTY, null, customer.property);
                op = predict(customer, node.children.get(key));
                if (op == null) {
                    return node.outputLabel;
                } else {
                    return op;
                }
            case ECREDIT:
                key = getArrowLabel(ECREDIT, null, customer.eCredit);
                op = predict(customer, node.children.get(key));
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
     *        Second argument is for the input file path of the test data set.
     * @param args - array of arguments (of the file paths)
     */

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Please enter the file paths for train and test data sets.");
            return;
        }

        String trainDataPath = args[0];
        String testDataPath = args[1];

        ID3 id3 = new ID3();
        List<CustomerInfo> trainingData = id3.readData(trainDataPath);
        Node root = id3.train(trainingData, attributes);

        System.out.println("Training successfully completed");

        List<CustomerInfo> validationList = new ArrayList<>(trainingData);

        int folds = 10;
        int validationSize = trainingData.size() / folds, foldCount = 1;
        double sum = 0;

        while (foldCount <= folds) {
            Collections.shuffle(validationList);
            List<CustomerInfo> testValidationSet = validationList.subList(0, validationSize);
            int count = 0;

            for (CustomerInfo customer : testValidationSet) {
                String validateLabel = id3.predict(customer, root);
                if (validateLabel.equals(customer.label))
                    count++;
            }

            double accuracy = (double) count / testValidationSet.size() * 100;
            System.out.println("Accuracy for fold " + foldCount + " : " + String.format("%.2f", (double) accuracy));
            sum += accuracy;
            foldCount++;
        }

        System.out.println("Cross-validation accuracy: " + String.format("%.2f", sum / folds) + "\n");

        List<CustomerInfo> testData = id3.readData(testDataPath);

        System.out.println("Successfully loaded test data");
        System.out.println("Output class labels for the test set:");

        for (int i = 0; i < testData.size(); i++) {
            CustomerInfo customer = testData.get(i);
            System.out.println(id3.predict(customer, root));
        }
    }

    /**
     * Method to read instances from the data set.
     * @param filePath - path location from where the data is read
     * @return - list of instances that are read from the specified file path.
     */
    private List<CustomerInfo> readData(String filePath) {
        List<CustomerInfo> instanceList = new ArrayList<>();
        BufferedReader bufferedReader;
        String inputFile = filePath;
        try {
            bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(inputFile),
                    StandardCharsets.UTF_8));
            String line;
            line = bufferedReader.readLine();
            line = line.toLowerCase(); // to maintain consistency of header names in CSV file
            attributes = new LinkedList<>(Arrays.asList(line.split(",")));
            attributes.remove(attributes.size() - 1);

            while ((line = bufferedReader.readLine()) != null) {
                String[] temp = line.split(",");
                if (temp.length == 7) {
                    try {
                        CustomerInfo customerInfo = new CustomerInfo(temp[0], temp[1],
                                Double.parseDouble(temp[2]), Double.parseDouble(temp[3]),
                                Double.parseDouble(temp[4]), Double.parseDouble(temp[5]),
                                temp[6]);
                        instanceList.add(customerInfo);
                    } catch (NumberFormatException e) {
                        System.out.println("Error in parsing double value. ");
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
