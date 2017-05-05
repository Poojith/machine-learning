# Machine Learning 
An ID3 Machine Learning algorithm for the classification of customers and products.

  * Supports nominal, discrete, and continuous attributes.
  * Does not make use of any external library or implementation.
  * Assumes the given data set is in the form of a CSV file.
  * Assumes all the numeric values in the data set are normalized.
  * Assumes the features/attribute names in the data set to not change.
  * Assumes constant ordering of features/attributes in the data set.
  * Performs pruning to reduce the size of the tree and to avoid overfitting.

Training phase - building the decision tree:
-------------------------------------------
* Construction of the tree begins with the root node - attribute with the highest information gain.
* On every iteration of the algorithm, the unused attributes of the set are considered and information gain is computed for all those attributes.
* Attributes are split by considering information gain on every level of the tree traversal for a filtered set of data. Data is filtered based on the arrow label(edge) in the tree.
* Majority class labels are considered in scenarios when all attributes are exhausted and the traversal of the tree has not reached a leaf node. 


Testing phase - prediction:
---------------------------
The trained decision tree is used to classify labels by traversing down the tree using the values of a given instance. The leaf node’s label (or, in some cases, the majority label) gives the output class.


Running the decision tree:
--------------------------

* There are two separate .java source files for parts A (ID3.java) and B (ID3PartB.java). 
* These .java files can be run either on the terminal/command prompt or on an IDE.
* Each of these classes have a main method. Therefore, they can be run separately without relying on either of them.
* To run the implementation, please provide the file paths for the training and test data sets.
* The file paths are read into the program in the form of command line arguments.
* The first argument takes the path for the train data set.
* The second argument takes into consideration the test data set.


Output of the decision tree:
----------------------------
* Accuracy of each fold of execution.
* Predicted output class labels for the test set.
