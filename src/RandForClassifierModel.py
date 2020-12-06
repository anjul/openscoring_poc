from sklearn import datasets, tree
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from com.mycompany import Aggregator, PowerFunction

if __name__ == '__main__':
    iris = datasets.load_iris()
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(iris.data, iris.target)

    default_mapper = DataFrameMapper([
		(["Sepal.Length", "Petal.Length"], [ContinuousDomain(), Aggregator(function = "mean")]),
		(["Sepal.Width", "Petal.Width"], [ContinuousDomain(), PowerFunction(power = 2)])
	])
    sklearn2pmml(classifier, default_mapper,"IrisClassificationTree.pmml")

    iris_pipeline = PMMLPipeline([
        ("mapper", default_mapper),
        ("classifier", LogisticRegression())
    ])

    sklearn2pmml(iris_pipeline, "Iris.pmml")