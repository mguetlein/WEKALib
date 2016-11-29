package org.mg.wekalib.test;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.instance.RemovePercentage;

public class FilteredClassifierTest
{
	public static void main(String[] args) throws Exception
	{
		String data = "vote";
		Instances inst = new Instances(new FileReader(
				System.getProperty("user.home") + "/data/weka/nominal/" + data + ".arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		inst.randomize(new Random(1));

		Classifier classifier = new RandomForest();

		FilteredClassifier filteredClassifier = new FilteredClassifier();
		filteredClassifier.setClassifier(classifier);
		RemovePercentage filter = new RemovePercentage();
		filter.setPercentage(30);
		//		System.out.println(inst.numInstances());
		//		filter.setInputFormat(inst);
		//		Instances instX = Filter.useFilter(inst, filter);
		//		System.out.println(instX.numInstances());
		filteredClassifier.setFilter(filter);
		classifier = filteredClassifier;

		int trainSize = (int) Math.round(inst.numInstances() * 0.95);
		int testSize = inst.numInstances() - trainSize;
		Instances train = new Instances(inst, 0, trainSize);
		Instances test = new Instances(inst, trainSize, testSize);

		System.out.println(train.numInstances() + "/" + test.numInstances());
		classifier.buildClassifier(train);

		for (Instance testI : test)
		{
			System.out.println(classifier.classifyInstance(testI));
		}

		Evaluation eval = new Evaluation(inst);
		eval.evaluateModel(classifier, test, (Object) null);

		System.out.println(eval.numInstances());
		System.out.println(eval.toMatrixString());
		System.out.println(eval.pctCorrect());
		System.out.println(eval.areaUnderROC(0));

	}
}
