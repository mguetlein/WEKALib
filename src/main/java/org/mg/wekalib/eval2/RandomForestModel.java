package org.mg.wekalib.eval2;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;

public class RandomForestModel extends AbstractModel
{
	private static final long serialVersionUID = 1L;

	int numTrees = 100;

	@Override
	public Classifier getWekaClassifer()
	{
		RandomForest rf = new RandomForest();
		rf.setNumTrees(numTrees);
		return rf;
	}

	@Override
	public String getParamKey()
	{
		return Integer.toString(numTrees);
	}

}
