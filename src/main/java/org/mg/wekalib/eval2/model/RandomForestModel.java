package org.mg.wekalib.eval2.model;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;

public class RandomForestModel extends AbstractModel
{
	//	private static final long serialVersionUID = 1L;

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

	@Override
	public String getName()
	{
		return "RandomForest " + numTrees;
	}

	public void setNumTrees(int numTrees)
	{
		this.numTrees = numTrees;
	}

	@Override
	protected void cloneParams(Model clonedModel)
	{
		((RandomForestModel) clonedModel).setNumTrees(numTrees);
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return numTrees == 100 ? "" : ("trees:" + numTrees);
	}

	@Override
	public String getAlgorithmShortName()
	{
		return "RF";
	}

}
