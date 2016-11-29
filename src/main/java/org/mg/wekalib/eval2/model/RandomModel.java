package org.mg.wekalib.eval2.model;

import org.mg.wekalib.classifier.RandomClassifier;

import weka.classifiers.Classifier;

public class RandomModel extends AbstractModel
{
	@Override
	public String getAlgorithmShortName()
	{
		return "Random";
	}

	@Override
	public String getName()
	{
		return "RandomModel";
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return "";
	}

	@Override
	public Classifier getWekaClassifer()
	{
		return new RandomClassifier();
	}

	@Override
	public boolean isFast()
	{
		return true;
	}

	@Override
	protected String getParamKey()
	{
		return "";
	}

	@Override
	protected void cloneParams(Model clonedModel)
	{
	}

}
