package org.mg.wekalib.eval2.model;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class NaiveBayesModel extends AbstractModel
{
	//	private static final long serialVersionUID = 1L;

	@Override
	public String getParamKey()
	{
		return "";
	}

	@Override
	public Classifier getWekaClassifer()
	{
		return new NaiveBayes();
	}

	@Override
	protected void cloneParams(Model clonedModel)
	{
	}

	@Override
	public String getAlgorithmShortName()
	{
		return "NB";
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return "";
	}

	@Override
	public boolean isFast()
	{
		return true;
	}
}
