package org.mg.wekalib.eval2.model;

import java.util.List;

import org.mg.wekalib.appdomain.DistanceBasedADClassifier;
import org.mg.wekalib.appdomain.PCAEuclideanADModel;
import org.mg.wekalib.appdomain.TanimotoCentroidADModel;
import org.mg.wekalib.appdomain.TanimotoNNADModel;

import weka.classifiers.Classifier;

public class DistanceBasedAppDomainPredictionModel extends AbstractModel
{
	public static enum Type
	{
		PCAEuclidean, TanimotoNN, TanimotoCentroidV2
	}

	Type type;

	public void setType(Type type)
	{
		this.type = type;
	}

	@Override
	public String getAlgorithmShortName()
	{
		return "Dist";
	}

	List<String> trainingSmiles;
	List<String> testSmiles;

	public void setTrainingSmiles(List<String> trainingSmiles)
	{
		this.trainingSmiles = trainingSmiles;
	}

	public void setTestSmiles(List<String> testSmiles)
	{
		this.testSmiles = testSmiles;
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return "type: " + type;
	}

	@Override
	public Classifier getWekaClassifer()
	{
		switch (type)
		{
			case PCAEuclidean:
				return new DistanceBasedADClassifier(new PCAEuclideanADModel(3, true, 0.95), 1);
			case TanimotoNN:
				return new DistanceBasedADClassifier(new TanimotoNNADModel(3, true, 0.95), 1);
			case TanimotoCentroidV2:
				return new DistanceBasedADClassifier(new TanimotoCentroidADModel(0.95, 0.05), 1);
			default:
				throw new IllegalArgumentException();
		}
	}

	@Override
	public boolean isFast()
	{
		return true;
	}

	@Override
	protected String getParamKey()
	{
		return type + "";
	}

	@Override
	protected void cloneParams(Model clonedModel)
	{
		((DistanceBasedAppDomainPredictionModel) clonedModel).setType(type);
	}

}
