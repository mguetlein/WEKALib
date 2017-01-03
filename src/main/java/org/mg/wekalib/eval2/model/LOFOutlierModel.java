package org.mg.wekalib.eval2.model;

import org.mg.wekalib.distance.TanimotoDistance;

import weka.classifiers.Classifier;
import weka.classifiers.misc.LOF;
import weka.core.DistanceFunction;
import weka.core.neighboursearch.LinearNNSearch;

public class LOFOutlierModel extends AbstractModel
{
	String distanceFunctionClassName = TanimotoDistance.class.getName();
	//String distanceFunctionClassName = EuclideanDistance.class.getName();

	int lowerBoundK = 3;
	int upperBoundK = 4;

	public void setDistanceFunctionClassName(String distanceFunctionClassName)
	{
		this.distanceFunctionClassName = distanceFunctionClassName;
	}

	public void setLowerBoundK(int lowerBoundK)
	{
		this.lowerBoundK = lowerBoundK;
	}

	public void setUpperBoundK(int upperBoundK)
	{
		this.upperBoundK = upperBoundK;
	}

	@Override
	public String getParamKey()
	{
		return distanceFunctionClassName + "#" + lowerBoundK + "#" + upperBoundK;
	}

	@Override
	public Classifier getWekaClassifer()
	{
		try
		{
			LOF l = new LOF();
			LinearNNSearch s = new LinearNNSearch();
			DistanceFunction df = (DistanceFunction) Class.forName(distanceFunctionClassName)
					.newInstance();
			s.setDistanceFunction(df);
			l.setNNSearch(s);
			l.setMinPointsLowerBound(lowerBoundK + "");
			l.setMinPointsUpperBound(upperBoundK + "");
			return l;
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	@Override
	protected void cloneParams(Model clonedModel)
	{
		((LOFOutlierModel) clonedModel).setDistanceFunctionClassName(distanceFunctionClassName);
		((LOFOutlierModel) clonedModel).setLowerBoundK(lowerBoundK);
		((LOFOutlierModel) clonedModel).setUpperBoundK(upperBoundK);
	}

	@Override
	public String getAlgorithmShortName()
	{
		return "LOF-"
				+ distanceFunctionClassName
						.substring(distanceFunctionClassName.lastIndexOf('.') + 1)
				+ "-k:" + lowerBoundK + "-" + upperBoundK;
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return distanceFunctionClassName + " - k:" + lowerBoundK + "-" + upperBoundK;
	}

	@Override
	public boolean isFast()
	{
		return false;
	}
}
