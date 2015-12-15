package org.mg.wekalib.evaluation.alt;

import weka.classifiers.evaluation.Evaluation;

public enum RegressionMeasure implements Measure
{
	Correlation, RMSE;

	@Override
	public double getValue(Evaluation eval) throws Exception
	{
		switch (this)
		{
			case Correlation:
				return eval.correlationCoefficient();
			case RMSE:
				return eval.rootMeanSquaredError();
		}
		throw new IllegalStateException();
	}

	@Override
	public boolean lowerIsBetter()
	{
		switch (this)
		{
			case Correlation:
				return false;
			case RMSE:
				return true;
		}
		throw new IllegalStateException();
	}

	@Override
	public double getValue(Evaluation eval, int classIndex) throws Exception
	{
		return getValue(eval);
	}
}
