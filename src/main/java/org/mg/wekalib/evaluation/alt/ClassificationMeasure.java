package org.mg.wekalib.evaluation.alt;

import weka.classifiers.evaluation.Evaluation;

public enum ClassificationMeasure implements Measure
{
	AUC, AUP, ER1, ER5, Accuracy;

	public double getValue(Evaluation eval)
	{
		return getValue(eval, 1);
	}

	public double getValue(Evaluation eval, int classIndex)
	{
		switch (this)
		{
			case AUC:
				return eval.areaUnderROC(classIndex);
			case AUP:
				return eval.areaUnderPRC(classIndex);
			case ER1:
				return ((ExtendedEvaluation) eval).enrichmentFactor(classIndex, 0.01);
			case ER5:
				return ((ExtendedEvaluation) eval).enrichmentFactor(classIndex, 0.05);
			case Accuracy:
				return eval.pctCorrect() / 100.0;
		}
		throw new IllegalStateException();
	}

	@Override
	public boolean lowerIsBetter()
	{
		return false;
	}
}
