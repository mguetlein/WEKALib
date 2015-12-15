package org.mg.wekalib.evaluation.alt;

import weka.classifiers.evaluation.Evaluation;

public interface Measure
{
	public boolean lowerIsBetter();

	public double getValue(Evaluation eval) throws Exception;

	public double getValue(Evaluation eval, int classIndex) throws Exception;
}
