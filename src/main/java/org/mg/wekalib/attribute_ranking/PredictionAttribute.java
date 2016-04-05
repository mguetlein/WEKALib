package org.mg.wekalib.attribute_ranking;

public interface PredictionAttribute
{
	public int getAttribute();

	public double getDiffToOrigProp();

	public int getAlternativePredictionIdx();

	public double[] getAlternativeDistributionForInstance();
}
