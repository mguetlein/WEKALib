package org.mg.wekalib.attribute_ranking;

import java.io.Serializable;

import org.mg.javalib.util.ArrayUtil;

public class PredictionAttributeImpl implements PredictionAttribute, Serializable
{
	private static final long serialVersionUID = 4L;

	private int attribute;
	private int alternativePredictionIdx;
	private double alternativeDistributionForInstance[];
	private double diffToOrigProp;

	public PredictionAttributeImpl()
	{
	}

	public PredictionAttributeImpl(int attribute, double[] alternativeDistributionForInstance,
			double diffToOrigProp)
	{
		this.attribute = attribute;
		this.alternativeDistributionForInstance = alternativeDistributionForInstance;
		this.alternativePredictionIdx = ArrayUtil.getMaxIndex(alternativeDistributionForInstance);
		this.diffToOrigProp = diffToOrigProp;
	}

	@Override
	public int getAttribute()
	{
		return attribute;
	}

	@Override
	public double getDiffToOrigProp()
	{
		return diffToOrigProp;
	}

	@Override
	public int getAlternativePredictionIdx()
	{
		return alternativePredictionIdx;
	}

	@Override
	public double[] getAlternativeDistributionForInstance()
	{
		return alternativeDistributionForInstance;
	}

	@Override
	public String toString()
	{
		StringBuffer b = new StringBuffer();
		b.append(PredictionAttributeImpl.class.getSimpleName() + "\n");
		b.append("attribute-index: " + attribute + "\n");
		b.append("alternate-prediction-idx: " + alternativePredictionIdx + "\n");
		b.append("alternate-prediction: " + ArrayUtil.toString(alternativeDistributionForInstance)
				+ "\n");
		b.append("diff-to-orig: " + diffToOrigProp + "\n");
		return b.toString();
	}
}
