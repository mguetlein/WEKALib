package org.mg.wekalib.attribute_ranking;

import java.io.Serializable;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

import org.mg.javalib.util.ArrayUtil;

@XmlRootElement
public class PredictionAttribute implements Serializable
{
	private static final long serialVersionUID = 3L;

	private int attribute;
	private int alternativePredictionIdx;
	private double alternativeDistributionForInstance[];
	private double diffToOrigProp;

	public PredictionAttribute()
	{
	}

	public PredictionAttribute(int attribute, double[] alternativeDistributionForInstance,
			double diffToOrigProp)
	{
		this.attribute = attribute;
		this.alternativeDistributionForInstance = alternativeDistributionForInstance;
		this.alternativePredictionIdx = ArrayUtil.getMaxIndex(alternativeDistributionForInstance);
		this.diffToOrigProp = diffToOrigProp;
	}

	@XmlAttribute
	public int getAttribute()
	{
		return attribute;
	}

	@XmlAttribute
	public double getDiffToOrigProp()
	{
		return diffToOrigProp;
	}

	@XmlAttribute
	public int getAlternativePredictionIdx()
	{
		return alternativePredictionIdx;
	}

	@XmlAttribute
	public double[] getAlternativeDistributionForInstance()
	{
		return alternativeDistributionForInstance;
	}

	public String toString()
	{
		StringBuffer b = new StringBuffer();
		b.append(PredictionAttribute.class.getSimpleName() + "\n");
		b.append("attribute-index: " + attribute + "\n");
		b.append("alternate-prediction-idx: " + alternativePredictionIdx + "\n");
		b.append("alternate-prediction: " + ArrayUtil.toString(alternativeDistributionForInstance)
				+ "\n");
		b.append("diff-to-orig: " + diffToOrigProp + "\n");
		return b.toString();
	}
}
