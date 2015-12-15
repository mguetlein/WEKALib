package org.mg.wekalib.attribute_ranking;

import java.util.Set;

import weka.core.Instance;

public interface AttributeProvidingClassifier
{
	public String getName();

	public Set<Integer> getAttributesEmployedForPrediction(Instance instance);
}
