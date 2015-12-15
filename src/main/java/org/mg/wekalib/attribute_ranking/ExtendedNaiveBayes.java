package org.mg.wekalib.attribute_ranking;

import java.util.Set;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;

public class ExtendedNaiveBayes extends NaiveBayes implements AttributeProvidingClassifier
{
	private static final long serialVersionUID = 1L;

	@Override
	public Set<Integer> getAttributesEmployedForPrediction(Instance instance)
	{
		return PredictionAttributeComputation.allAttributes(instance);
	}

	public String getName()
	{
		return "Naive Bayes";
	}
}
