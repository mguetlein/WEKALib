package org.mg.wekalib.appdomain;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class DistanceBasedADClassifier extends AbstractClassifier
{
	private static final long serialVersionUID = 1L;

	DistanceBasedADModel ad;
	int positiveClass;

	public DistanceBasedADClassifier(DistanceBasedADModel ad, int positiveClass)
	{
		this.ad = ad;
		this.positiveClass = positiveClass;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception
	{
		if (data.numClasses() != 2)
			throw new IllegalStateException();
		ad.build(data);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		// predicted distribution is not class probability
		// instead the similarity to the training dataset is returned
		// the similarity will be scaled between
		// 0.5 := greater-or-equal to max distance and 1.0 := very close to training dataset
		// the similarity is stored in the positive class
		// the negative class will be 1 - the similarity and always be below 0.5

		double distance = ad.computeDistance(instance) / (double) ad.getMaxTrainingDistance();
		if (distance > 1 || distance < 0)
			throw new IllegalArgumentException(distance + "");
		double similarity = 1.0 - (distance * 0.5);
		if (similarity > 1 || similarity < 0.5)
			throw new IllegalArgumentException(similarity + "");
		double negative = 1 - similarity;
		if (positiveClass == 0)
			return new double[] { similarity, negative };
		else if (positiveClass == 1)
			return new double[] { negative, similarity };
		else
			throw new IllegalStateException();
	}

}
