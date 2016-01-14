package org.mg.wekalib.attribute_ranking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.mg.javalib.util.ArrayUtil;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PredictionAttributeComputation
{

	public static Set<Integer> allAttributes(Instance instance)
	{
		Set<Integer> atts = new HashSet<>();
		for (int i = 0; i < instance.dataset().numAttributes(); i++)
			if (instance.dataset().classIndex() != i)
				atts.add(i);
		return atts;
	}

	public static List<PredictionAttribute> compute(Classifier classifier, Instance instance,
			double[] distributionForInstance, Set<Integer> attributesUsedForPrediction)
					throws Exception
	{

		int predictionIndex = ArrayUtil.getMaxIndex(distributionForInstance);

		List<PredictionAttribute> attributes = new ArrayList<PredictionAttribute>();
		for (Integer a : attributesUsedForPrediction)
		{
			Attribute att = instance.dataset().attribute(a);

			//			System.out.println();
			//			System.out.println(att.name() + " = " + instance.stringValue(a));
			//			System.out.println("---");

			Instances dCopy = new Instances(instance.dataset());
			Instance copy = new DenseInstance(instance);
			copy.setDataset(dCopy);
			copy.setValue(a, instance.stringValue(att).equals("0") ? "1" : "0");

			double[] newDistri = classifier.distributionForInstance(copy);
			if (distributionForInstance.length != 2)
				throw new IllegalStateException("not binary classification");
			if (Math.abs((1 - distributionForInstance[0]) - distributionForInstance[1]) > 0.0001)
				throw new IllegalStateException("sum not 1");

			//			System.out.println(att.name() + " = " + copy.stringValue(a) + ": " + ArrayUtil.toString(newDistri));

			double newProp = newDistri[predictionIndex];
			double diff = Math.abs(distributionForInstance[predictionIndex] - newProp);

			PredictionAttribute pa = new PredictionAttribute(a, newDistri, diff);
			attributes.add(pa);
		}
		//		System.out.println();
		//		System.out.println(instance);

		Collections.sort(attributes, new Comparator<PredictionAttribute>()
		{
			@Override
			public int compare(PredictionAttribute o1, PredictionAttribute o2)
			{
				return Double.valueOf(o2.getDiffToOrigProp())
						.compareTo(Double.valueOf(o1.getDiffToOrigProp()));
			}
		});

		//		System.out.println("orig prob: " + distributionForInstance[predictionIndex]);
		//		for (PredictionAttribute pa : attributes)
		//		{
		//			System.out.println(instance.dataset().attribute(pa.attribute).name() + " " + pa.attribute + " : newProp "
		//					+ StringUtil.formatDouble(pa.alternativeDistributionForInstance[pa.alternativePredictionIdx])
		//					+ " diff: " + StringUtil.formatDouble(pa.diffToOrigProp));
		//		}

		return attributes;
	}

}
