package org.mg.wekalib.attribute_ranking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import org.mg.javalib.util.ArrayUtil;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class PredictionAttributeComputation
{
	public static List<PredictionAttributeImpl> compute(Classifier classifier, Instance instance,
			double[] distributionForInstance, HashMap<Integer, Set<Integer>> attributesToSwitch)
					throws Exception
	{
		int predictionIndex = ArrayUtil.getMaxIndex(distributionForInstance);

		List<PredictionAttributeImpl> attributes = new ArrayList<PredictionAttributeImpl>();
		for (int a = 0; a < instance.dataset().numAttributes(); a++)
		{
			if (instance.dataset().classIndex() == a)
				continue;
			Attribute att = instance.dataset().attribute(a);

			//			System.out.println();
			//			System.out.println(att.name() + " = " + instance.stringValue(a));
			//			System.out.println("---");

			Instances dCopy = new Instances(instance.dataset());
			Instance copy = new SparseInstance(instance);
			copy.setDataset(dCopy);

			String newVal = instance.stringValue(att).equals("0") ? "1" : "0";
			copy.setValue(a, newVal);
			if (attributesToSwitch != null)
				for (Integer otherA : attributesToSwitch.get(a))
				{
					//					if (copy.stringValue(otherA).equals(newVal))
					//						System.out.println("other " + otherA + " already at " + newVal);
					//					else
					//						System.out.println("switching other " + otherA + " from "
					//								+ copy.stringValue(otherA) + " to " + newVal);
					copy.setValue(otherA, newVal);
				}

			double[] newDistri = classifier.distributionForInstance(copy);

			if (distributionForInstance.length != 2)
				throw new IllegalStateException("not binary classification");
			if (Math.abs((1 - distributionForInstance[0]) - distributionForInstance[1]) > 0.0001)
				throw new IllegalStateException("sum not 1");

			//			System.out.println(att.name() + " = " + copy.stringValue(a) + ": " + ArrayUtil.toString(newDistri));

			double newProp = newDistri[predictionIndex];
			double diff = Math.abs(distributionForInstance[predictionIndex] - newProp);

			PredictionAttributeImpl pa = new PredictionAttributeImpl(a, newDistri, diff);
			attributes.add(pa);
		}
		//		System.out.println();
		//		System.out.println(instance);

		Collections.sort(attributes, new Comparator<PredictionAttributeImpl>()
		{
			@Override
			public int compare(PredictionAttributeImpl o1, PredictionAttributeImpl o2)
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
