package org.mg.wekalib.classifier;

import java.util.HashMap;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;

public abstract class LocalModel extends SingleClassifierEnhancer
{
	transient HashMap<Integer, Classifier> classifiers;

	transient Classifier allDataClassifier;

	LocalModelClusterer clusterer;

	Instances train;

	public abstract LocalModelClusterer getClusterer(Instances data);

	@Override
	public void buildClassifier(Instances data) throws Exception
	{
		this.train = data;
		this.clusterer = getClusterer(train);
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception
	{
		int idx = clusterer.clusterIdx(instance);
		if (classifiers == null)
			classifiers = new HashMap<>();
		if (!classifiers.containsKey(idx))
		{
			Instances filtered = new Instances(train);
			for (int i = filtered.numInstances() - 1; i >= 0; i--)
				if (clusterer.clusterIdx(filtered.get(i)) != idx)
					filtered.remove(i);
			Classifier classi;
			if (filtered.size() == 0)
			{
				System.err.println("local model found no instance, using entire data");
				if (allDataClassifier == null)
				{
					allDataClassifier = AbstractClassifier.makeCopy(getClassifier());
					allDataClassifier.buildClassifier(train);
				}
				classi = allDataClassifier;
			}
			else if (filtered.size() <= 2)
			{
				System.err.println("local model found 1 or 2 instances, using zeroR");
				classi = new ZeroR();
				classi.buildClassifier(filtered);
			}
			else
			{
				classi = AbstractClassifier.makeCopy(getClassifier());
				//			System.out.println();
				//			System.out.println(instance);
				//			System.out.println();
				//System.out.println("building local model with " + filtered.size() + " instances");
				//			System.out.println();
				//			System.out.println(filtered);
				//			System.out.println();
				classi.buildClassifier(filtered);
			}
			classifiers.put(idx, classi);
		}
		return classifiers.get(idx).classifyInstance(instance);
	}
}
