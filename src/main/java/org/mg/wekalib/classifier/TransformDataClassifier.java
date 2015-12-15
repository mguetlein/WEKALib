package org.mg.wekalib.classifier;

import java.util.ArrayList;

import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public abstract class TransformDataClassifier extends SingleClassifierEnhancer
{
	public abstract Instances transformData(Instances data, boolean train) throws Exception;

	@Override
	public void buildClassifier(Instances data) throws Exception
	{
		Instances pca = transformData(data, true);
		m_Classifier.buildClassifier(pca);
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception
	{
		ArrayList<Attribute> att = new ArrayList<>();
		for (int i = 0; i < instance.numAttributes(); i++)
			att.add(instance.attribute(i));
		Instances inst = new Instances("test", att, 1);
		inst.add(instance);
		Instances pca = transformData(inst, false);
		return m_Classifier.classifyInstance(pca.instance(0));
	}
}