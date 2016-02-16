package org.mg.wekalib.data;

import java.util.ArrayList;

import org.mg.javalib.util.ArrayUtil;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class InstancesCreator
{
	public static Instances create(ArffWritable w) throws Exception
	{
		ArrayList<Attribute> a = new ArrayList<Attribute>();
		for (int i = 0; i < w.getNumAttributes(); i++)
		{
			String d[] = w.getAttributeDomain(i);
			if (d == null)
				a.add(new Attribute(w.getAttributeName(i)));
			else
				a.add(new Attribute(w.getAttributeName(i), ArrayUtil.toList(d)));
		}
		Instances data = new Instances(w.getRelationName(), a, w.getNumInstances());
		//data.setClassIndex(a.size() - 1);

		for (int i = 0; i < w.getNumInstances(); i++)
		{
			double vals[] = new double[data.numAttributes()];
			for (int j = 0; j < data.numAttributes(); j++)
				vals[j] = w.getAttributeValueAsDouble(i, j);
			Instance inst;
			if (w.isSparse())
				inst = new SparseInstance(1.0, vals);
			else
				inst = new DenseInstance(1.0, vals);
			inst.setDataset(data);

			data.add(inst);
		}

		return data;
	}
}
