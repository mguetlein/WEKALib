package org.mg.wekalib.classifier;

import java.util.HashMap;
import java.util.Map;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class SameValueLocalModel extends LocalModel
{
	String attrName;

	Map<Double, Integer> vals = new HashMap<>();

	public SameValueLocalModel(String attrName)
	{
		this.attrName = attrName;
	}

	@Override
	public LocalModelClusterer getClusterer(Instances data)
	{
		final Attribute attr = data.attribute(attrName);
		return new LocalModelClusterer()
		{
			@Override
			public int clusterIdx(Instance inst)
			{
				double val = inst.value(attr);
				if (!vals.containsKey(val))
					vals.put(val, vals.size());
				return vals.get(val);
			}
		};
	}
}
