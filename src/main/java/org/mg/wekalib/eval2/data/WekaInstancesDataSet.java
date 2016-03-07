package org.mg.wekalib.eval2.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.codec.digest.DigestUtils;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class WekaInstancesDataSet extends AbstractDataSet implements Serializable
{
	private static final long serialVersionUID = 2L;

	Instances inst;
	String key;
	int positiveClass;
	List<String> endpoints;

	public WekaInstancesDataSet(Instances inst, int positiveClass)
	{
		this.inst = inst;
		this.positiveClass = positiveClass;

		// avoid toString() which produces arff and takes too long on large datasets
		//key = DigestUtils.md5Hex(inst.toString() + "#" + positiveClass);
		StringBuffer keyStr = new StringBuffer(
				inst.numInstances() + "#" + inst.numAttributes() + "#" + positiveClass + "#");
		for (int i = 0; i < inst.numAttributes(); i++)
			keyStr.append(inst.attribute(i).name() + "#" + inst.attribute(i).numValues() + "#");
		key = DigestUtils.md5Hex(keyStr.toString());
	}

	@Override
	public int getPositiveClass()
	{
		return positiveClass;
	}

	@Override
	public String getName()
	{
		return inst.relationName();
	}

	@Override
	public String getKeyContent()
	{
		return getKeyContent(key);
	}

	@Override
	public Instances getWekaInstances()
	{
		return inst;
	}

	@Override
	public int getSize()
	{
		return inst.numInstances();
	}

	@Override
	public List<String> getEndpoints()
	{
		if (endpoints == null)
		{
			endpoints = new ArrayList<>();
			for (Instance instance : inst)
				endpoints.add(instance.stringValue(inst.classAttribute()));
		}
		return endpoints;
	}

	@Override
	public DataSet getFilteredDataset(String name, List<Integer> idx)
	{
		ArrayList<Attribute> attributes = new ArrayList<>();
		for (int i = 0; i < inst.numAttributes(); i++)
			attributes.add(inst.attribute(i));
		Instances data = new Instances(name, attributes, 0);
		data.setClassIndex(inst.classIndex());
		for (Integer i : idx)
			data.add(inst.get(i));
		return new WekaInstancesDataSet(data, positiveClass);
	}

}
