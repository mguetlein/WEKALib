package org.mg.wekalib.eval2.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.codec.digest.DigestUtils;

import weka.core.Attribute;
import weka.core.Instances;

public class WekaInstancesDataSet extends AbstractDataSet implements Serializable
{
	private static final long serialVersionUID = 1L;

	Instances inst;
	String key;

	public WekaInstancesDataSet(Instances inst)
	{
		this.inst = inst;
		key = DigestUtils.md5Hex(inst.toString());
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
	public DataSet getFilteredDataset(String name, List<Integer> idx)
	{
		ArrayList<Attribute> attributes = new ArrayList<>();
		for (int i = 0; i < inst.numAttributes(); i++)
			attributes.add(inst.attribute(i));
		Instances data = new Instances(name, attributes, 0);
		data.setClassIndex(inst.classIndex());
		for (Integer i : idx)
			data.add(inst.get(i));
		return new WekaInstancesDataSet(data);
	}

}
