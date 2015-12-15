package org.mg.wekalib.data;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class InstanceUtil
{
	public static Instances stripAttributes(Instances data, List<Attribute> attr) throws Exception
	{
		return getAttributes(data, attr, false);
	}

	public static Instances getAttributes(Instances data, List<Attribute> attr) throws Exception
	{
		return getAttributes(data, attr, true);
	}

	private static Instances getAttributes(Instances data, List<Attribute> attr, boolean invert) throws Exception
	{
		String rem = "";
		for (int i = 0; i < data.numAttributes(); i++)
		{
			boolean c = attr.contains(data.attribute(i));
			if ((c && !invert) || (!c && invert))
			{
				if (!rem.isEmpty())
					rem += ",";
				rem += (i + 1);
			}
		}
		Remove remove = new Remove();
		remove.setAttributeIndices(rem);
		remove.setInputFormat(data);
		return Filter.useFilter(data, remove);
	}

	public static void attachAttributes(Instances data, List<Attribute> attr, Instances attrData, boolean end)
	{
		int startInsert = end ? data.numAttributes() : 0;
		List<Attribute> newAttributes = new ArrayList<>();
		for (int a = 0; a < attr.size(); a++)
		{
			data.insertAttributeAt(attr.get(a), startInsert + a);
			newAttributes.add(data.attribute(startInsert + a));
		}
		for (int a = 0; a < attr.size(); a++)
			for (int i = 0; i < data.numInstances(); i++)
			{
				double v = attrData.instance(i).value(attr.get(a));
				data.instance(i).setValue(newAttributes.get(a), v);
			}
	}

	public static void attachNominalAttribute(Instances data, String name, List<String> domain, List<String> values,
			boolean end)
	{
		int idx = end ? data.numAttributes() : 0;
		Attribute attr = new Attribute(name, domain);
		data.insertAttributeAt(attr, idx);
		Attribute newAttr = data.attribute(idx);
		for (int i = 0; i < data.numInstances(); i++)
			if (values.get(i) == null)
				data.instance(i).setMissing(newAttr);
			else
				data.instance(i).setValue(newAttr, values.get(i));
	}

	public static void attachNumericAttribute(Instances data, String name, List<Double> values, boolean end)
	{
		int idx = end ? data.numAttributes() : 0;
		Attribute attr = new Attribute(name);
		data.insertAttributeAt(attr, idx);
		Attribute newAttr = data.attribute(idx);
		for (int i = 0; i < data.numInstances(); i++)
			data.instance(i).setValue(newAttr, values.get(i));
	}

	public static void concatColumns(Instances data, Instances concatData)
	{
		List<Attribute> newAttributes = new ArrayList<>();
		for (int a = 0; a < concatData.numAttributes(); a++)
		{
			data.insertAttributeAt(concatData.attribute(a), data.numAttributes());
			newAttributes.add(data.attribute(data.numAttributes() - 1));
		}
		for (int a = 0; a < newAttributes.size(); a++)
			for (int i = 0; i < data.numInstances(); i++)
			{
				double v = concatData.instance(i).value(a);
				data.instance(i).setValue(newAttributes.get(a), v);
			}
	}

	public static void main(String[] args) throws Exception
	{
		Instances inst = new Instances(new FileReader("/home/martin/workspace/external/weka-3-7-12/data/iris.arff"));
		//		Instances inst2 = new Instances(new FileReader("/home/martin/workspace/external/weka-3-7-12/data/glass.arff"));

		while (inst.size() > 10)
			inst.remove(10);
		//		while (inst2.size() > 10)
		//			inst2.remove(10);

		//		System.out.println(inst);
		//		System.out.println(inst2);

		List<Attribute> attr = new ArrayList<>();
		attr.add(inst.attribute("sepallength"));
		attr.add(inst.attribute("sepalwidth"));

		System.out.println(inst + "\n\n");

		Instances inst_ = stripAttributes(inst, attr);
		System.out.println(inst_ + "\n\n");

		Instances inst__ = getAttributes(inst, attr);
		System.out.println(inst__ + "\n\n");

		concatColumns(inst_, inst__);
		System.out.println(inst_ + "\n\n");

		//		attachAttributes(inst_, attr, inst, false);
		//		System.out.println(inst_);
	}

}
