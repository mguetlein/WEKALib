package org.mg.wekalib.data;

import java.io.FileReader;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;

public class ArffWritableTest
{
	@Test
	public void test() throws Exception
	{
		for (boolean toSparse : new boolean[] { false, true })
		{
			for (String data : new String[] { "nominal/sonar", "numeric/cloud",
					"numeric/pwLinear" })
			{
				Instances instances = new Instances(new FileReader(
						System.getProperty("user.home") + "/data/weka/" + data + ".arff"));
				final Instances inst;
				if (toSparse)
				{
					NonSparseToSparse sp = new NonSparseToSparse();
					sp.setInputFormat(instances);
					inst = Filter.useFilter(instances, sp);
				}
				else
					inst = instances;

				ArffWritable arff = new ArffWritable()
				{
					@Override
					public boolean isSparse()
					{
						return inst.get(0) instanceof SparseInstance;
					}

					@Override
					public String getRelationName()
					{
						return inst.relationName();
					}

					@Override
					public int getNumInstances()
					{
						return inst.numInstances();
					}

					@Override
					public int getNumAttributes()
					{
						return inst.numAttributes();
					}

					@Override
					public String getMissingValue(int attribute)
					{
						return null;
					}

					@Override
					public String[] getAttributeDomain(int attribute)
					{
						Attribute a = inst.attribute(attribute);
						if (a.isNumeric())
							return null;
						else
						{
							String d[] = new String[a.numValues()];
							for (int i = 0; i < a.numValues(); i++)
								d[i] = a.value(i);
							return d;
						}
					}

					@Override
					public double getAttributeValueAsDouble(int instance, int attribute)
							throws Exception
					{
						return inst.get(instance).value(attribute);
					}

					@Override
					public String getAttributeValue(int instance, int attribute) throws Exception
					{
						if (inst.get(instance).isMissing(attribute))
							return null;
						else if (inst.attribute(attribute).isNumeric())
							return String.valueOf(inst.get(instance).value(attribute));
						else
							return inst.get(instance).stringValue(attribute);
					}

					@Override
					public String getAttributeName(int attribute)
					{
						return inst.attribute(attribute).name();
					}

					@Override
					public List<String> getAdditionalInfo()
					{
						return null;
					}
				};

				if (!toSparse)
					instancesEqual(inst, ArffWriter.toInstances(arff));
				instancesEqual(inst, InstancesCreator.create(arff));
			}
		}
	}

	public static void instancesEqual(Instances inst, Instances inst2)
	{
		Assert.assertEquals(inst.relationName(), inst2.relationName());
		Assert.assertEquals(inst.numAttributes(), inst2.numAttributes());
		Assert.assertEquals(inst.numInstances(), inst2.numInstances());
		for (int a = 0; a < inst.numAttributes(); a++)
			Assert.assertEquals(inst.attribute(a).name(), inst2.attribute(a).name());
		for (int i = 0; i < inst.numInstances(); i++)
		{
			for (int a = 0; a < inst.numAttributes(); a++)
			{
				Assert.assertEquals(inst.get(i).value(a), inst2.get(i).value(a), 0.0);
			}
		}
	}

	//	public static void main(String[] args) throws Exception
	//	{
	//		Instances inst = new Instances(new FileReader(
	//				System.getProperty("user.home") + "/data/weka/numeric/pwLinear.arff"));
	//		System.out.println(inst);
	//		NonSparseToSparse sp = new NonSparseToSparse();
	//		sp.setInputFormat(inst);
	//		inst = Filter.useFilter(inst, sp);
	//		System.out.println(inst);
	//	}
}
