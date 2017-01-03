package org.mg.wekalib.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.mg.wekalib.data.InstanceUtil;

import weka.core.Instances;
import weka.core.converters.CSVSaver;

public class ArffToCSV
{
	public static void toCSV(Instances inst, String outfile)
	{
		try
		{
			toCSV(inst, new FileOutputStream(new File(outfile)));
		}
		catch (IOException e)
		{
			throw new RuntimeException(e);
		}
	}

	public static void toCSV(Instances inst, OutputStream os)
	{
		try
		{
			CSVSaver s = new CSVSaver();
			s.setInstances(inst);
			s.setDestination(os);
			//s.setFile(new File(outfile));
			s.writeBatch();
		}
		catch (IOException e)
		{
			throw new RuntimeException(e);
		}
	}

	public static void main(String[] args) throws Exception
	{
		Instances inst = new Instances(new FileReader("/home/martin/data/weka/nominal/iris.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);

		String name = inst.classAttribute().name();
		Double values[] = ArrayUtils.toObject(inst.attributeToDoubleArray(inst.classIndex()));
		inst = InstanceUtil.stripAttributes(inst, Arrays.asList(inst.classAttribute()));
		InstanceUtil.attachNumericAttribute(inst, name, Arrays.asList(values), true);

		ArffToCSV.toCSV(inst, "/tmp/iris2.csv");
	}
}
