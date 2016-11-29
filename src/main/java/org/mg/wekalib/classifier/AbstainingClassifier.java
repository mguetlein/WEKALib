package org.mg.wekalib.classifier;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

public class AbstainingClassifier extends SingleClassifierEnhancer implements OptionHandler
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 2L;

	public static final char OPTION_MIN_PROB = 'P';
	public static final double DEFAULT_MIN_PROB = 0.66;

	public double minProb = DEFAULT_MIN_PROB;

	@Override
	public void buildClassifier(Instances data) throws Exception
	{
		getClassifier().buildClassifier(data);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		double d[] = getClassifier().distributionForInstance(instance);
		double max = 0;
		for (double v : d)
			if (v > max)
				max = v;
		if (max >= minProb)
			return d;
		else
			return new double[d.length];
	}

	@Override
	public String[] getOptions()
	{
		Vector<String> result = new Vector<String>();
		result.add("-" + OPTION_MIN_PROB);
		result.add("" + minProb);
		Collections.addAll(result, super.getOptions());
		return result.toArray(new String[result.size()]);
	}

	@Override
	public void setOptions(String[] options) throws Exception
	{
		String tmpStr = Utils.getOption(OPTION_MIN_PROB, options);
		if (tmpStr.length() != 0)
			minProb = Double.valueOf(tmpStr);
		else
			minProb = DEFAULT_MIN_PROB;
		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
	}

	@Override
	public Enumeration<Option> listOptions()
	{
		Vector<Option> newVector = new Vector<Option>();
		newVector.addElement(new Option("\tMin probability to not abstain from classification.",
				OPTION_MIN_PROB + "", 1, "-" + OPTION_MIN_PROB + " <min prob>"));
		newVector.addAll(Collections.list(super.listOptions()));
		return newVector.elements();
	}

}
