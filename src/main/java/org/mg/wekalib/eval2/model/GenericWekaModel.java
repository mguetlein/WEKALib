package org.mg.wekalib.eval2.model;

import java.util.Arrays;

import org.mg.javalib.util.ArrayUtil;

import weka.classifiers.Classifier;
import weka.core.OptionHandler;

public class GenericWekaModel extends AbstractModel
{
	String wekaClassifierClassName;
	String options[];

	public GenericWekaModel()
	{
	}

	public GenericWekaModel(Classifier c)
	{
		setWekaClassiferClassName(c.getClass().getName());
		if (c instanceof OptionHandler)
			setOptions(((OptionHandler) c).getOptions());
	}

	public void setWekaClassiferClassName(String c)
	{
		this.wekaClassifierClassName = c;
	}

	public void setOptions(String[] options)
	{
		this.options = options;
	}

	@Override
	public String getName()
	{
		return getAlgorithmShortName();
	}

	@Override
	public String getAlgorithmShortName()
	{
		return wekaClassifierClassName.substring(wekaClassifierClassName.lastIndexOf('.') + 1) + " "
				+ (options == null ? "" : (" " + ArrayUtil.toString(options)));
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return wekaClassifierClassName
				+ (options == null ? "" : (" " + ArrayUtil.toString(options)));
	}

	@Override
	public Classifier getWekaClassifer()
	{
		try
		{
			Classifier c = (Classifier) Class.forName(wekaClassifierClassName).newInstance();
			if (options != null)
				((OptionHandler) c).setOptions(Arrays.copyOf(options, options.length));
			return c;
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	@Override
	public boolean isFast()
	{
		return true;
	}

	@Override
	protected String getParamKey()
	{
		return wekaClassifierClassName
				+ (options == null ? "" : (" " + ArrayUtil.toString(options)));
	}

	@Override
	protected void cloneParams(Model clonedModel)
	{
		((GenericWekaModel) clonedModel).setWekaClassiferClassName(wekaClassifierClassName);
		((GenericWekaModel) clonedModel).setOptions(Arrays.copyOf(options, options.length));
	}

}
