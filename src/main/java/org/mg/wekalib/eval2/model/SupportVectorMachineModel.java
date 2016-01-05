package org.mg.wekalib.eval2.model;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

public class SupportVectorMachineModel extends AbstractModel
{
	private Kernel kernel = new PolyKernel();
	private double c = 1.0;
	private double gamma = 0.01;
	private double exp = 1;
	private boolean buildLogisticModels = true;

	@Override
	public Classifier getWekaClassifer()
	{
		try
		{
			SMO smo = new SMO();
			smo.setC(c);
			smo.setBuildLogisticModels(buildLogisticModels);
			Kernel kernel = this.kernel.getClass().newInstance();
			if (kernel instanceof PolyKernel)
				((PolyKernel) kernel).setExponent(exp);
			else if (kernel instanceof RBFKernel)
				((RBFKernel) kernel).setGamma(gamma);
			smo.setKernel(kernel);
			return smo;
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	@Override
	public String getName()
	{
		StringBuffer b = new StringBuffer();
		b.append("SMV c" + c + " l" + buildLogisticModels + " ");
		if (kernel instanceof PolyKernel)
			b.append("poly e" + exp);
		else if (kernel instanceof RBFKernel)
			b.append("rbf g" + gamma);
		return b.toString();

	}

	public void setC(double c)
	{
		this.c = c;
	}

	public void setGamma(double gamma)
	{
		this.gamma = gamma;
	}

	public void setExp(double exp)
	{
		this.exp = exp;
	}

	public void setKernel(Kernel kernel)
	{
		this.kernel = kernel;
	}

	public void setBuildLogisticModels(boolean buildLogisticModels)
	{
		this.buildLogisticModels = buildLogisticModels;
	}

	@Override
	protected void cloneParams(Model clonedModel)
	{
		SupportVectorMachineModel m = (SupportVectorMachineModel) clonedModel;
		m.setC(c);
		m.setBuildLogisticModels(buildLogisticModels);
		m.setGamma(gamma);
		m.setExp(exp);
		m.setKernel(kernel);
	}

	@Override
	public String getParamKey()
	{
		StringBuffer b = new StringBuffer();
		b.append(kernel.getClass().getSimpleName());
		b.append('#');
		b.append(c);
		b.append('#');
		b.append(gamma);
		b.append('#');
		b.append(exp);
		b.append('#');
		b.append(buildLogisticModels);
		return b.toString();
	}

}
