package org.mg.wekalib.eval2;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.job.DataSetJobOwner;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.eval2.model.Model;
import org.mg.wekalib.evaluation.Predictions;

public abstract class Validation extends DefaultJobOwner<Predictions>
		implements DataSetJobOwner<Predictions>
{
	protected DataSet dataSet;
	protected Model model;
	protected long randomSeed = 1;
	protected boolean stratified = false;

	public Model getModel()
	{
		return model;
	}

	public void setModel(Model mod)
	{
		model = mod;
	}

	@Override
	public void setDataSet(DataSet data)
	{
		dataSet = data;
	}

	public DataSet getDataSet()
	{
		return dataSet;
	}

	public long getRandomSeed()
	{
		return randomSeed;
	}

	public void setRandomSeed(long r)
	{
		randomSeed = r;
	}

	public void setStratified(boolean stratified)
	{
		this.stratified = stratified;
	}

	public boolean isStratified()
	{
		return stratified;
	}

	public boolean isValid()
	{
		return model.isValid(dataSet);
	}
}
