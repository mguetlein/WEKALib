package org.mg.wekalib.eval2.model;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.eval2.job.FeatureProvider;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekautil.Predictions;

public class FeatureModel extends DefaultJobOwner<Predictions> implements Model
{
	FeatureProvider featureProvider;
	Model model;
	DataSet train;
	DataSet test;

	@Override
	public String getKey()
	{
		return getKey(featureProvider, model, train, test);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		FeatureProvider feat = (FeatureProvider) featureProvider.cloneJob();
		feat.setTrainingDataset(train);
		feat.setTestDataset(test);
		if (!feat.isDone())
			return Printer.wrapRunnable("FeatureModel: compute features", feat.nextJob());

		final Model mod = (Model) model.cloneJob();
		DataSet res[] = feat.getResult();
		mod.setTrainingDataset(res[0]);
		mod.setTestDataset(res[1]);
		if (!mod.isDone())
			return Printer.wrapRunnable("FeatureModel: build model", mod.nextJob());

		return blockedJob("FeatureModel: storing results", new Runnable()
		{
			@Override
			public void run()
			{
				Predictions p = mod.getResult();
				//				System.err.println(PredictionUtil.summaryClassification(p));
				setResult(p);
			}
		});
	}

	@Override
	public Model cloneJob()
	{
		FeatureModel fm = new FeatureModel();
		fm.featureProvider = (FeatureProvider) featureProvider.cloneJob();
		fm.model = (Model) model.cloneJob();
		fm.train = train;
		fm.test = test;
		return fm;
	}

	public void setModel(Model model)
	{
		this.model = model;
	}

	public void setFeatureProvider(FeatureProvider featureProvider)
	{
		this.featureProvider = featureProvider;
	}

	@Override
	public void setTrainingDataset(DataSet train)
	{
		this.train = train;
	}

	@Override
	public void setTestDataset(DataSet test)
	{
		this.test = test;
	}

	@Override
	public String getName()
	{
		return "FeatureModel (" + featureProvider.getName() + ", " + model.getName() + ")";
	}

}
