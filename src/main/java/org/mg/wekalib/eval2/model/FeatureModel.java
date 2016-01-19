package org.mg.wekalib.eval2.model;

import java.io.File;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.eval2.job.FeatureProvider;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekalib.evaluation.Predictions;

public class FeatureModel extends DefaultJobOwner<Predictions> implements Model
{
	FeatureProvider featureProvider;
	Model model;
	DataSet train;
	DataSet test;

	@Override
	public String getName()
	{
		return "FeatureModel (" + featureProvider.getName() + ", " + model.getName() + ")";
	}

	@Override
	public String getKeyPrefix()
	{
		return "FeatureModel" + File.separator + featureProvider.getKeyPrefix() + File.separator
				+ model.getKeyPrefix()
				+ ((train != null && featureProvider.getTrainingDataset() == null
						&& model.getTrainingDataset() == null)
								? (File.separator + train.getKeyPrefix()) : "");
	}

	@Override
	public String getKeyContent()
	{
		return getKeyContent(train, test, featureProvider, model);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		FeatureProvider feat = (FeatureProvider) featureProvider.cloneJob();
		feat.setTrainingDataset(train);
		feat.setTestDataset(test);
		if (!feat.isDone())
			return Printer.wrapRunnable("FeatureModel: compute features " + getName(),
					feat.nextJob());
		else
		{
			Model mod = (Model) model.cloneJob();
			DataSet res[] = feat.getResult();
			mod.setTrainingDataset(res[0]);
			mod.setTestDataset(res[1]);
			if (!mod.isDone())
			{
				Runnable r = mod.nextJob();
				if (r == null)
					return null;
				else
					// to avoid having a lot of jobs that only store results, this jobs are concated
					// does only work if model is a single job model
					return Printer.wrapRunnable("FeatureModel: build model " + getName(), r,
							storeResults(mod));
			}
			else
			{
				// model is done, but results are missing (could happen if model has been invoked directly)
				return storeResults(mod);
			}
		}
	}

	private Runnable storeResults(final Model m)
	{
		return blockedJob("FeatureModel: storing results", new Runnable()
		{
			public void run()
			{
				Predictions p = m.getResult();
				//System.err.println(PredictionUtil.summaryClassification(p));
				setResult(p);
			};
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

	public Model getModel()
	{
		return model;
	}

	public void setFeatureProvider(FeatureProvider featureProvider)
	{
		this.featureProvider = featureProvider;
	}

	public FeatureProvider getFeatureProvider()
	{
		return featureProvider;
	}

	@Override
	public void setTrainingDataset(DataSet train)
	{
		this.train = train;
	}

	@Override
	public DataSet getTrainingDataset()
	{
		return train;
	}

	@Override
	public void setTestDataset(DataSet test)
	{
		this.test = test;
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return model.getAlgorithmParamsNice();
	}

	@Override
	public String getAlgorithmShortName()
	{
		return model.getAlgorithmShortName();
	}

}
