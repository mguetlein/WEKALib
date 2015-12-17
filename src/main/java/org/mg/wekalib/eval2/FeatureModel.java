package org.mg.wekalib.eval2;

import org.mg.javalib.util.HashUtil;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekautil.Predictions;

public class FeatureModel extends DefaultJobOwner<Predictions> implements Model
{
	private static final long serialVersionUID = 1L;

	FeatureProvider featureProvider;
	Model model;
	DataSet train;
	DataSet test;

	@Override
	public int hashCode()
	{
		return HashUtil.hashCode(featureProvider, model, train, test);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		FeatureProvider feat = featureProvider.cloneFeatureProvider();
		feat.setTrainingDataset(train);
		feat.setTestDataset(test);
		if (!feat.isDone())
			return feat.nextJob();

		final Model mod = model.cloneModel();
		DataSet res[] = feat.getResult();
		mod.setTrainingDataset(res[0]);
		mod.setTestDataset(res[1]);
		if (!mod.isDone())
			return mod.nextJob();

		if (!Blocker.block(hashCode()))
			return null;
		return new Runnable()
		{
			@Override
			public void run()
			{
				System.out.println(FeatureModel.this.hashCode() + " storing feature model results");
				Predictions p = mod.getResult();
				System.err.println(PredictionUtil.summaryClassification(p));
				setResult(p);
				Blocker.unblock(FeatureModel.this.hashCode());
			}
		};
	}

	@Override
	public Model cloneModel()
	{
		FeatureModel fm = new FeatureModel();
		fm.featureProvider = featureProvider.cloneFeatureProvider();
		fm.model = model.cloneModel();
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
